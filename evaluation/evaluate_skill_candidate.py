"""STEP 4のSkills Candidateをcombined-v2.4.0で評価する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from tech_blog_mlflow.article_v3_checks import (
    ARTICLE_MAX_CHARS,
    ARTICLE_MIN_CHARS,
    article_checks,
)


GENERATION_RESULTS_DIR = Path(
    "generation_results"
)

GENERATION_PATTERN = (
    "skill_v1_generation_*.json"
)

EXPECTED_PROMPT_VERSION = (
    "article-v3.5.2"
)

EXPECTED_CONFIG_VERSION = (
    "generation-v4.0-skills-v1"
)

EXPECTED_SKILL_NAME = "technical-blog-quality"
EXPECTED_SKILL_VERSION = "technical-blog-quality-v1"
EXPECTED_GENERATION_PARAMETERS = {
    "max_tokens": 4096,
    "temperature": 0.0,
    "seed": 42,
    "enable_thinking": False,
}
ADOPTED_BASE_PROMPT_SHA256 = (
    "19416090fd3ddd3de09b354ccf125247c248125d915da3191849ce4350781c89"
)


def text_sha256(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Skills-v1の生成Metadataから"
            "記事を特定し、combined-v2.4.0で"
            "評価する"
        )
    )

    parser.add_argument(
        "--generation-json",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--run-name",
        default=(
            "skill-v1-evaluation-v2.4.0"
        ),
    )

    return parser.parse_args()


def latest_generation_json() -> Path:
    candidates = list(
        GENERATION_RESULTS_DIR.glob(
            GENERATION_PATTERN
        )
    )

    if not candidates:
        raise FileNotFoundError(
            "Skills-v1のGeneration JSONが"
            "ありません。先に"
            "generate_with_skillを実行してください。"
        )

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime,
    )


def load_metadata(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Generation JSONがありません: {path}"
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    required_keys = {
        "run_id",
        "article_path",
        "article_sha256",
        "base_prompt_sha256",
        "rendered_prompt_path",
        "prompt_sha256",
        "formatted_prompt_sha256",
        "system_prompt_sha256",
        "prompt_version",
        "generation_config_version",
        "generation_parameters",
        "metrics",
        "prechecks",
        "all_prechecks_passed",
        "failed_prechecks",
        "skill",
        "controlled_change",
    }

    missing = required_keys.difference(
        payload
    )

    if missing:
        raise KeyError(
            "Generation JSONの必須Keyが"
            f"ありません: {sorted(missing)}"
        )

    if (
        payload["prompt_version"]
        != EXPECTED_PROMPT_VERSION
    ):
        raise ValueError(
            "Prompt Versionが違います: "
            f"{payload['prompt_version']}"
        )

    if (
        payload["generation_config_version"]
        != EXPECTED_CONFIG_VERSION
    ):
        raise ValueError(
            "Generation Config Versionが違います: "
            f"{payload['generation_config_version']}"
        )

    if (
        payload["base_prompt_sha256"]
        != ADOPTED_BASE_PROMPT_SHA256
    ):
        raise ValueError(
            "Base Prompt SHA-256がSTEP 3採用値と一致しません。"
        )

    if (
        payload["generation_parameters"]
        != EXPECTED_GENERATION_PARAMETERS
    ):
        raise ValueError(
            "生成条件がSTEP 3採用値と一致しません: "
            f"{payload['generation_parameters']}"
        )

    skill = payload["skill"]
    expected_skill = {
        "enabled": True,
        "name": EXPECTED_SKILL_NAME,
        "version": EXPECTED_SKILL_VERSION,
    }
    for name, expected in expected_skill.items():
        if skill.get(name) != expected:
            raise ValueError(
                "Skill Metadataが想定値と一致しません: "
                f"{name}={skill.get(name)!r}"
            )

    skill_path = Path(str(skill.get("path", "")))
    if not skill_path.exists():
        raise FileNotFoundError(
            f"Skillがありません: {skill_path}"
        )
    skill_text = skill_path.read_text(encoding="utf-8")
    if text_sha256(skill_text) != skill.get("sha256"):
        raise ValueError(
            "Skill SHA-256がGeneration Metadataと一致しません。"
        )

    controlled = payload["controlled_change"]
    if not (
        controlled.get("name") == "skills"
        and controlled.get("baseline") is False
        and controlled.get("candidate") is True
        and controlled.get("base_prompt_changed") is False
        and controlled.get("max_tokens_changed") is False
    ):
        raise ValueError(
            "Skills以外の変更がGeneration Metadataに含まれています。"
        )

    if (
        controlled.get("base_prompt_sha256")
        != payload["base_prompt_sha256"]
        or controlled.get("skill_sha256")
        != skill.get("sha256")
    ):
        raise ValueError(
            "Controlled ChangeのSHA-256がMetadataと一致しません。"
        )

    if not payload["all_prechecks_passed"]:
        raise ValueError(
            "生成後の事前検査が失敗しています: "
            f"{payload['failed_prechecks']}"
        )

    article_path = Path(
        payload["article_path"]
    )

    if not article_path.exists():
        raise FileNotFoundError(
            f"記事がありません: {article_path}"
        )

    rendered_prompt_path = Path(
        payload["rendered_prompt_path"]
    )
    if not rendered_prompt_path.exists():
        raise FileNotFoundError(
            "Rendered Promptがありません: "
            f"{rendered_prompt_path}"
        )

    article = article_path.read_text(
        encoding="utf-8"
    )
    rendered_prompt = rendered_prompt_path.read_text(
        encoding="utf-8"
    )

    if text_sha256(article) != payload["article_sha256"]:
        raise ValueError(
            "記事SHA-256がGeneration Metadataと一致しません。"
        )

    if text_sha256(rendered_prompt) != payload["prompt_sha256"]:
        raise ValueError(
            "Rendered Prompt SHA-256がGeneration Metadataと一致しません。"
        )

    metrics = payload["metrics"]
    parameters = payload["generation_parameters"]

    if int(metrics["article_length"]) != len(article):
        raise ValueError(
            "記事文字数がGeneration Metadataと一致しません。"
        )

    recomputed = article_checks(
        article,
        rendered_prompt,
    )
    recomputed["output_tokens_below_safety_limit"] = int(
        metrics["output_tokens"]
    ) < int(parameters["max_tokens"]) - 64
    recomputed["article_length_in_range"] = (
        ARTICLE_MIN_CHARS
        <= len(article)
        <= ARTICLE_MAX_CHARS
    )

    if payload["prechecks"] != recomputed:
        raise ValueError(
            "Generation Metadataの事前検査結果を再現できません。"
        )

    failed = [
        name
        for name, passed in recomputed.items()
        if not passed
    ]
    if payload["failed_prechecks"] != failed:
        raise ValueError(
            "failed_prechecksが再計算結果と一致しません。"
        )

    if payload["all_prechecks_passed"] != (not failed):
        raise ValueError(
            "all_prechecks_passedが再計算結果と一致しません。"
        )

    return payload


def build_evaluation_command(
    *,
    metadata: dict[str, Any],
    run_name: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "evaluation.evaluate_combined_v2_4",
        "--article",
        str(metadata["article_path"]),
        "--source-run-id",
        str(metadata["run_id"]),
        "--variant",
        "skill-v1",
        "--generator-prompt-version",
        EXPECTED_PROMPT_VERSION,
        "--run-name",
        run_name,
    ]


def main() -> None:
    args = parse_args()

    generation_json = (
        args.generation_json
        if args.generation_json
        is not None
        else latest_generation_json()
    )

    metadata = load_metadata(
        generation_json
    )

    command = build_evaluation_command(
        metadata=metadata,
        run_name=args.run_name,
    )

    print("Generation JSON:", generation_json)
    print("Article        :", metadata["article_path"])
    print("Generation Run :", metadata["run_id"])
    print(
        "Generation Config:",
        metadata["generation_config_version"],
    )
    print("Skill          :", EXPECTED_SKILL_VERSION)
    print("Judge          : article-judge-v2.4")
    print()

    subprocess.run(
        command,
        check=True,
    )


if __name__ == "__main__":
    main()
