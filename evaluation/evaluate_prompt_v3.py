"""最新のPrompt-v3.4生成記事をcombined-v2.2で評価する。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


GENERATION_RESULTS_DIR = Path(
    "generation_results"
)

GENERATION_PATTERN = (
    "prompt_v3_4_generation_*.json"
)

EXPECTED_PROMPT_VERSION = (
    "article-v3.3"
)

EXPECTED_CONFIG_VERSION = (
    "generation-v3.4"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prompt-v3.4の生成Metadataから"
            "記事を特定し、combined-v2.2で"
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
            "prompt-v3.4-evaluation-v2.2"
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
            "Prompt-v3.4のGeneration JSONが"
            "ありません。先に"
            "generate_prompt_v3を実行してください。"
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
        "prompt_version",
        "generation_config_version",
        "all_prechecks_passed",
        "failed_prechecks",
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

    return payload


def build_evaluation_command(
    *,
    metadata: dict[str, Any],
    run_name: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "evaluation.evaluate_combined_v2_2",
        "--article",
        str(metadata["article_path"]),
        "--source-run-id",
        str(metadata["run_id"]),
        "--variant",
        "prompt-v3.4",
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
    print("Judge          : article-judge-v2.2")
    print()

    subprocess.run(
        command,
        check=True,
    )


if __name__ == "__main__":
    main()
