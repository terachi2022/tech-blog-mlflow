"""STEP 4のno-skill/skill-v1制御実験を比較する。"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from evaluation.compare_runs import (
    METRICS,
    comparable_versions,
    metric_value,
)
from evaluation.skill_experiment_checks import (
    build_skill_success_checks,
    llm_judge_mean,
)


TRACKING_URI = "http://127.0.0.1:5000"
ADOPTED_GENERATION_RUN_ID = (
    "b5c925c2322b4e30b04f07e24d160a04"
)
ADOPTED_EVALUATION_RUN_ID = (
    "fdf0c239445f44a0999a6b1fe7a419b6"
)
EXPECTED_PROMPT_VERSION = "article-v3.5.2"
EXPECTED_SKILL_VERSION = "technical-blog-quality-v1"

GENERATION_METRICS = (
    "generation_time_sec",
    "article_length",
    "output_tokens",
    "generation_tokens_per_sec",
    "peak_memory_gb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "STEP 3採用版とSkills-v1を比較し、"
            "Skillsだけの効果を判定する"
        )
    )
    parser.add_argument(
        "--baseline-generation-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--candidate-generation-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--baseline-evaluation-run-id",
        default=ADOPTED_EVALUATION_RUN_ID,
    )
    parser.add_argument(
        "--candidate-evaluation-run-id",
        required=True,
    )
    parser.add_argument(
        "--output-prefix",
        default="comparison_no_skill_vs_skill_v1",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"JSONがありません: {path}"
        )
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise TypeError(
            f"JSON RootはObjectである必要があります: {path}"
        )
    return payload


def safe_output_prefix(value: str) -> str:
    sanitized = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value,
    ).strip("._")
    if not sanitized:
        raise ValueError(
            "output-prefixが空です。"
        )
    return sanitized


def rounded(value: float) -> float:
    return round(float(value), 4)


def validate_controlled_change(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Skill以外の主要生成条件が同一であることを検証する。"""
    required = (
        "run_id",
        "model",
        "theme",
        "prompt_version",
        "prompt_sha256",
        "system_prompt_sha256",
        "generation_parameters",
        "all_prechecks_passed",
    )
    for label, payload in (
        ("baseline", baseline),
        ("candidate", candidate),
    ):
        missing = [
            name
            for name in required
            if name not in payload
        ]
        if missing:
            raise KeyError(
                f"{label} Metadataの必須Keyがありません: {missing}"
            )

    if baseline["run_id"] != ADOPTED_GENERATION_RUN_ID:
        raise ValueError(
            "Baseline Generation RunがSTEP 3採用値ではありません。"
        )

    for name in (
        "model",
        "theme",
        "prompt_version",
        "system_prompt_sha256",
        "generation_parameters",
    ):
        if baseline[name] != candidate[name]:
            raise ValueError(
                "Skills以外の生成条件が変わっています: "
                f"{name}: baseline={baseline[name]!r}, "
                f"candidate={candidate[name]!r}"
            )

    if baseline["prompt_version"] != EXPECTED_PROMPT_VERSION:
        raise ValueError(
            "採用Prompt Versionではありません: "
            f"{baseline['prompt_version']}"
        )

    if candidate.get("base_prompt_sha256") != baseline["prompt_sha256"]:
        raise ValueError(
            "CandidateのBase PromptがSTEP 3採用Promptと一致しません。"
        )

    baseline_skill = baseline.get("skill", {})
    if baseline_skill and baseline_skill.get("enabled") is not False:
        raise ValueError(
            "BaselineでSkillが有効になっています。"
        )

    skill = candidate.get("skill")
    if not isinstance(skill, dict):
        raise ValueError(
            "CandidateにSkill Metadataがありません。"
        )
    if not (
        skill.get("enabled") is True
        and skill.get("version") == EXPECTED_SKILL_VERSION
    ):
        raise ValueError(
            "CandidateのSkill設定が想定値と一致しません。"
        )

    controlled = candidate.get("controlled_change")
    if not isinstance(controlled, dict):
        raise ValueError(
            "CandidateにControlled Changeがありません。"
        )
    expected_control = {
        "name": "skills",
        "baseline": False,
        "candidate": True,
        "base_prompt_changed": False,
        "max_tokens_changed": False,
    }
    for name, expected in expected_control.items():
        if controlled.get(name) != expected:
            raise ValueError(
                "Skills-only条件が成立していません: "
                f"{name}={controlled.get(name)!r}"
            )

    adopted = candidate.get("adopted_candidate", {})
    if adopted.get("generation_run_id") != baseline["run_id"]:
        raise ValueError(
            "Candidateが参照する採用Generation Runが不一致です。"
        )

    if not baseline["all_prechecks_passed"]:
        raise ValueError(
            "Baselineの事前検査が成功していません。"
        )

    return {
        "changed_variable": "skills",
        "baseline": False,
        "candidate": True,
        "model": baseline["model"],
        "theme": baseline["theme"],
        "prompt_version": baseline["prompt_version"],
        "base_prompt_sha256": baseline["prompt_sha256"],
        "generation_parameters": baseline["generation_parameters"],
        "skill": skill,
        "candidate_prechecks_passed": bool(
            candidate["all_prechecks_passed"]
        ),
        "candidate_failed_prechecks": candidate.get(
            "failed_prechecks",
            [],
        ),
    }


def build_rows(
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    names: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in names:
        baseline = metric_value(
            baseline_metrics,
            name,
        )
        candidate = metric_value(
            candidate_metrics,
            name,
        )
        rows.append(
            {
                "metric": name,
                "baseline": rounded(baseline),
                "candidate": rounded(candidate),
                "delta": rounded(candidate - baseline),
            }
        )
    return rows


def build_generation_rows(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_metrics = baseline.get("metrics", {})
    candidate_metrics = candidate.get("metrics", {})
    rows: list[dict[str, Any]] = []
    for name in GENERATION_METRICS:
        if name not in baseline_metrics or name not in candidate_metrics:
            raise KeyError(
                f"Generation Metricがありません: {name}"
            )
        baseline_value = float(
            baseline_metrics[name]
        )
        candidate_value = float(
            candidate_metrics[name]
        )
        delta = candidate_value - baseline_value
        delta_percent = (
            delta / baseline_value * 100
            if baseline_value != 0
            else None
        )
        rows.append(
            {
                "metric": name,
                "baseline": rounded(baseline_value),
                "candidate": rounded(candidate_value),
                "delta": rounded(delta),
                "delta_percent": (
                    rounded(delta_percent)
                    if delta_percent is not None
                    else None
                ),
            }
        )
    return rows


def print_table(
    title: str,
    rows: list[dict[str, Any]],
    *,
    include_percent: bool = False,
) -> None:
    print()
    print(title)
    if include_percent:
        print(
            "| Metric | No Skill | Skill | Delta | Delta % |"
        )
        print("|---|---:|---:|---:|---:|")
    else:
        print("| Metric | No Skill | Skill | Delta |")
        print("|---|---:|---:|---:|")

    for row in rows:
        if include_percent:
            percent = row["delta_percent"]
            percent_text = (
                f"{percent:+.2f}%"
                if percent is not None
                else "n/a"
            )
            print(
                f"| `{row['metric']}` | {row['baseline']} | "
                f"{row['candidate']} | {row['delta']:+} | "
                f"{percent_text} |"
            )
        else:
            print(
                f"| `{row['metric']}` | {row['baseline']} | "
                f"{row['candidate']} | {row['delta']:+} |"
            )


def main() -> None:
    args = parse_args()
    baseline_generation = load_json(
        args.baseline_generation_json
    )
    candidate_generation = load_json(
        args.candidate_generation_json
    )
    control = validate_controlled_change(
        baseline_generation,
        candidate_generation,
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    baseline_run = client.get_run(
        args.baseline_evaluation_run_id
    )
    candidate_run = client.get_run(
        args.candidate_evaluation_run_id
    )

    if (
        baseline_run.data.params.get("source_run_id")
        != baseline_generation["run_id"]
    ):
        raise ValueError(
            "Baseline Evaluation RunとGeneration Runの対応が不正です。"
        )
    if (
        candidate_run.data.params.get("source_run_id")
        != candidate_generation["run_id"]
    ):
        raise ValueError(
            "Candidate Evaluation RunとGeneration Runの対応が不正です。"
        )

    versions = comparable_versions(
        baseline_run.data.params,
        candidate_run.data.params,
    )
    evaluation_rows = build_rows(
        baseline_run.data.metrics,
        candidate_run.data.metrics,
        METRICS,
    )
    generation_rows = build_generation_rows(
        baseline_generation,
        candidate_generation,
    )
    checks = build_skill_success_checks(
        baseline_run.data.metrics,
        candidate_run.data.metrics,
        candidate_prechecks_passed=bool(
            candidate_generation["all_prechecks_passed"]
        ),
    )
    passed = all(checks.values())

    baseline_mean = llm_judge_mean(
        baseline_run.data.metrics
    )
    candidate_mean = llm_judge_mean(
        candidate_run.data.metrics
    )
    payload = {
        "experiment": "step-4-skills-only",
        "baseline_generation_json": str(
            args.baseline_generation_json
        ),
        "candidate_generation_json": str(
            args.candidate_generation_json
        ),
        "baseline_generation_run_id": baseline_generation["run_id"],
        "candidate_generation_run_id": candidate_generation["run_id"],
        "baseline_evaluation_run_id": args.baseline_evaluation_run_id,
        "candidate_evaluation_run_id": args.candidate_evaluation_run_id,
        "controlled_change": control,
        "evaluation_versions": versions,
        "evaluation_metrics": evaluation_rows,
        "generation_metrics": generation_rows,
        "llm_judge_mean": {
            "baseline": rounded(baseline_mean),
            "candidate": rounded(candidate_mean),
            "delta": rounded(candidate_mean - baseline_mean),
        },
        "success_checks": checks,
        "all_checks_passed": passed,
        "decision": (
            "adopt-skill-v1"
            if passed
            else "keep-step-3-no-skill"
        ),
    }

    output_dir = Path("evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    output_path = output_dir / (
        f"{safe_output_prefix(args.output_prefix)}_{timestamp}.json"
    )
    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print_table("Evaluation metrics", evaluation_rows)
    print_table(
        "Generation cost",
        generation_rows,
        include_percent=True,
    )
    print()
    print("LLM Judge mean:")
    print(f"  No Skill : {baseline_mean:.3f}")
    print(f"  Skill    : {candidate_mean:.3f}")
    print(
        "  Delta    : "
        f"{candidate_mean - baseline_mean:+.3f}"
    )
    print()
    print("Success checks:")
    for name, value in checks.items():
        mark = "PASS" if value else "FAIL"
        print(f"  {mark}: {name}")
    print()
    print("Overall:", "PASS" if passed else "FAIL")
    print("Decision:", payload["decision"])
    print("Result  :", output_path)

    client.log_artifact(
        args.candidate_evaluation_run_id,
        str(output_path),
        artifact_path="comparison",
    )


if __name__ == "__main__":
    main()
