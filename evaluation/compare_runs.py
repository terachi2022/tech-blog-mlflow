"""2つのMLflow評価Runを比較し、判定結果を保存する。"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from evaluation.comparison_checks import (
    build_success_checks,
)


TRACKING_URI = "http://127.0.0.1:5000"

DEFAULT_BASELINE_RUN_ID = (
    "bacd99883951428a99f4d91cf75f3852"
)

METRICS = (
    "structure_score/mean",
    "reproducibility_proxy/mean",
    "has_prerequisites/mean",
    "has_version_info/mean",
    "has_failure_cases/mean",
    "public_external_link_count/mean",
    "technical_accuracy/mean",
    "helpfulness/mean",
    "reproducibility/mean",
    "citation_quality/mean",
    "readability_ja/mean",
    "original_value/mean",
)

LLM_JUDGE_METRICS = (
    "technical_accuracy/mean",
    "helpfulness/mean",
    "reproducibility/mean",
    "citation_quality/mean",
    "readability_ja/mean",
    "original_value/mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "BaselineとCandidateの"
            "MLflow Metricsを比較する"
        )
    )

    parser.add_argument(
        "--baseline-run-id",
        default=DEFAULT_BASELINE_RUN_ID,
    )
    parser.add_argument(
        "--candidate-run-id",
        required=True,
    )
    parser.add_argument(
        "--baseline-label",
        default="baseline-v1",
    )
    parser.add_argument(
        "--changed-variable-name",
        default=(
            "generator_prompt_and_max_tokens"
        ),
    )
    parser.add_argument(
        "--candidate-label",
        default="article-v3.3-max4096",
    )
    parser.add_argument(
        "--output-prefix",
        default=(
            "comparison_baseline_vs_prompt_v3_4"
        ),
    )

    return parser.parse_args()


def metric_value(
    metrics: dict[str, float],
    name: str,
) -> float:
    if name not in metrics:
        raise KeyError(
            f"Metricがありません: {name}"
        )

    return float(metrics[name])


def rounded(
    value: float,
) -> float:
    return round(value, 4)


def safe_output_prefix(
    value: str,
) -> str:
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


def llm_judge_mean(
    metrics: dict[str, float],
) -> float:
    return sum(
        metric_value(metrics, name)
        for name in LLM_JUDGE_METRICS
    ) / len(LLM_JUDGE_METRICS)


def main() -> None:
    args = parse_args()

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    client = MlflowClient()

    baseline = client.get_run(
        args.baseline_run_id
    )
    candidate = client.get_run(
        args.candidate_run_id
    )

    rows: list[dict[str, Any]] = []

    for name in METRICS:
        baseline_value = metric_value(
            baseline.data.metrics,
            name,
        )
        candidate_value = metric_value(
            candidate.data.metrics,
            name,
        )

        rows.append(
            {
                "metric": name,
                "baseline": rounded(
                    baseline_value
                ),
                "candidate": rounded(
                    candidate_value
                ),
                "delta": rounded(
                    candidate_value
                    - baseline_value
                ),
            }
        )

    baseline_mean = llm_judge_mean(
        baseline.data.metrics
    )
    candidate_mean = llm_judge_mean(
        candidate.data.metrics
    )

    checks = build_success_checks(
        baseline.data.metrics,
        candidate.data.metrics,
    )
    passed = all(checks.values())

    payload = {
        "baseline_run_id": (
            args.baseline_run_id
        ),
        "candidate_run_id": (
            args.candidate_run_id
        ),
        "changed_variable": {
            "name": args.changed_variable_name,
            "baseline": args.baseline_label,
            "candidate": args.candidate_label,
        },
        "metrics": rows,
        "llm_judge_mean": {
            "baseline": rounded(
                baseline_mean
            ),
            "candidate": rounded(
                candidate_mean
            ),
            "delta": rounded(
                candidate_mean
                - baseline_mean
            ),
        },
        "success_checks": checks,
        "all_checks_passed": passed,
    }

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = Path(
        "evaluation_results"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / (
        f"{safe_output_prefix(args.output_prefix)}_"
        f"{timestamp}.json"
    )

    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "| Metric | Baseline | "
        "Candidate | Delta |"
    )
    print("|---|---:|---:|---:|")

    for row in rows:
        print(
            f"| `{row['metric']}` "
            f"| {row['baseline']} "
            f"| {row['candidate']} "
            f"| {row['delta']:+} |"
        )

    print()
    print("LLM Judge mean:")
    print(
        f"  Baseline  : {baseline_mean:.3f}"
    )
    print(
        f"  Candidate : {candidate_mean:.3f}"
    )
    print(
        "  Delta     : "
        f"{candidate_mean - baseline_mean:+.3f}"
    )

    print()
    print("Success checks:")

    for name, value in checks.items():
        mark = "PASS" if value else "FAIL"
        print(f"  {mark}: {name}")

    print()
    print(
        "Overall:",
        "PASS" if passed else "FAIL",
    )
    print("Result :", output_path)

    client.log_artifact(
        args.candidate_run_id,
        str(output_path),
        artifact_path="comparison",
    )


if __name__ == "__main__":
    main()
