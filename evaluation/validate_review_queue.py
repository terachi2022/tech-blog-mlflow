"""STEP-α-1: Review Queueと人手評価の有効性を検証するCLI。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from tech_blog_mlflow.review_workflow import (
    ADOPTED_EVALUATION_RUN_ID,
    BASELINE_EVALUATION_RUN_ID,
    EXPERIMENT_NAME,
    QUEUE_NAME,
    TRACKING_URI,
    validate_review_workflow,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MLflow ReviewのQueue、回答保存、完了状態、Judge一致度を検証する。"
        )
    )
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--queue-name", default=QUEUE_NAME)
    parser.add_argument(
        "--baseline-evaluation-run-id",
        default=BASELINE_EVALUATION_RUN_ID,
    )
    parser.add_argument(
        "--candidate-evaluation-run-id",
        default=ADOPTED_EVALUATION_RUN_ID,
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="人手回答とQueue完了を必須にし、未完了ならExit 1にする。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_review_workflow(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        queue_name=args.queue_name,
        expected_evaluation_run_ids=(
            args.baseline_evaluation_run_id,
            args.candidate_evaluation_run_id,
        ),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / (
        f"review_validation_alpha_1_{timestamp}.json"
    )
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    queue = result["queue"]
    effective = result["effectiveness"]
    alignment = result["judge_alignment"]

    print("=" * 60)
    print("STEP-α-1 MLflow Review Validation")
    print("=" * 60)
    print("Status         :", result["validation_status"])
    print("Queue ID       :", queue["queue_id"])
    print("Items          :", queue["item_count"])
    print("Pending        :", queue["status_counts"].get("pending", 0))
    print("Complete       :", queue["status_counts"].get("complete", 0))
    print(
        "Markdown valid:",
        effective["markdown_previews_valid"],
    )
    print("Setup effective:", effective["setup_effective"])
    print(
        "Review effective:",
        effective["workflow_completion_effective"],
    )
    print("Judge MAE      :", alignment["mean_absolute_error"])
    print(
        "Within ±1     :",
        alignment["within_one_agreement_rate"],
    )
    print("Review URL     :", result["review_url"])
    print("Result JSON    :", output_path)
    print()
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.require_complete and not effective[
        "workflow_completion_effective"
    ]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
