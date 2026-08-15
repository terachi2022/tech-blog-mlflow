"""STEP-α-1: MLflow Review Queueを構成するCLI。"""

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
    setup_review_workflow,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baselineと採用CandidateをMLflow Review Queueへ登録する。"
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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = setup_review_workflow(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        queue_name=args.queue_name,
        baseline_evaluation_run_id=args.baseline_evaluation_run_id,
        candidate_evaluation_run_id=args.candidate_evaluation_run_id,
        dry_run=args.dry_run,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "dry_run" if args.dry_run else "applied"
    output_path = output_dir / (
        f"review_setup_alpha_1_{suffix}_{timestamp}.json"
    )
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 60)
    print("STEP-α-1 MLflow Review Setup")
    print("=" * 60)
    print("Mode           :", result["mode"])
    print("Experiment     :", result["experiment_name"])
    print("Experiment ID  :", result["experiment_id"])
    print("Queue          :", result["queue_name"])
    print("Target Traces  :", len(result["targets"]))
    print("Questions      :", len(result["question_contracts"]))
    if not args.dry_run:
        print("Queue ID       :", result["queue"]["queue_id"])
        print("Review Traces  :", len(result["presentations"]))
        print(
            "Markdown valid:",
            all(
                item["markdown_preview_valid"]
                for item in result["presentations"]
            ),
        )
        print("Review URL     :", result["review_url"])
    print("Result JSON    :", output_path)
    print()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
