"""既存Candidate RunをModels、Review、比較対象としてMLflow GUIへ公開する。"""

from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
import mlflow
from mlflow import MlflowClient
from tech_blog_mlflow.candidate_gui import EXPERIMENT_NAME, TRACKING_URI, latest_candidate_run_ids, publish_candidate_models, publish_candidate_review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--generation-run-id")
    parser.add_argument("--review-run-id")
    parser.add_argument("--primary-run-id")
    parser.add_argument("--independent-run-id")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_results"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    experiment = mlflow.set_experiment(args.experiment_name)
    client = MlflowClient()
    discovered = latest_candidate_run_ids(client, experiment.experiment_id)
    run_ids = {
        "generator": args.generation_run_id or discovered["generator"],
        "reviewer": args.review_run_id or discovered["reviewer"],
        "primary-judge": args.primary_run_id or discovered["primary-judge"],
        "independent-judge": args.independent_run_id or discovered["independent-judge"],
    }
    result = {
        "tracking_uri": args.tracking_uri, "experiment_name": args.experiment_name,
        "experiment_id": experiment.experiment_id, "run_ids": run_ids,
        "compare_run_ids": [run_ids["primary-judge"], run_ids["independent-judge"]],
        "mode": "dry-run" if args.dry_run else "apply",
    }
    if not args.dry_run:
        result["models"] = publish_candidate_models(
            mlflow_module=mlflow, client=client, experiment_id=experiment.experiment_id, run_ids=run_ids,
        )
        result["review"] = publish_candidate_review(
            mlflow_module=mlflow, experiment_id=experiment.experiment_id,
            evaluation_run_id=run_ids["primary-judge"],
        )
        result["models_url"] = f"{args.tracking_uri.rstrip('/')}/#/models"
        result["review_url"] = f"{args.tracking_uri.rstrip('/')}/#/experiments/{experiment.experiment_id}/review-queue"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = args.output_dir / f"candidate_gui_{result['mode'].replace('-', '_')}_{timestamp}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Result JSON:", path)


if __name__ == "__main__":
    main()
