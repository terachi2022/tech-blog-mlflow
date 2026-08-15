"""STEP-α-4: MLflow Evaluation Datasetを登録するCLI。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from tech_blog_mlflow.evaluation_dataset_registry import (
    EXPERIMENT_NAME,
    TRACKING_URI,
    setup_evaluation_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--output-dir", default="evaluation_results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = setup_evaluation_dataset(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        dry_run=args.dry_run,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "dry_run" if args.dry_run else "applied"
    path = output_dir / f"evaluation_dataset_alpha_4_{mode}_{timestamp}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 60)
    print("STEP-α-4 MLflow Evaluation Dataset Setup")
    print("=" * 60)
    print("Mode       :", result["mode"])
    print("Dataset    :", result["dataset_name"])
    print("Records    :", result["record_count"])
    if not args.dry_run:
        print("Dataset ID :", result["dataset_id"])
        print("Digest     :", result["digest"])
        print("Created    :", result["created"])
    print("Result JSON:", path)


if __name__ == "__main__":
    main()
