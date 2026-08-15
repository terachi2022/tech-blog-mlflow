"""STEP-α-4: MLflow Evaluation Datasetを検証するCLI。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from tech_blog_mlflow.evaluation_dataset_registry import (
    EXPERIMENT_NAME,
    TRACKING_URI,
    validate_evaluation_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--output-dir", default="evaluation_results")
    args = parser.parse_args()
    result = validate_evaluation_dataset(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"evaluation_dataset_alpha_4_validation_{timestamp}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 60)
    print("STEP-α-4 MLflow Evaluation Dataset Validation")
    print("=" * 60)
    print("Status     :", result["validation_status"])
    print("Dataset    :", result["dataset_name"])
    print("Dataset ID :", result.get("dataset_id"))
    print("Records    :", result.get("record_count"))
    print("Variants   :", result.get("variants"))
    print("Result JSON:", path)
    if result["validation_status"] != "validated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
