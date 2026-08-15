"""STEP-α-3: External Modelを検証するCLI。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from tech_blog_mlflow.external_model_registry import (
    EXPERIMENT_NAME,
    TRACKING_URI,
    validate_external_models,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--output-dir", default="evaluation_results")
    args = parser.parse_args()
    result = validate_external_models(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"external_models_alpha_3_validation_{timestamp}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 60)
    print("STEP-α-3 MLflow External Model Validation")
    print("=" * 60)
    print("Status     :", result["validation_status"])
    for item in result["models"]:
        print(f"{item['role'].title():10}: {item.get('model_uri')} valid={item['valid']}")
    print("Result JSON:", path)
    if result["validation_status"] != "validated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
