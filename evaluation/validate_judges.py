"""STEP-α-5: Judges/Scorers統合を検証するCLI。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from tech_blog_mlflow.judge_integration import (
    EXPERIMENT_NAME,
    TRACKING_URI,
    validate_judge_integration,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--output-dir", default="evaluation_results")
    args = parser.parse_args()
    result = validate_judge_integration(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"judges_alpha_5_validation_{timestamp}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 60)
    print("STEP-α-5 MLflow Judges Integration Validation")
    print("=" * 60)
    print("Status       :", result["validation_status"])
    print("Registered   :", [item["name"] for item in result["registered_scorers"]])
    print("Local MLX GUI:", "unsupported (tracked as offline scorer evidence)")
    print("Result JSON  :", path)
    if result["validation_status"] != "validated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
