"""STEP-α-5: Judges/Scorers統合を構成するCLI。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from tech_blog_mlflow.judge_integration import (
    EXPERIMENT_NAME,
    TRACKING_URI,
    setup_judge_integration,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--output-dir", default="evaluation_results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = setup_judge_integration(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        dry_run=args.dry_run,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "dry_run" if args.dry_run else "applied"
    path = output_dir / f"judges_alpha_5_{mode}_{timestamp}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 60)
    print("STEP-α-5 MLflow Judges Integration Setup")
    print("=" * 60)
    print("Mode          :", result["mode"])
    print("Local MLX     : registration_supported=False")
    if not args.dry_run:
        print("Registered    :", result["registered_scorer"]["name"])
        print("Created       :", result["registered_scorer"]["created"])
        print("Auto sampling :", result["registered_scorer"]["sample_rate"])
    print("Result JSON   :", path)


if __name__ == "__main__":
    main()
