"""STEP-α-2: Generator/Judge PromptをMLflowへ登録するCLI。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from tech_blog_mlflow.prompt_registry import (
    EXPERIMENT_NAME,
    TRACKING_URI,
    setup_prompt_registry,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--output-dir", default="evaluation_results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = setup_prompt_registry(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        dry_run=args.dry_run,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "dry_run" if args.dry_run else "applied"
    path = output_dir / f"prompt_registry_alpha_2_{mode}_{timestamp}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 60)
    print("STEP-α-2 MLflow Prompt Registry Setup")
    print("=" * 60)
    print("Mode       :", result["mode"])
    print("Experiment :", result["experiment_name"])
    print("Prompts    :", len(result.get("registrations", result["plans"])))
    if not args.dry_run:
        for item in result["registrations"]:
            print(f"{item['role'].title():10}: {item['uri']} (@{item['alias']})")
    print("Result JSON:", path)


if __name__ == "__main__":
    main()
