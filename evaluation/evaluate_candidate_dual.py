"""Candidate記事をQwen主JudgeとGemma独立Judgeで順番に評価する。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tech_blog_mlflow.candidate_models import GENERATOR, INDEPENDENT_JUDGE, PRIMARY_JUDGE, PIPELINE_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--article", type=Path, required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--variant", default="candidate-reviewed")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def evaluation_commands(*, article: Path, source_run_id: str, variant: str) -> list[list[str]]:
    common = [
        sys.executable, "-m", "evaluation.evaluate_combined_v2_4",
        "--article", str(article), "--source-run-id", source_run_id,
        "--variant", variant, "--generator-prompt-version", "article-v3.5.2",
        "--generator-model", GENERATOR.model_id,
    ]
    return [
        common + [
            "--model", PRIMARY_JUDGE.model_id,
            "--max-tokens", str(PRIMARY_JUDGE.max_tokens),
            "--judge-role", "primary",
            "--run-name", "candidate-qwen3.6-primary-evaluation-v1",
        ],
        common + [
            "--model", INDEPENDENT_JUDGE.model_id,
            "--max-tokens", str(INDEPENDENT_JUDGE.max_tokens),
            "--judge-role", "independent",
            "--run-name", "candidate-gemma3-independent-evaluation-v1",
        ],
    ]


def main() -> None:
    args = parse_args()
    if not args.article.exists():
        raise FileNotFoundError(f"記事がありません: {args.article}")
    commands = evaluation_commands(
        article=args.article, source_run_id=args.source_run_id, variant=args.variant
    )
    if args.dry_run:
        print(json.dumps({"pipeline_version": PIPELINE_VERSION, "commands": commands}, ensure_ascii=False, indent=2))
        return
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
