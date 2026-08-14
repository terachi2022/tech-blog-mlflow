import argparse
from pathlib import Path

import mlflow

from evaluation.scorers import (
    article_length_chars,
    code_block_count,
    conclusion_near_top,
    has_failure_cases,
    has_h1,
    has_prerequisites,
    has_version_info,
    public_external_link_count,
    reproducibility_proxy,
    structure_score,
)


TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "tech-blog-generation"

DEFAULT_ARTICLE = (
    "articles/baseline_20260814_004017.md"
)

DEFAULT_SOURCE_RUN_ID = (
    "b7dfd7ec5d0c4439873da3684fc2c5b2"
)

DEFAULT_THEME = (
    "MLflowを使って機械学習の実験を管理する方法"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "生成済み技術ブログを"
            "MLflow GenAI Scorerで評価する"
        )
    )

    parser.add_argument(
        "--article",
        default=DEFAULT_ARTICLE,
        help="評価対象Markdownファイル",
    )

    parser.add_argument(
        "--source-run-id",
        default=DEFAULT_SOURCE_RUN_ID,
        help="記事を生成したMLflow Run ID",
    )

    parser.add_argument(
        "--theme",
        default=DEFAULT_THEME,
        help="記事生成時のテーマ",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    article_path = Path(args.article)

    if not article_path.exists():
        raise FileNotFoundError(
            f"記事がありません: {article_path}"
        )

    article = article_path.read_text(
        encoding="utf-8"
    )

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    eval_data = [
        {
            "inputs": {
                "theme": args.theme,
                "source_run_id": args.source_run_id,
                "article_path": str(article_path),
                "generator": (
                    "Qwen/Qwen3-8B-MLX-4bit"
                ),
                "prompt_version": (
                    "baseline-v1"
                ),
            },
            "outputs": article,
        }
    ]

    scorers = [
        has_h1,
        conclusion_near_top,
        code_block_count,
        public_external_link_count,
        has_version_info,
        has_prerequisites,
        has_failure_cases,
        structure_score,
        reproducibility_proxy,
        article_length_chars,
    ]

    print("=" * 60)
    print("MLflow Offline Evaluation")
    print("=" * 60)

    print(
        "Article       :",
        article_path,
    )

    print(
        "Source Run ID :",
        args.source_run_id,
    )

    print(
        "Theme         :",
        args.theme,
    )

    print(
        "Article chars :",
        len(article),
    )

    print()

    result = mlflow.genai.evaluate(
        data=eval_data,
        scorers=scorers,
    )

    print()
    print("=" * 60)
    print("Evaluation finished")
    print("=" * 60)

    print(result)


if __name__ == "__main__":
    main()
