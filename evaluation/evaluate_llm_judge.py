import argparse
from datetime import datetime
from importlib.metadata import (
    PackageNotFoundError,
    version,
)
from pathlib import Path

import mlflow
from mlflow.entities import Feedback
from mlflow.genai import scorer

from evaluation.judge_schema import (
    JUDGE_DIMENSIONS,
)
from evaluation.local_judge import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_PROMPT_PATH,
    LocalArticleJudge,
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
    "MLflowを使って"
    "機械学習の実験を管理する方法"
)

GENERATOR_MODEL = (
    "Qwen/Qwen3-8B-MLX-4bit"
)

GENERATOR_PROMPT_VERSION = "baseline-v1"
JUDGE_PROMPT_VERSION = "article-judge-v1"


def package_version(
    package_name: str,
) -> str:
    """
    インストール済みpackageのversionを取得する。
    """
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gemma 3 Local LLM Judgeで"
            "技術ブログ記事を評価し、"
            "MLflowへ保存する"
        )
    )

    parser.add_argument(
        "--article",
        default=DEFAULT_ARTICLE,
        help="評価対象Markdown",
    )

    parser.add_argument(
        "--source-run-id",
        default=DEFAULT_SOURCE_RUN_ID,
        help="記事を生成したMLflow Run ID",
    )

    parser.add_argument(
        "--theme",
        default=DEFAULT_THEME,
        help="記事テーマ",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_JUDGE_MODEL,
        help=(
            "JudgeモデルのHugging Face ID"
            "またはローカルパス"
        ),
    )

    parser.add_argument(
        "--prompt",
        default=str(
            DEFAULT_PROMPT_PATH
        ),
        help="Judge promptファイル",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1600,
        help="Judge最大出力token数",
    )

    parser.add_argument(
        "--run-name",
        default=(
            "local-llm-judge-baseline"
        ),
        help="MLflow Run名",
    )

    return parser.parse_args()


def build_llm_judge_scorer(
    judge: LocalArticleJudge,
):
    """
    LocalArticleJudgeをMLflow Scorerへ変換する。

    Gemmaを1回呼び出し、
    6個の名前付きFeedbackを返す。
    """

    @scorer(
        name="local_article_judge_v1"
    )
    def local_article_judge(
        outputs: str,
    ) -> list[Feedback]:
        result = judge.evaluate(outputs)

        feedbacks: list[Feedback] = []

        for dimension in JUDGE_DIMENSIONS:
            item = getattr(
                result,
                dimension,
            )

            feedbacks.append(
                Feedback(
                    name=dimension,
                    value=item.score,
                    rationale=item.rationale,
                )
            )

        return feedbacks

    return local_article_judge


def main() -> None:
    args = parse_args()

    article_path = Path(args.article)
    prompt_path = Path(args.prompt)

    if not article_path.exists():
        raise FileNotFoundError(
            "評価対象記事がありません: "
            f"{article_path}"
        )

    if not prompt_path.exists():
        raise FileNotFoundError(
            "Judge promptがありません: "
            f"{prompt_path}"
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

    judge = LocalArticleJudge(
        model_id=args.model,
        prompt_path=prompt_path,
        max_tokens=args.max_tokens,
    )

    # 評価Run開始前にモデルをロードする。
    # 初回は約16GBのモデルdownloadが発生する。
    judge.load_model()

    local_judge_scorer = (
        build_llm_judge_scorer(judge)
    )

    eval_data = [
        {
            "inputs": {
                "theme": args.theme,
                "source_run_id": (
                    args.source_run_id
                ),
                "article_path": str(
                    article_path
                ),
                "generator_model": (
                    GENERATOR_MODEL
                ),
                "generator_prompt_version": (
                    GENERATOR_PROMPT_VERSION
                ),
                "judge_model": args.model,
                "judge_prompt_version": (
                    JUDGE_PROMPT_VERSION
                ),
            },
            "outputs": article,
            "tags": {
                "evaluation_level": (
                    "local-llm-judge"
                ),
                "article_type": (
                    "baseline"
                ),
            },
        }
    ]

    print()
    print("=" * 60)
    print("MLflow Local LLM Judge")
    print("=" * 60)
    print("Article :", article_path)
    print("Theme   :", args.theme)
    print("Judge   :", args.model)
    print("Prompt  :", prompt_path)
    print()

    with mlflow.start_run(
        run_name=args.run_name
    ) as run:
        mlflow.log_params(
            {
                "evaluation_stage": (
                    "step-2-11"
                ),
                "evaluation_type": (
                    "offline-local-llm-judge"
                ),
                "source_run_id": (
                    args.source_run_id
                ),
                "article_path": str(
                    article_path
                ),
                "generator_model": (
                    GENERATOR_MODEL
                ),
                "generator_prompt_version": (
                    GENERATOR_PROMPT_VERSION
                ),
                "judge_model": args.model,
                "judge_prompt_version": (
                    JUDGE_PROMPT_VERSION
                ),
                "judge_max_tokens": (
                    args.max_tokens
                ),
                "judge_temperature": 0.0,
                "judge_score_min": 1,
                "judge_score_max": 5,
                "mlflow_version": (
                    package_version("mlflow")
                ),
                "mlx_lm_version": (
                    package_version("mlx-lm")
                ),
                "pydantic_version": (
                    package_version("pydantic")
                ),
            }
        )

        mlflow.set_tags(
            {
                "stage": (
                    "baseline-evaluation"
                ),
                "judge_runtime": (
                    "local-mlx"
                ),
                "judge_family": (
                    "gemma-3"
                ),
                "judge_provider": (
                    "google"
                ),
                "generator_family": (
                    "qwen-3"
                ),
                "generator_provider": (
                    "alibaba"
                ),
            }
        )

        result = mlflow.genai.evaluate(
            data=eval_data,
            scorers=[
                local_judge_scorer,
            ],
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        result_path = Path(
            "evaluation_results"
        ) / (
            "llm_judge_"
            f"{timestamp}_"
            f"{run.info.run_id}.json"
        )

        judge.save_records(
            result_path
        )

        mlflow.log_metric(
            "judge_model_load_time_sec",
            judge.load_elapsed_sec,
        )

        mlflow.log_metric(
            "judge_generation_time_sec",
            judge.total_generation_time_sec,
        )

        mlflow.log_artifact(
            str(article_path),
            artifact_path=(
                "evaluated_article"
            ),
        )

        mlflow.log_artifact(
            str(prompt_path),
            artifact_path=(
                "judge_prompt"
            ),
        )

        mlflow.log_artifact(
            str(result_path),
            artifact_path=(
                "llm_judge"
            ),
        )

        print()
        print("=" * 60)
        print("Evaluation finished")
        print("=" * 60)
        print("Run ID      :", run.info.run_id)
        print("Result JSON :", result_path)
        print()
        print(result)


if __name__ == "__main__":
    main()
