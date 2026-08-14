import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow

from evaluation.evaluate_llm_judge import (
    GENERATOR_MODEL,
    GENERATOR_PROMPT_VERSION,
    JUDGE_PROMPT_VERSION,
    build_llm_judge_scorer,
    package_version,
)
from evaluation.local_judge import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_PROMPT_PATH,
    LocalArticleJudge,
)
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
    "MLflowを使って"
    "機械学習の実験を管理する方法"
)

CODE_SCORER_VERSION = "code-scorer-v2"
COMBINED_EVALUATION_VERSION = "combined-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Code ScorerとLocal LLM Judgeを"
            "同じMLflow Runで実行する"
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
        help="記事生成元のMLflow Run ID",
    )

    parser.add_argument(
        "--theme",
        default=DEFAULT_THEME,
        help="記事テーマ",
    )

    parser.add_argument(
        "--variant",
        default="baseline",
        help=(
            "記事の種類。"
            "baseline、prompt-v2、rag-v1等"
        ),
    )

    parser.add_argument(
        "--generator-prompt-version",
        default=GENERATOR_PROMPT_VERSION,
        help="記事生成時のPrompt version",
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
            "combined-baseline-evaluation-v1"
        ),
        help="MLflow Run名",
    )

    return parser.parse_args()


def normalize_metric_value(
    value: Any,
) -> Any:
    """
    NumPy型などをJSONへ保存可能な
    Python標準型へ変換する。
    """
    if hasattr(value, "item"):
        return value.item()

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if value is None:
        return None

    return str(value)


def save_combined_result(
    *,
    output_path: Path,
    run_id: str,
    article_path: Path,
    theme: str,
    variant: str,
    judge: LocalArticleJudge,
    evaluation_result: Any,
) -> None:
    """
    Code ScorerとLLM Judgeの結果を
    1つのJSONへ保存する。
    """
    metrics = {
        name: normalize_metric_value(value)
        for name, value
        in evaluation_result.metrics.items()
    }

    code_metric_names = (
        "has_h1/mean",
        "conclusion_near_top/mean",
        "code_block_count/mean",
        "public_external_link_count/mean",
        "has_version_info/mean",
        "has_prerequisites/mean",
        "has_failure_cases/mean",
        "structure_score/mean",
        "reproducibility_proxy/mean",
        "article_length_chars/mean",
    )

    llm_metric_names = (
        "technical_accuracy/mean",
        "helpfulness/mean",
        "reproducibility/mean",
        "citation_quality/mean",
        "readability_ja/mean",
        "original_value/mean",
    )

    code_metrics = {
        name: metrics.get(name)
        for name in code_metric_names
    }

    llm_metrics = {
        name: metrics.get(name)
        for name in llm_metric_names
    }

    payload = {
        "run_id": run_id,
        "evaluation_version": (
            COMBINED_EVALUATION_VERSION
        ),
        "article": {
            "path": str(article_path),
            "theme": theme,
            "variant": variant,
        },
        "code_scorer": {
            "version": CODE_SCORER_VERSION,
            "metrics": code_metrics,
        },
        "local_llm_judge": {
            "model": judge.model_id,
            "prompt_path": str(
                judge.prompt_path
            ),
            "prompt_version": (
                JUDGE_PROMPT_VERSION
            ),
            "model_load_time_sec": round(
                judge.load_elapsed_sec,
                3,
            ),
            "generation_time_sec": (
                judge.total_generation_time_sec
            ),
            "metrics": llm_metrics,
            "records": judge.records,
        },
        "all_metrics": metrics,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


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

    if not article.strip():
        raise ValueError(
            "評価対象記事が空です: "
            f"{article_path}"
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

    # MLflow Runを始める前にモデルをロードする。
    # モデル取得済みなら数秒程度でロードされる。
    judge.load_model()

    local_llm_judge = (
        build_llm_judge_scorer(judge)
    )

    code_scorers = [
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

    all_scorers = [
        *code_scorers,
        local_llm_judge,
    ]

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
                "article_variant": (
                    args.variant
                ),
                "generator_model": (
                    GENERATOR_MODEL
                ),
                "generator_prompt_version": (
                    args.generator_prompt_version
                ),
                "code_scorer_version": (
                    CODE_SCORER_VERSION
                ),
                "judge_model": args.model,
                "judge_prompt_version": (
                    JUDGE_PROMPT_VERSION
                ),
                "combined_evaluation_version": (
                    COMBINED_EVALUATION_VERSION
                ),
            },
            "outputs": article,
            "tags": {
                "evaluation_level": (
                    "combined-offline-evaluation"
                ),
                "article_variant": (
                    args.variant
                ),
            },
        }
    ]

    print()
    print("=" * 60)
    print("MLflow Combined Offline Evaluation")
    print("=" * 60)
    print("Article       :", article_path)
    print("Theme         :", args.theme)
    print("Variant       :", args.variant)
    print("Code Scorers  :", len(code_scorers))
    print("LLM Metrics   :", 6)
    print("Judge         :", args.model)
    print("Judge Prompt  :", prompt_path)
    print()

    with mlflow.start_run(
        run_name=args.run_name
    ) as run:
        run_id = run.info.run_id

        mlflow.log_params(
            {
                "evaluation_stage": (
                    "step-2-12"
                ),
                "evaluation_type": (
                    "combined-code-and-local-llm"
                ),
                "combined_evaluation_version": (
                    COMBINED_EVALUATION_VERSION
                ),
                "source_run_id": (
                    args.source_run_id
                ),
                "article_path": str(
                    article_path
                ),
                "article_variant": (
                    args.variant
                ),
                "generator_model": (
                    GENERATOR_MODEL
                ),
                "generator_prompt_version": (
                    args.generator_prompt_version
                ),
                "code_scorer_version": (
                    CODE_SCORER_VERSION
                ),
                "code_scorer_count": (
                    len(code_scorers)
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
                "llm_judge_metric_count": 6,
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
                    "combined-baseline-evaluation"
                ),
                "evaluation_layer": (
                    "code-and-local-llm"
                ),
                "article_variant": (
                    args.variant
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
            scorers=all_scorers,
        )

        # MLflow評価以外の実行情報
        mlflow.log_metric(
            "judge_model_load_time_sec",
            judge.load_elapsed_sec,
        )

        mlflow.log_metric(
            "judge_generation_time_sec",
            judge.total_generation_time_sec,
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        result_dir = Path(
            "evaluation_results"
        )

        combined_json_path = (
            result_dir
            / (
                "combined_evaluation_"
                f"{timestamp}_"
                f"{run_id}.json"
            )
        )

        result_csv_path = (
            result_dir
            / (
                "combined_result_df_"
                f"{timestamp}_"
                f"{run_id}.csv"
            )
        )

        save_combined_result(
            output_path=combined_json_path,
            run_id=run_id,
            article_path=article_path,
            theme=args.theme,
            variant=args.variant,
            judge=judge,
            evaluation_result=result,
        )

        result.result_df.to_csv(
            result_csv_path,
            index=False,
        )

        # 評価対象と評価条件をArtifactへ保存
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
            str(combined_json_path),
            artifact_path=(
                "combined_evaluation"
            ),
        )

        mlflow.log_artifact(
            str(result_csv_path),
            artifact_path=(
                "combined_evaluation"
            ),
        )

        print()
        print("=" * 60)
        print("Combined evaluation finished")
        print("=" * 60)
        print("Run ID :", run_id)
        print(
            "JSON   :",
            combined_json_path,
        )
        print(
            "CSV    :",
            result_csv_path,
        )
        print()
        print(result)


if __name__ == "__main__":
    main()
