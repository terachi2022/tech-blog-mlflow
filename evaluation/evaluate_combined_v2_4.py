import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from evaluation.evaluate_llm_judge import (
    GENERATOR_MODEL,
    package_version,
)
from evaluation.content_calibration_v2_4 import (
    CONTENT_CALIBRATION_VERSION,
)
from evaluation.llm_scorer_v2_4 import (
    build_llm_judge_v2_4_scorer,
)
from evaluation.local_judge import (
    DEFAULT_JUDGE_MODEL,
)
from evaluation.local_judge_v2_4 import (
    DEFAULT_JUDGE_V2_4_PROMPT,
    LocalArticleJudgeV2_4,
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

CODE_SCORER_VERSION = "code-scorer-v3.1"
JUDGE_PROMPT_VERSION = "article-judge-v2.4"
COMBINED_VERSION = "combined-v2.4.0"
CITATION_CALIBRATION_VERSION = (
    "citation-calibration-v2.3.1"
)
JUDGE_SUBSCORE_COUNT = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Code Scorerと校正済み"
            "Local LLM Judge-v2を実行する"
        )
    )

    parser.add_argument(
        "--article",
        default=DEFAULT_ARTICLE,
    )

    parser.add_argument(
        "--source-run-id",
        default=DEFAULT_SOURCE_RUN_ID,
    )

    parser.add_argument(
        "--theme",
        default=DEFAULT_THEME,
    )

    parser.add_argument(
        "--variant",
        default="baseline",
    )

    parser.add_argument(
        "--generator-prompt-version",
        default="baseline-v1",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_JUDGE_MODEL,
    )

    parser.add_argument(
        "--prompt",
        default=str(
            DEFAULT_JUDGE_V2_4_PROMPT
        ),
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3600,
    )

    parser.add_argument(
        "--run-name",
        default=(
            "calibrated-evaluation-v2.4.0"
        ),
    )

    return parser.parse_args()


def normalize(
    value: Any,
) -> Any:
    if hasattr(value, "item"):
        return value.item()

    if isinstance(
        value,
        (str, int, float, bool),
    ) or value is None:
        return value

    return str(value)


def main() -> None:
    args = parse_args()

    article_path = Path(args.article)
    prompt_path = Path(args.prompt)

    if not article_path.exists():
        raise FileNotFoundError(
            f"記事がありません: {article_path}"
        )

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Promptがありません: {prompt_path}"
        )

    article = article_path.read_text(
        encoding="utf-8"
    )
    prompt = prompt_path.read_text(
        encoding="utf-8"
    )
    article_sha256 = hashlib.sha256(
        article.encode("utf-8")
    ).hexdigest()
    judge_prompt_sha256 = hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    source_run = MlflowClient().get_run(
        args.source_run_id
    )
    source_prompt_version = (
        source_run.data.params.get(
            "prompt_version"
        )
    )
    if (
        source_prompt_version
        and source_prompt_version
        != args.generator_prompt_version
    ):
        raise ValueError(
            "Generation RunのPrompt Versionが"
            "評価引数と一致しません: "
            f"run={source_prompt_version}, "
            f"argument={args.generator_prompt_version}"
        )

    source_article_sha256 = (
        source_run.data.params.get(
            "article_sha256"
        )
    )
    if (
        source_article_sha256
        and source_article_sha256
        != article_sha256
    ):
        raise ValueError(
            "Generation RunのArticle SHA-256が"
            "評価対象記事と一致しません。"
        )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    judge = LocalArticleJudgeV2_4(
        model_id=args.model,
        prompt_path=prompt_path,
        max_tokens=args.max_tokens,
    )

    judge.load_model()

    llm_judge_v2_4 = (
        build_llm_judge_v2_4_scorer(
            judge
        )
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
        llm_judge_v2_4,
    ]

    data = [
        {
            "inputs": {
                "theme": args.theme,
                "article_path": str(
                    article_path
                ),
                "article_sha256": (
                    article_sha256
                ),
                "article_variant": (
                    args.variant
                ),
                "source_run_id": (
                    args.source_run_id
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
                "judge_prompt_sha256": (
                    judge_prompt_sha256
                ),
                "combined_version": (
                    COMBINED_VERSION
                ),
                "citation_calibration_version": (
                    CITATION_CALIBRATION_VERSION
                ),
                "content_calibration_version": (
                    CONTENT_CALIBRATION_VERSION
                ),
            },
            "outputs": article,
            "tags": {
                "article_variant": (
                    args.variant
                ),
                "evaluation_version": (
                    COMBINED_VERSION
                ),
            },
        }
    ]

    print()
    print("=" * 60)
    print("Calibrated Combined Evaluation")
    print("=" * 60)
    print("Article      :", article_path)
    print("Variant      :", args.variant)
    print(
        "Generator    :",
        GENERATOR_MODEL,
    )
    print(
        "Gen Prompt   :",
        args.generator_prompt_version,
    )
    print("Judge        :", args.model)
    print(
        "Judge Prompt :",
        JUDGE_PROMPT_VERSION,
    )
    print(
        "Subscores    :",
        JUDGE_SUBSCORE_COUNT,
    )
    print()

    with mlflow.start_run(
        run_name=args.run_name
    ) as run:
        run_id = run.info.run_id

        mlflow.log_params(
            {
                "evaluation_stage": (
                    "step-3-c-content-and-citation-calibration"
                ),
                "evaluation_type": (
                    "calibrated-code-local-llm-citation"
                ),
                "article_sha256": article_sha256,
                "judge_prompt_sha256": judge_prompt_sha256,
                "combined_version": (
                    COMBINED_VERSION
                ),
                "citation_calibration_version": (
                    CITATION_CALIBRATION_VERSION
                ),
                "content_calibration_version": (
                    CONTENT_CALIBRATION_VERSION
                ),
                "article_variant": (
                    args.variant
                ),
                "article_path": str(
                    article_path
                ),
                "source_run_id": (
                    args.source_run_id
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
                "judge_max_tokens": (
                    args.max_tokens
                ),
                "judge_temperature": 0.0,
                "judge_subscore_count": (
                    JUDGE_SUBSCORE_COUNT
                ),
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
                    "evaluation-calibration"
                ),
                "judge_runtime": (
                    "local-mlx"
                ),
                "judge_family": (
                    "gemma-3"
                ),
                "score_method": (
                    "subcriteria-mean"
                ),
                "citation_calibration": (
                    CITATION_CALIBRATION_VERSION
                ),
                "content_calibration": (
                    CONTENT_CALIBRATION_VERSION
                ),
                "article_variant": (
                    args.variant
                ),
            }
        )

        result = mlflow.genai.evaluate(
            data=data,
            scorers=all_scorers,
        )

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

        safe_variant = (
            args.variant.replace(
                "/",
                "-",
            )
        )

        output_dir = Path(
            "evaluation_results"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_path = output_dir / (
            f"combined_v2_4_0_{safe_variant}_"
            f"{timestamp}_{run_id}.json"
        )

        csv_path = output_dir / (
            f"combined_v2_4_0_{safe_variant}_"
            f"{timestamp}_{run_id}.csv"
        )

        metrics = {
            name: normalize(value)
            for name, value
            in result.metrics.items()
        }

        payload = {
            "run_id": run_id,
            "combined_version": (
                COMBINED_VERSION
            ),
            "citation_calibration_version": (
                CITATION_CALIBRATION_VERSION
            ),
            "content_calibration_version": (
                CONTENT_CALIBRATION_VERSION
            ),
            "article": {
                "path": str(article_path),
                "sha256": article_sha256,
                "variant": args.variant,
                "source_run_id": (
                    args.source_run_id
                ),
                "generator_prompt_version": (
                    args.generator_prompt_version
                ),
            },
            "judge": {
                "model": args.model,
                "prompt_version": (
                    JUDGE_PROMPT_VERSION
                ),
                "prompt_sha256": (
                    judge_prompt_sha256
                ),
                "subscore_count": (
                    JUDGE_SUBSCORE_COUNT
                ),
                "model_load_time_sec": round(
                    judge.load_elapsed_sec,
                    3,
                ),
                "generation_time_sec": (
                    judge.total_generation_time_sec
                ),
                "records": judge.records,
            },
            "metrics": metrics,
        }

        json_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        result.result_df.to_csv(
            csv_path,
            index=False,
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
            str(json_path),
            artifact_path=(
                "calibrated_evaluation"
            ),
        )

        mlflow.log_artifact(
            str(csv_path),
            artifact_path=(
                "calibrated_evaluation"
            ),
        )

        print()
        print("=" * 60)
        print("Evaluation finished")
        print("=" * 60)
        print("Run ID :", run_id)
        print("JSON   :", json_path)
        print("CSV    :", csv_path)
        print()
        print(result)


if __name__ == "__main__":
    main()
