import hashlib
import json
import platform
import time
from datetime import datetime
from importlib.metadata import (
    PackageNotFoundError,
    version,
)
from pathlib import Path
from typing import Final

import mlflow
import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


TRACKING_URI: Final = (
    "http://127.0.0.1:5000"
)

EXPERIMENT_NAME: Final = (
    "tech-blog-generation"
)

MODEL_ID: Final = (
    "Qwen/Qwen3-8B-MLX-4bit"
)

PROMPT_PATH: Final = Path(
    "prompts/article_generation_v2.md"
)

PROMPT_VERSION: Final = "article-v2"

THEME: Final = (
    "MLflowを使って"
    "機械学習の実験を管理する方法"
)

MAX_TOKENS: Final = 2048
TEMPERATURE: Final = 0.0
SEED: Final = 42
ENABLE_THINKING: Final = False

BASELINE_GENERATION_RUN_ID: Final = (
    "b7dfd7ec5d0c4439873da3684fc2c5b2"
)

BASELINE_EVALUATION_RUN_ID: Final = (
    "e4bffd1a4e1f45af99821036114debb2"
)


def package_version(
    package_name: str,
) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def render_prompt() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Promptがありません: {PROMPT_PATH}"
        )

    template = PROMPT_PATH.read_text(
        encoding="utf-8"
    )

    replacements = {
        "{{THEME}}": THEME,
        "{{MACOS_VERSION}}": (
            platform.mac_ver()[0]
            or "unknown"
        ),
        "{{PYTHON_VERSION}}": (
            platform.python_version()
        ),
        "{{MLFLOW_VERSION}}": (
            package_version("mlflow")
        ),
        "{{MLX_LM_VERSION}}": (
            package_version("mlx-lm")
        ),
    }

    rendered = template

    for marker, value in replacements.items():
        rendered = rendered.replace(
            marker,
            value,
        )

    if "{{" in rendered or "}}" in rendered:
        raise ValueError(
            "Promptに未置換のplaceholderが"
            "残っています。"
        )

    return rendered


def strip_outer_markdown_fence(
    text: str,
) -> str:
    """
    モデルが記事全体をMarkdown fenceで
    囲んだ場合だけ、外側のfenceを除去する。
    """
    result = text.strip()

    prefixes = (
        "```markdown\n",
        "```md\n",
        "```\n",
    )

    for prefix in prefixes:
        if (
            result.startswith(prefix)
            and result.endswith("```")
        ):
            result = result[
                len(prefix):-3
            ].strip()
            break

    return result


def prompt_sha256(
    prompt: str,
) -> str:
    return hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()


def article_checks(
    article: str,
) -> dict[str, bool]:
    return {
        "starts_with_h1": (
            article.startswith("# ")
        ),
        "has_prerequisites": (
            "## 前提条件" in article
        ),
        "has_failure_cases": (
            "## よくあるエラー"
            in article
        ),
        "has_references": (
            "## 参考資料" in article
        ),
        "has_summary": (
            "## まとめ" in article
        ),
        "has_thinking_output": (
            "<think>" in article
            or "</think>" in article
        ),
    }


def main() -> None:
    articles_dir = Path("articles")
    results_dir = Path(
        "generation_results"
    )

    articles_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rendered_prompt = render_prompt()
    prompt_hash = prompt_sha256(
        rendered_prompt
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    article_path = (
        articles_dir
        / f"prompt_v2_{timestamp}.md"
    )

    rendered_prompt_path = (
        results_dir
        / (
            "rendered_prompt_v2_"
            f"{timestamp}.md"
        )
    )

    rendered_prompt_path.write_text(
        rendered_prompt,
        encoding="utf-8",
    )

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    mx.random.seed(SEED)
    mx.reset_peak_memory()

    sampler = make_sampler(
        temp=TEMPERATURE
    )

    print("=" * 60)
    print("Prompt-v2 Article Generation")
    print("=" * 60)
    print("Model          :", MODEL_ID)
    print("Prompt version :", PROMPT_VERSION)
    print("Prompt SHA-256 :", prompt_hash)
    print("Max tokens     :", MAX_TOKENS)
    print("Temperature    :", TEMPERATURE)
    print("Seed           :", SEED)
    print("Thinking       :", ENABLE_THINKING)
    print()

    with mlflow.start_run(
        run_name="prompt-v2-qwen-v1"
    ) as run:
        run_id = run.info.run_id

        mlflow.log_params(
            {
                "model": MODEL_ID,
                "prompt_version": (
                    PROMPT_VERSION
                ),
                "prompt_sha256": (
                    prompt_hash
                ),
                "language": "ja",
                "max_tokens": (
                    MAX_TOKENS
                ),
                "temperature": (
                    TEMPERATURE
                ),
                "seed": SEED,
                "enable_thinking": (
                    ENABLE_THINKING
                ),
                "skills": False,
                "subagents": False,
                "rag": False,
                "lora": False,
                "theme": THEME,
                "baseline_generation_run_id": (
                    BASELINE_GENERATION_RUN_ID
                ),
                "baseline_evaluation_run_id": (
                    BASELINE_EVALUATION_RUN_ID
                ),
                "mlflow_version": (
                    package_version("mlflow")
                ),
                "mlx_lm_version": (
                    package_version("mlx-lm")
                ),
                "mlx_version": (
                    package_version("mlx")
                ),
                "python_version": (
                    platform.python_version()
                ),
                "macos_version": (
                    platform.mac_ver()[0]
                ),
            }
        )

        mlflow.set_tags(
            {
                "stage": (
                    "prompt-improvement"
                ),
                "purpose": (
                    "improve-offline-quality"
                ),
                "change_scope": (
                    "prompt-only"
                ),
                "article_variant": (
                    "prompt-v2"
                ),
                "baseline_prompt_version": (
                    "baseline-v1"
                ),
                "candidate_prompt_version": (
                    PROMPT_VERSION
                ),
            }
        )

        model_load_started = (
            time.perf_counter()
        )

        model, tokenizer = load(
            MODEL_ID
        )

        model_load_time_sec = (
            time.perf_counter()
            - model_load_started
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "指示された要件に従い、"
                    "日本語の技術記事本文だけを"
                    "Markdownで出力してください。"
                ),
            },
            {
                "role": "user",
                "content": rendered_prompt,
            },
        ]

        formatted_prompt = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=(
                    ENABLE_THINKING
                ),
            )
        )

        generation_started = (
            time.perf_counter()
        )

        raw_article = generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=MAX_TOKENS,
            sampler=sampler,
            verbose=False,
        )

        generation_time_sec = (
            time.perf_counter()
            - generation_started
        )

        article = strip_outer_markdown_fence(
            raw_article
        )

        article_path.write_text(
            article + "\n",
            encoding="utf-8",
        )

        output_tokens = len(
            tokenizer.encode(article)
        )

        generation_tokens_per_sec = (
            output_tokens
            / generation_time_sec
            if generation_time_sec > 0
            else 0.0
        )

        peak_memory_gb = (
            mx.get_peak_memory()
            / 1_000_000_000
        )

        checks = article_checks(article)

        mlflow.log_metrics(
            {
                "model_load_time_sec": (
                    model_load_time_sec
                ),
                "generation_time_sec": (
                    generation_time_sec
                ),
                "article_length": (
                    len(article)
                ),
                "output_tokens": (
                    output_tokens
                ),
                "generation_tokens_per_sec": (
                    generation_tokens_per_sec
                ),
                "peak_memory_gb": (
                    peak_memory_gb
                ),
            }
        )

        for check_name, check_value in (
            checks.items()
        ):
            mlflow.log_metric(
                f"precheck_{check_name}",
                int(check_value),
            )

        metadata_path = (
            results_dir
            / (
                "prompt_v2_generation_"
                f"{timestamp}_"
                f"{run_id}.json"
            )
        )

        metadata = {
            "run_id": run_id,
            "run_name": (
                "prompt-v2-qwen-v1"
            ),
            "article_path": str(
                article_path
            ),
            "rendered_prompt_path": str(
                rendered_prompt_path
            ),
            "model": MODEL_ID,
            "prompt_version": (
                PROMPT_VERSION
            ),
            "prompt_sha256": (
                prompt_hash
            ),
            "theme": THEME,
            "generation_parameters": {
                "max_tokens": (
                    MAX_TOKENS
                ),
                "temperature": (
                    TEMPERATURE
                ),
                "seed": SEED,
                "enable_thinking": (
                    ENABLE_THINKING
                ),
            },
            "metrics": {
                "model_load_time_sec": round(
                    model_load_time_sec,
                    3,
                ),
                "generation_time_sec": round(
                    generation_time_sec,
                    3,
                ),
                "article_length": len(
                    article
                ),
                "output_tokens": (
                    output_tokens
                ),
                "generation_tokens_per_sec": (
                    round(
                        generation_tokens_per_sec,
                        3,
                    )
                ),
                "peak_memory_gb": round(
                    peak_memory_gb,
                    3,
                ),
            },
            "prechecks": checks,
            "baseline": {
                "generation_run_id": (
                    BASELINE_GENERATION_RUN_ID
                ),
                "evaluation_run_id": (
                    BASELINE_EVALUATION_RUN_ID
                ),
            },
        }

        metadata_path.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        mlflow.log_artifact(
            str(article_path),
            artifact_path="articles",
        )

        mlflow.log_artifact(
            str(PROMPT_PATH),
            artifact_path="prompts",
        )

        mlflow.log_artifact(
            str(rendered_prompt_path),
            artifact_path="prompts",
        )

        mlflow.log_artifact(
            str(metadata_path),
            artifact_path=(
                "generation_metadata"
            ),
        )

        print()
        print("=" * 60)
        print("Generation finished")
        print("=" * 60)
        print("Run ID        :", run_id)
        print("Article       :", article_path)
        print("Metadata      :", metadata_path)
        print(
            "Generation    :",
            f"{generation_time_sec:.3f} sec",
        )
        print(
            "Article chars :",
            len(article),
        )
        print(
            "Output tokens :",
            output_tokens,
        )
        print(
            "Generation TPS:",
            (
                f"{generation_tokens_per_sec:.3f}"
            ),
        )
        print(
            "Peak memory   :",
            f"{peak_memory_gb:.3f} GB",
        )
        print("Prechecks     :", checks)


if __name__ == "__main__":
    main()
