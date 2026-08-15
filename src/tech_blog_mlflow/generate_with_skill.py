"""STEP 4: 採用Promptを固定し、Skillだけを追加して記事を生成する。"""

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

from tech_blog_mlflow.article_v3_checks import (
    ARTICLE_MAX_CHARS,
    ARTICLE_MIN_CHARS,
    article_checks,
)


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
    "prompts/article_generation_v3_5_2.md"
)

SKILL_PATH: Final = Path(
    "skills/technical-blog-quality/SKILL.md"
)

PROMPT_VERSION: Final = "article-v3.5.2"
GENERATION_CONFIG_VERSION: Final = (
    "generation-v4.0-skills-v1"
)
SKILL_NAME: Final = "technical-blog-quality"
SKILL_VERSION: Final = "technical-blog-quality-v1"

THEME: Final = (
    "MLflowを使って"
    "機械学習の実験を管理する方法"
)

MAX_TOKENS: Final = 4096
TEMPERATURE: Final = 0.0
SEED: Final = 42
ENABLE_THINKING: Final = False

SYSTEM_PROMPT: Final = (
    "指示された要件に従い、日本語の技術記事本文だけを"
    "Markdownで出力してください。Prompt中の執筆指示は"
    "本文へ転記しないでください。架空の実行結果を作らず、"
    "提供された観測値だけを実測値として扱ってください。"
)

ADOPTED_GENERATION_RUN_ID: Final = (
    "b5c925c2322b4e30b04f07e24d160a04"
)

ADOPTED_EVALUATION_RUN_ID: Final = (
    "fdf0c239445f44a0999a6b1fe7a419b6"
)

ADOPTED_BASE_PROMPT_SHA256: Final = (
    "19416090fd3ddd3de09b354ccf125247c248125d915da3191849ce4350781c89"
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


def load_skill() -> str:
    """Project-local Skillを読み、最低限の構造を検証する。"""
    if not SKILL_PATH.exists():
        raise FileNotFoundError(
            f"Skillがありません: {SKILL_PATH}"
        )

    skill = SKILL_PATH.read_text(
        encoding="utf-8"
    ).strip()

    if not skill.startswith("---\n"):
        raise ValueError(
            "SKILL.mdにYAML frontmatterがありません。"
        )

    if f"name: {SKILL_NAME}" not in skill:
        raise ValueError(
            "SKILL.mdのnameが想定値と一致しません。"
        )

    if "description:" not in skill:
        raise ValueError(
            "SKILL.mdにdescriptionがありません。"
        )

    return skill + "\n"


def build_effective_prompt(
    base_prompt: str,
    skill: str,
) -> str:
    """採用PromptへSkillを1回だけ追加する。"""
    marker = "# Applied Skill"

    if marker in base_prompt:
        raise ValueError(
            "Base PromptにSkill markerが既にあります。"
        )

    return (
        base_prompt.rstrip()
        + "\n\n"
        + marker
        + "\n\n"
        + skill.strip()
        + "\n"
    )


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


def text_sha256(
    text: str,
) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


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

    base_rendered_prompt = render_prompt()
    base_prompt_hash = text_sha256(
        base_rendered_prompt
    )

    if base_prompt_hash != ADOPTED_BASE_PROMPT_SHA256:
        raise ValueError(
            "採用PromptのSHA-256がSTEP 3成果物と一致しません: "
            f"expected={ADOPTED_BASE_PROMPT_SHA256}, "
            f"actual={base_prompt_hash}"
        )

    skill = load_skill()
    skill_hash = text_sha256(skill)
    rendered_prompt = build_effective_prompt(
        base_rendered_prompt,
        skill,
    )
    prompt_hash = text_sha256(
        rendered_prompt
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    article_path = (
        articles_dir
        / f"skill_v1_{timestamp}.md"
    )

    rendered_prompt_path = (
        results_dir
        / (
            "rendered_prompt_skill_v1_"
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
    print("STEP 4 Skills-v1 Article Generation")
    print("=" * 60)
    print("Model          :", MODEL_ID)
    print("Prompt version :", PROMPT_VERSION)
    print(
        "Config version :",
        GENERATION_CONFIG_VERSION,
    )
    print("Base Prompt SHA:", base_prompt_hash)
    print("Skill version  :", SKILL_VERSION)
    print("Skill SHA-256  :", skill_hash)
    print("Effective SHA  :", prompt_hash)
    print("Max tokens     :", MAX_TOKENS)
    print("Temperature    :", TEMPERATURE)
    print("Seed           :", SEED)
    print("Thinking       :", ENABLE_THINKING)
    print()

    with mlflow.start_run(
        run_name="skill-v1-article-v3.5.2"
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
                "base_prompt_sha256": (
                    base_prompt_hash
                ),
                "generation_config_version": (
                    GENERATION_CONFIG_VERSION
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
                "skills": True,
                "skill_name": SKILL_NAME,
                "skill_version": (
                    SKILL_VERSION
                ),
                "skill_sha256": skill_hash,
                "subagents": False,
                "rag": False,
                "lora": False,
                "theme": THEME,
                "adopted_generation_run_id": (
                    ADOPTED_GENERATION_RUN_ID
                ),
                "adopted_evaluation_run_id": (
                    ADOPTED_EVALUATION_RUN_ID
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
                    "step-4-skills-experiment"
                ),
                "purpose": (
                    "measure-project-local-skill-effect"
                ),
                "change_scope": (
                    "skills-only"
                ),
                "article_variant": (
                    "skill-v1"
                ),
                "adopted_prompt_version": (
                    PROMPT_VERSION
                ),
                "skill_version": (
                    SKILL_VERSION
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
                "content": SYSTEM_PROMPT,
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

        formatted_prompt_hash = text_sha256(
            formatted_prompt
        )
        system_prompt_hash = text_sha256(
            SYSTEM_PROMPT
        )

        mlflow.log_params(
            {
                "formatted_prompt_sha256": formatted_prompt_hash,
                "system_prompt_sha256": system_prompt_hash,
            }
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

        # Artifact、Hash、文字数、評価入力で同じ文字列を使う。
        stored_article = article.rstrip() + "\n"

        article_hash = text_sha256(
            stored_article
        )

        mlflow.log_param(
            "article_sha256",
            article_hash,
        )

        article_path.write_text(
            stored_article,
            encoding="utf-8",
        )

        output_tokens = len(
            tokenizer.encode(stored_article)
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

        checks = article_checks(
            stored_article,
            rendered_prompt,
        )
        checks[
            "output_tokens_below_safety_limit"
        ] = output_tokens < (
            MAX_TOKENS - 64
        )
        checks[
            "article_length_in_range"
        ] = (
            ARTICLE_MIN_CHARS
            <= len(stored_article)
            <= ARTICLE_MAX_CHARS
        )
        failed_checks = (
            [
                name
                for name, passed in checks.items()
                if not passed
            ]
        )
        all_prechecks_passed = not (
            failed_checks
        )

        mlflow.log_metrics(
            {
                "model_load_time_sec": (
                    model_load_time_sec
                ),
                "generation_time_sec": (
                    generation_time_sec
                ),
                "article_length": (
                    len(stored_article)
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
                "precheck_all_passed": int(
                    all_prechecks_passed
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
                "skill_v1_generation_"
                f"{timestamp}_"
                f"{run_id}.json"
            )
        )

        metadata = {
            "run_id": run_id,
            "run_name": (
                "skill-v1-article-v3.5.2"
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
            "generation_config_version": (
                GENERATION_CONFIG_VERSION
            ),
            "prompt_sha256": (
                prompt_hash
            ),
            "base_prompt_sha256": (
                base_prompt_hash
            ),
            "formatted_prompt_sha256": (
                formatted_prompt_hash
            ),
            "system_prompt_sha256": (
                system_prompt_hash
            ),
            "skill": {
                "enabled": True,
                "name": SKILL_NAME,
                "version": SKILL_VERSION,
                "path": str(SKILL_PATH),
                "sha256": skill_hash,
            },
            "article_sha256": article_hash,
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
                    stored_article
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
            "all_prechecks_passed": (
                all_prechecks_passed
            ),
            "failed_prechecks": (
                failed_checks
            ),
            "adopted_candidate": {
                "prompt_version": PROMPT_VERSION,
                "generation_run_id": (
                    ADOPTED_GENERATION_RUN_ID
                ),
                "evaluation_run_id": (
                    ADOPTED_EVALUATION_RUN_ID
                ),
            },
            "controlled_change": {
                "name": "skills",
                "baseline": False,
                "candidate": True,
                "base_prompt_changed": False,
                "base_prompt_sha256": (
                    base_prompt_hash
                ),
                "effective_prompt_sha256": (
                    prompt_hash
                ),
                "skill_name": SKILL_NAME,
                "skill_version": SKILL_VERSION,
                "skill_sha256": skill_hash,
                "max_tokens_changed": False,
                "baseline_max_tokens": MAX_TOKENS,
                "candidate_max_tokens": (
                    MAX_TOKENS
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
            str(SKILL_PATH),
            artifact_path="skills",
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
            len(stored_article),
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
        print(
            "Failed checks :",
            failed_checks,
        )


if __name__ == "__main__":
    main()
