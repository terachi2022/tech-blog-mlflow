"""GPT-OSS-Swallow Candidateで記事を生成してMLflowへ記録する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from tech_blog_mlflow.article_v3_checks import ARTICLE_MAX_CHARS, ARTICLE_MIN_CHARS, article_checks
from tech_blog_mlflow.candidate_models import GENERATOR, PIPELINE_VERSION
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "tech-blog-generation"
PROMPT_PATH = Path("prompts/article_generation_v3_5_2.md")
PROMPT_VERSION = "article-v3.5.2"
CONFIG_VERSION = "candidate-generation-v1.0.0"
DEFAULT_THEME = "MLflowを使って機械学習の実験を管理する方法"
SYSTEM_PROMPT = (
    "指示された要件に従い、日本語の技術記事本文だけをMarkdownで出力してください。"
    "Prompt中の執筆指示は本文へ転記しないでください。架空の実行結果を作らず、"
    "提供された観測値だけを実測値として扱ってください。"
)


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def strip_outer_markdown_fence(text: str) -> str:
    result = text.strip()
    for prefix in ("```markdown\n", "```md\n", "```\n"):
        if result.startswith(prefix) and result.endswith("```"):
            return result[len(prefix):-3].strip()
    return result


def extract_final_channel(text: str) -> str:
    """Harmony形式から公開用final本文だけを取り出す。"""
    marker = "<|channel|>final<|message|>"
    if marker not in text:
        if "<|channel|>analysis<|message|>" in text:
            raise ValueError("生成がanalysis channel内で終了し、final本文がありません。")
        return text
    final = text.rsplit(marker, 1)[1]
    for terminator in ("<|return|>", "<|end|>"):
        final = final.split(terminator, 1)[0]
    return final.strip()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_prompt(path: Path, theme: str) -> str:
    rendered = path.read_text(encoding="utf-8")
    replacements = {
        "{{THEME}}": theme,
        "{{MACOS_VERSION}}": platform.mac_ver()[0] or "unknown",
        "{{PYTHON_VERSION}}": platform.python_version(),
        "{{MLFLOW_VERSION}}": package_version("mlflow"),
        "{{MLX_LM_VERSION}}": package_version("mlx-lm"),
    }
    for marker, value in replacements.items():
        rendered = rendered.replace(marker, value)
    if "{{" in rendered or "}}" in rendered:
        raise ValueError("Promptに未置換placeholderがあります。")
    return rendered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", default=DEFAULT_THEME)
    parser.add_argument("--model", default=GENERATOR.model_id)
    parser.add_argument("--prompt", type=Path, default=PROMPT_PATH)
    parser.add_argument("--max-tokens", type=int, default=GENERATOR.max_tokens)
    parser.add_argument("--run-name", default="candidate-swallow-120b-generation-v1")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rendered_prompt = render_prompt(args.prompt, args.theme)
    plan = {
        "pipeline_version": PIPELINE_VERSION,
        "model": args.model,
        "runtime": GENERATOR.runtime,
        "prompt": str(args.prompt),
        "prompt_version": PROMPT_VERSION,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "theme": args.theme,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    import mlflow
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    articles_dir = Path("articles")
    results_dir = Path("generation_results")
    articles_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    rendered_path = results_dir / f"candidate_rendered_prompt_{timestamp}.md"
    rendered_path.write_text(rendered_prompt, encoding="utf-8")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mx.random.seed(42)
    mx.reset_peak_memory()
    sampler = make_sampler(temp=0.0)

    with mlflow.start_run(run_name=args.run_name) as run:
        load_started = time.perf_counter()
        model, tokenizer = load(args.model)
        load_sec = time.perf_counter() - load_started
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": rendered_prompt},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            reasoning_effort="low",
        )
        started = time.perf_counter()
        raw = generate(
            model, tokenizer, prompt=formatted, max_tokens=args.max_tokens,
            sampler=sampler, verbose=False,
        )
        generation_sec = time.perf_counter() - started
        article = strip_outer_markdown_fence(extract_final_channel(raw)).rstrip() + "\n"
        generated_tokens = len(tokenizer.encode(raw))
        output_tokens = len(tokenizer.encode(article))
        article_path = articles_dir / f"candidate_swallow_{timestamp}.md"
        article_path.write_text(article, encoding="utf-8")
        checks = article_checks(article, rendered_prompt)
        checks["output_tokens_below_safety_limit"] = generated_tokens < args.max_tokens - 64
        checks["article_length_in_range"] = ARTICLE_MIN_CHARS <= len(article) <= ARTICLE_MAX_CHARS
        failed = [name for name, passed in checks.items() if not passed]
        metadata = {
            **plan,
            "run_id": run.info.run_id,
            "article_path": str(article_path),
            "article_sha256": sha256(article),
            "rendered_prompt_path": str(rendered_path),
            "prompt_sha256": sha256(rendered_prompt),
            "formatted_prompt_sha256": sha256(formatted),
            "system_prompt_sha256": sha256(SYSTEM_PROMPT),
            "generation_config_version": CONFIG_VERSION,
            "generation_parameters": {"max_tokens": args.max_tokens, "temperature": 0.0, "seed": 42},
            "metrics": {
                "model_load_time_sec": round(load_sec, 3),
                "generation_time_sec": round(generation_sec, 3),
                "article_length": len(article),
                "output_tokens": output_tokens,
                "generated_tokens_including_reasoning": generated_tokens,
                "peak_memory_gb": round(mx.get_peak_memory() / 1_000_000_000, 3),
            },
            "prechecks": checks,
            "all_prechecks_passed": not failed,
            "failed_prechecks": failed,
        }
        metadata_path = results_dir / f"candidate_swallow_generation_{timestamp}_{run.info.run_id}.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        mlflow.log_params({
            "model": args.model, "prompt_version": PROMPT_VERSION,
            "generation_config_version": CONFIG_VERSION, "max_tokens": args.max_tokens,
            "temperature": 0.0, "article_sha256": metadata["article_sha256"],
            "pipeline_version": PIPELINE_VERSION, "theme": args.theme,
        })
        mlflow.set_tags({"stage": "candidate-generation", "article_variant": "candidate-swallow-120b"})
        mlflow.log_metrics(metadata["metrics"] | {"precheck_all_passed": int(not failed)})
        mlflow.log_artifact(str(article_path), artifact_path="articles")
        mlflow.log_artifact(str(rendered_path), artifact_path="prompts")
        mlflow.log_artifact(str(metadata_path), artifact_path="generation_metadata")
        print("Run ID    :", run.info.run_id)
        print("Article   :", article_path)
        print("Metadata  :", metadata_path)
        print("Prechecks :", checks)


if __name__ == "__main__":
    main()
