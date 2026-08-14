from datetime import datetime
from pathlib import Path
import time

import mlflow
import mlx.core as mx

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


MODEL_NAME = "Qwen/Qwen3-8B-MLX-4bit"
EXPERIMENT_NAME = "tech-blog-generation"
TRACKING_URI = "http://127.0.0.1:5000"

PROMPT_FILE = Path("prompts/baseline.md")
ARTICLE_DIR = Path("articles")

MAX_TOKENS = 2048
SEED = 42


def load_prompt(theme: str) -> str:
    template = PROMPT_FILE.read_text(encoding="utf-8")
    return template.replace("{{theme}}", theme)


def main() -> None:
    theme = "MLflowを使って機械学習の実験を管理する方法"

    ARTICLE_DIR.mkdir(exist_ok=True)

    prompt_text = load_prompt(theme)

    print("Loading model...")
    model, tokenizer = load(MODEL_NAME)

    messages = [
        {
            "role": "user",
            "content": prompt_text,
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # 比較実験の再現性を上げる
    mx.random.seed(SEED)

    # Baselineはgreedy generation相当に固定
    sampler = make_sampler(temp=0.0)

    # このRunにおけるMLXメモリピークを測定
    mx.reset_peak_memory()

    print("Generating article...")

    start_time = time.perf_counter()

    article = generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=MAX_TOKENS,
        sampler=sampler,
        verbose=False,
    )

    generation_time = time.perf_counter() - start_time

    peak_memory_gb = mx.get_peak_memory() / (1024 ** 3)

    output_tokens = len(
        tokenizer.encode(
            article,
            add_special_tokens=False,
        )
    )

    chars = len(article)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    article_path = ARTICLE_DIR / f"baseline_{timestamp}.md"

    article_path.write_text(
        article,
        encoding="utf-8",
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="baseline-qwen-v1") as run:

        mlflow.log_params(
            {
                "model": MODEL_NAME,
                "prompt_version": "baseline-v1",
                "language": "ja",
                "thinking": False,
                "skills": False,
                "subagents": False,
                "rag": False,
                "lora": False,
                "max_tokens": MAX_TOKENS,
                "seed": SEED,
                "temperature": 0.0,
                "theme": theme,
            }
        )

        mlflow.log_metrics(
            {
                "generation_time_sec": generation_time,
                "article_length_chars": chars,
                "output_tokens": output_tokens,
                "tokens_per_sec": (
                    output_tokens / generation_time
                    if generation_time > 0
                    else 0
                ),
                "peak_memory_gb": peak_memory_gb,
            }
        )

        mlflow.set_tags(
            {
                "stage": "baseline",
                "generator": "mlx-lm",
            }
        )

        mlflow.log_artifact(
            str(article_path),
            artifact_path="article",
        )

        mlflow.log_artifact(
            str(PROMPT_FILE),
            artifact_path="prompt",
        )

        print()
        print("Run ID              :", run.info.run_id)
        print("Article              :", article_path)
        print("Generation time      :", f"{generation_time:.2f} sec")
        print("Article length       :", chars, "chars")
        print("Output tokens        :", output_tokens)
        print("Tokens/sec           :", f"{output_tokens / generation_time:.2f}")
        print("Peak memory          :", f"{peak_memory_gb:.3f} GB")


if __name__ == "__main__":
    main()
