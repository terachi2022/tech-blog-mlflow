import argparse
import json
from pathlib import Path

from evaluation.local_judge import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_PROMPT_PATH,
    LocalArticleJudge,
)


DEFAULT_ARTICLE = (
    "articles/baseline_20260814_004017.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gemma Local LLM Judgeの"
            "単体動作を確認する"
        )
    )

    parser.add_argument(
        "--article",
        default=DEFAULT_ARTICLE,
        help="評価対象Markdown",
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
        help="Judgeの最大出力token数",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    article_path = Path(args.article)
    prompt_path = Path(args.prompt)

    if not article_path.exists():
        raise FileNotFoundError(
            "記事がありません: "
            f"{article_path}"
        )

    article = article_path.read_text(
        encoding="utf-8"
    )

    judge = LocalArticleJudge(
        model_id=args.model,
        prompt_path=prompt_path,
        max_tokens=args.max_tokens,
    )

    result = judge.evaluate(article)

    print()
    print("=" * 60)
    print("Validated Judge Result")
    print("=" * 60)

    print(
        json.dumps(
            result.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print(
        "Model load time :",
        f"{judge.load_elapsed_sec:.2f} sec",
    )

    print(
        "Generation time :",
        (
            f"{judge.total_generation_time_sec}"
            " sec"
        ),
    )


if __name__ == "__main__":
    main()
