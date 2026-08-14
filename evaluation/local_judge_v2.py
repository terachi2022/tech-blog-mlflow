from pathlib import Path
from typing import Any

from evaluation.judge_schema_v2 import (
    ArticleJudgeResultV2,
)
from evaluation.local_judge import (
    DEFAULT_JUDGE_MODEL,
    LocalArticleJudge,
)


DEFAULT_JUDGE_V2_PROMPT = Path(
    "prompts/article_judge_v2.md"
)


class LocalArticleJudgeV2(
    LocalArticleJudge
):
    """
    25サブ項目を評価するJudge-v2。
    """

    def __init__(
        self,
        model_id: str = (
            DEFAULT_JUDGE_MODEL
        ),
        prompt_path: Path = (
            DEFAULT_JUDGE_V2_PROMPT
        ),
        max_tokens: int = 3600,
    ) -> None:
        super().__init__(
            model_id=model_id,
            prompt_path=prompt_path,
            max_tokens=max_tokens,
        )

    @classmethod
    def _validate_response(
        cls,
        raw_response: str,
    ) -> ArticleJudgeResultV2:
        payload: dict[str, Any] = (
            cls._extract_json_object(
                raw_response
            )
        )

        return (
            ArticleJudgeResultV2
            .model_validate(payload)
        )
