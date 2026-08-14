from pathlib import Path

from evaluation.local_judge import (
    DEFAULT_JUDGE_MODEL,
)
from evaluation.local_judge_v2 import (
    LocalArticleJudgeV2,
)


DEFAULT_JUDGE_V2_1_PROMPT = Path(
    "prompts/article_judge_v2_1.md"
)


class LocalArticleJudgeV2_1(
    LocalArticleJudgeV2
):
    def __init__(
        self,
        model_id: str = (
            DEFAULT_JUDGE_MODEL
        ),
        prompt_path: Path = (
            DEFAULT_JUDGE_V2_1_PROMPT
        ),
        max_tokens: int = 3600,
    ) -> None:
        super().__init__(
            model_id=model_id,
            prompt_path=prompt_path,
            max_tokens=max_tokens,
        )
