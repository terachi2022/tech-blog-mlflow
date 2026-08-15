from pathlib import Path

from evaluation.local_judge import (
    DEFAULT_JUDGE_MODEL,
)
from evaluation.local_judge_v2 import (
    LocalArticleJudgeV2,
)


DEFAULT_JUDGE_V2_3_PROMPT = Path(
    "prompts/article_judge_v2_3.md"
)


class LocalArticleJudgeV2_3(
    LocalArticleJudgeV2
):
    """内容とCitationを再校正したJudge-v2.3。"""

    def __init__(
        self,
        model_id: str = (
            DEFAULT_JUDGE_MODEL
        ),
        prompt_path: Path = (
            DEFAULT_JUDGE_V2_3_PROMPT
        ),
        max_tokens: int = 3600,
    ) -> None:
        super().__init__(
            model_id=model_id,
            prompt_path=prompt_path,
            max_tokens=max_tokens,
        )
