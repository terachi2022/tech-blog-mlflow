from typing import Final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


JUDGE_DIMENSIONS: Final[tuple[str, ...]] = (
    "technical_accuracy",
    "helpfulness",
    "reproducibility",
    "citation_quality",
    "readability_ja",
    "original_value",
)


class JudgeItem(BaseModel):
    """
    1項目分のLLM Judge評価。
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    score: int = Field(
        ge=1,
        le=5,
        description="1〜5の整数評価",
    )

    rationale: str = Field(
        min_length=5,
        description="日本語の評価理由",
    )


class ArticleJudgeResult(BaseModel):
    """
    技術記事に対する6項目の評価結果。
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    technical_accuracy: JudgeItem
    helpfulness: JudgeItem
    reproducibility: JudgeItem
    citation_quality: JudgeItem
    readability_ja: JudgeItem
    original_value: JudgeItem
