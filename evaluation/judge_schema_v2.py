from typing import Final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


DIMENSIONS: Final[tuple[str, ...]] = (
    "technical_accuracy",
    "helpfulness",
    "reproducibility",
    "citation_quality",
    "readability_ja",
    "original_value",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )


class SubScore(StrictModel):
    score: int = Field(
        ge=1,
        le=5,
    )

    rationale: str = Field(
        min_length=5,
    )


class TechnicalAccuracy(StrictModel):
    conceptual_correctness: SubScore
    api_command_correctness: SubScore
    internal_consistency: SubScore
    unsupported_claim_control: SubScore


class Helpfulness(StrictModel):
    goal_clarity: SubScore
    actionability: SubScore
    audience_fit: SubScore
    troubleshooting_value: SubScore


class Reproducibility(StrictModel):
    environment_specificity: SubScore
    dependency_completeness: SubScore
    code_completeness: SubScore
    execution_sequence: SubScore
    verification_clarity: SubScore


class CitationQuality(StrictModel):
    source_authority: SubScore
    claim_source_alignment: SubScore
    citation_coverage: SubScore
    link_context: SubScore


class ReadabilityJa(StrictModel):
    structure_flow: SubScore
    sentence_clarity: SubScore
    terminology_explanation: SubScore
    information_density: SubScore


class OriginalValue(StrictModel):
    concrete_evidence: SubScore
    failure_analysis: SubScore
    comparison_insight: SubScore
    environment_specific_insight: SubScore


class ArticleJudgeResultV2(StrictModel):
    technical_accuracy: TechnicalAccuracy
    helpfulness: Helpfulness
    reproducibility: Reproducibility
    citation_quality: CitationQuality
    readability_ja: ReadabilityJa
    original_value: OriginalValue

    def dimension_items(
        self,
        dimension_name: str,
    ) -> list[tuple[str, SubScore]]:
        dimension = getattr(
            self,
            dimension_name,
        )

        field_names = (
            dimension.__class__
            .model_fields
            .keys()
        )

        return [
            (
                field_name,
                getattr(
                    dimension,
                    field_name,
                ),
            )
            for field_name in field_names
        ]

    def dimension_score(
        self,
        dimension_name: str,
    ) -> float:
        items = self.dimension_items(
            dimension_name
        )

        return round(
            sum(
                item.score
                for _, item in items
            )
            / len(items),
            2,
        )

    def aggregate_scores(
        self,
    ) -> dict[str, float]:
        return {
            dimension: self.dimension_score(
                dimension
            )
            for dimension in DIMENSIONS
        }

    def dimension_rationale(
        self,
        dimension_name: str,
    ) -> str:
        items = self.dimension_items(
            dimension_name
        )

        return " | ".join(
            (
                f"{name}={item.score}: "
                f"{item.rationale}"
            )
            for name, item in items
        )

    def subscore_count(self) -> int:
        return sum(
            len(
                self.dimension_items(
                    dimension
                )
            )
            for dimension in DIMENSIONS
        )
