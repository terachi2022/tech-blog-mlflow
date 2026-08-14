from mlflow.entities import Feedback
from mlflow.genai import scorer

from evaluation.citation_calibration import (
    calibrate_citation_subscores,
)
from evaluation.judge_schema_v2 import (
    DIMENSIONS,
)
from evaluation.local_judge_v2_2 import (
    LocalArticleJudgeV2_2,
)


def build_llm_judge_v2_2_scorer(
    judge: LocalArticleJudgeV2_2,
):
    """25サブ項目JudgeとCitation校正を結合する。"""

    @scorer(
        name="local_article_judge_v2_2"
    )
    def local_article_judge_v2_2(
        outputs: str,
    ) -> list[Feedback]:
        result = judge.evaluate(outputs)

        scores = (
            result.aggregate_scores()
        )

        rationales = {
            dimension: (
                result.dimension_rationale(
                    dimension
                )
            )
            for dimension in DIMENSIONS
        }

        raw_citation_subscores = {
            name: item.score
            for name, item in (
                result.dimension_items(
                    "citation_quality"
                )
            )
        }

        (
            adjusted_citation_subscores,
            citation_audit,
        ) = calibrate_citation_subscores(
            article=outputs,
            raw_subscores=(
                raw_citation_subscores
            ),
        )

        raw_citation_score = float(
            scores["citation_quality"]
        )

        adjusted_citation_score = round(
            sum(
                adjusted_citation_subscores
                .values()
            )
            / len(
                adjusted_citation_subscores
            ),
            2,
        )

        scores[
            "citation_quality"
        ] = adjusted_citation_score

        adjustments = citation_audit[
            "adjustments"
        ]

        if adjustments:
            rationales[
                "citation_quality"
            ] = (
                rationales[
                    "citation_quality"
                ]
                + " | Deterministic "
                "citation calibration: "
                + str(adjustments)
            )

        citation_audit[
            "raw_score"
        ] = raw_citation_score
        citation_audit[
            "adjusted_score"
        ] = adjusted_citation_score

        # 生のJudge結果、サブスコア補正、
        # 集約後スコアをすべてArtifactへ残す。
        if judge.records:
            judge.records[-1][
                "citation_calibration"
            ] = citation_audit

            judge.records[-1][
                "adjusted_aggregate_scores"
            ] = dict(scores)

        return [
            Feedback(
                name=dimension,
                value=scores[dimension],
                rationale=(
                    rationales[dimension]
                ),
            )
            for dimension in DIMENSIONS
        ]

    return local_article_judge_v2_2
