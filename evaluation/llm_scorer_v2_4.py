"""Local Judge、内容校正、Citation校正を結合するScorer。"""

from __future__ import annotations

from mlflow.entities import Feedback
from mlflow.genai import scorer

from evaluation.citation_calibration_v2_3 import (
    calibrate_citation_subscores,
)
from evaluation.content_calibration_v2_4 import (
    aggregate_subscores,
    calibrate_content_subscores,
)
from evaluation.judge_schema_v2 import (
    DIMENSIONS,
)
from evaluation.local_judge_v2_4 import (
    LocalArticleJudgeV2_4,
)


def build_llm_judge_v2_4_scorer(
    judge: LocalArticleJudgeV2_4,
):
    """25サブ項目Judgeと2種類の決定論的校正を結合する。"""

    @scorer(
        name="local_article_judge_v2_4"
    )
    def local_article_judge_v2_4(
        outputs: str,
    ) -> list[Feedback]:
        result = judge.evaluate(outputs)

        raw_subscores = {
            dimension: {
                name: int(item.score)
                for name, item in result.dimension_items(
                    dimension
                )
            }
            for dimension in DIMENSIONS
        }
        item_rationales = {
            dimension: {
                name: str(item.rationale)
                for name, item in result.dimension_items(
                    dimension
                )
            }
            for dimension in DIMENSIONS
        }
        dimension_rationales = {
            dimension: result.dimension_rationale(
                dimension
            )
            for dimension in DIMENSIONS
        }

        (
            content_adjusted_subscores,
            content_audit,
        ) = calibrate_content_subscores(
            article=outputs,
            raw_subscores=raw_subscores,
            rationales=item_rationales,
        )

        raw_citation_subscores = raw_subscores[
            "citation_quality"
        ]
        (
            adjusted_citation_subscores,
            citation_audit,
        ) = calibrate_citation_subscores(
            article=outputs,
            raw_subscores=raw_citation_subscores,
        )
        content_adjusted_subscores[
            "citation_quality"
        ] = dict(adjusted_citation_subscores)

        raw_scores = aggregate_subscores(
            raw_subscores
        )
        adjusted_scores = aggregate_subscores(
            content_adjusted_subscores
        )

        citation_audit["raw_score"] = raw_scores[
            "citation_quality"
        ]
        citation_audit[
            "adjusted_score"
        ] = adjusted_scores["citation_quality"]

        for adjustment in content_audit[
            "adjustments"
        ]:
            dimension = adjustment["dimension"]
            dimension_rationales[dimension] = (
                dimension_rationales[dimension]
                + " | Deterministic content calibration: "
                + str(adjustment)
            )

        if citation_audit["adjustments"]:
            dimension_rationales[
                "citation_quality"
            ] = (
                dimension_rationales[
                    "citation_quality"
                ]
                + " | Deterministic citation calibration: "
                + str(citation_audit["adjustments"])
            )

        # Raw Judge、根拠、補正内容、最終値を同じRecordへ残す。
        if judge.records:
            record = judge.records[-1]
            record["raw_aggregate_scores"] = raw_scores
            record[
                "content_calibration"
            ] = content_audit
            record[
                "citation_calibration"
            ] = citation_audit
            record[
                "adjusted_aggregate_scores"
            ] = dict(adjusted_scores)

        return [
            Feedback(
                name=dimension,
                value=adjusted_scores[dimension],
                rationale=dimension_rationales[
                    dimension
                ],
            )
            for dimension in DIMENSIONS
        ]

    return local_article_judge_v2_4

