from mlflow.entities import Feedback
from mlflow.genai import scorer

from evaluation.judge_schema_v2 import (
    DIMENSIONS,
)
from evaluation.local_judge_v2_1 import (
    LocalArticleJudgeV2_1,
)
from evaluation.scorers import (
    _public_external_urls,
)


def build_llm_judge_v2_1_1_scorer(
    judge: LocalArticleJudgeV2_1,
):
    @scorer(
        name="local_article_judge_v2_1_1"
    )
    def local_article_judge_v2_1_1(
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

        public_urls = (
            _public_external_urls(outputs)
        )

        original_citation_score = float(
            scores["citation_quality"]
        )

        guardrails: dict[str, object] = {
            "version": (
                "citation-url-guardrail-v2.1.1"
            ),
            "public_external_urls": (
                public_urls
            ),
            "public_external_url_count": (
                len(public_urls)
            ),
            "citation_override": False,
            "citation_original_score": (
                original_citation_score
            ),
            "citation_adjusted_score": (
                original_citation_score
            ),
            "reason": (
                "Public external URL found; "
                "LLM Judge score retained"
                if public_urls
                else "No public external URL"
            ),
        }

        # このプロジェクトでは、引用として
        # 少なくとも1件の公開URLを要求する。
        if not public_urls:
            scores[
                "citation_quality"
            ] = 1.0

            rationales[
                "citation_quality"
            ] = (
                "Deterministic guardrail: "
                "記事中に公開外部URLがないため、"
                "citation_qualityを1.0に"
                "補正しました。"
                "元のLLM評価: "
                f"{original_citation_score}。"
            )

            guardrails.update(
                {
                    "citation_override": True,
                    "citation_original_score": (
                        original_citation_score
                    ),
                    "citation_adjusted_score": (
                        1.0
                    ),
                }
            )

        # 生のJudge結果と補正後の結果を
        # 両方Artifactへ残す。
        if judge.records:
            judge.records[-1][
                "guardrails"
            ] = guardrails

            judge.records[-1][
                "adjusted_aggregate_scores"
            ] = scores

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

    return local_article_judge_v2_1_1
