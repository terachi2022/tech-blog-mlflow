from mlflow.entities import Feedback
from mlflow.genai import scorer

from evaluation.judge_schema_v2 import (
    DIMENSIONS,
)
from evaluation.local_judge_v2 import (
    LocalArticleJudgeV2,
)


def build_llm_judge_v2_scorer(
    judge: LocalArticleJudgeV2,
):
    @scorer(
        name="local_article_judge_v2"
    )
    def local_article_judge_v2(
        outputs: str,
    ) -> list[Feedback]:
        result = judge.evaluate(outputs)

        return [
            Feedback(
                name=dimension,
                value=(
                    result.dimension_score(
                        dimension
                    )
                ),
                rationale=(
                    result.dimension_rationale(
                        dimension
                    )
                ),
            )
            for dimension in DIMENSIONS
        ]

    return local_article_judge_v2
