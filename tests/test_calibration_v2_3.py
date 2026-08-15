"""combined-v2.3の回帰テスト。"""

from __future__ import annotations

import unittest
from pathlib import Path
import sys
import types


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


try:
    import mlflow  # noqa: F401
except ModuleNotFoundError:
    mlflow_stub = types.ModuleType("mlflow")
    tracking_stub = types.ModuleType(
        "mlflow.tracking"
    )
    tracking_stub.MlflowClient = object
    sys.modules["mlflow"] = mlflow_stub
    sys.modules[
        "mlflow.tracking"
    ] = tracking_stub

from evaluation.citation_calibration_v2_3 import (
    calibrate_citation_subscores,
    markdown_public_links,
    public_external_urls,
)
from evaluation.comparison_checks import (
    build_success_checks,
)
from evaluation.compare_runs import (
    comparable_versions,
)


RAW_ALL_ONE = {
    "source_authority": 1,
    "claim_source_alignment": 1,
    "citation_coverage": 1,
    "link_context": 1,
}

JUDGE_PROMPT_PATH = (
    PROJECT_ROOT
    / "prompts"
    / "article_judge_v2_3.md"
)


class JudgePromptCalibrationTest(
    unittest.TestCase
):
    def test_v2_3_contains_required_calibration_rules(
        self,
    ) -> None:
        prompt = JUDGE_PROMPT_PATH.read_text(
            encoding="utf-8"
        )

        for fragment in (
            "Iris分類の創作結果と判定してはいけません",
            "別Download手順がないことをcode_completenessの欠点にしない",
            "try`が`pass`だけを囲み",
            "他環境とのbenchmark比較を必須としません",
            "自己計測値へ外部Citationを要求しない",
            "github.com/ml-explore/mlx-lm",
            "執筆指示の混入を可読性で評価する",
        ):
            self.assertIn(fragment, prompt)


class CitationCalibrationTest(
    unittest.TestCase
):
    def test_candidate_urls_are_detected(
        self,
    ) -> None:
        article = """
        ## 参考資料

        - [MLflow Tracking Quickstart](https://mlflow.org/docs/latest/ml/tracking/quickstart/)
        - [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
        - [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/)
        """

        self.assertEqual(
            len(public_external_urls(article)),
            3,
        )
        self.assertEqual(
            len(markdown_public_links(article)),
            3,
        )

    def test_official_sources_raise_authority(
        self,
    ) -> None:
        article = """
        - [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
        """

        adjusted, audit = (
            calibrate_citation_subscores(
                article=article,
                raw_subscores=RAW_ALL_ONE,
            )
        )

        self.assertEqual(
            adjusted["source_authority"],
            5,
        )
        self.assertEqual(
            audit["public_external_url_count"],
            1,
        )

    def test_astral_and_mlx_sources_are_primary(
        self,
    ) -> None:
        article = """
        - [uv Installation](https://docs.astral.sh/uv/getting-started/installation/)
        - [MLX-LM](https://github.com/ml-explore/mlx-lm)
        - [MLX](https://github.com/ml-explore/mlx)
        """

        adjusted, audit = (
            calibrate_citation_subscores(
                article=article,
                raw_subscores=RAW_ALL_ONE,
            )
        )

        self.assertEqual(
            adjusted["source_authority"],
            5,
        )
        self.assertEqual(
            audit["version"],
            "citation-calibration-v2.3.1",
        )

    def test_public_urls_are_unique(
        self,
    ) -> None:
        url = (
            "https://mlflow.org/docs/latest/"
            "ml/tracking/"
        )
        article = f"[Tracking]({url})\n[Tracking]({url})"

        self.assertEqual(
            public_external_urls(article),
            [url],
        )

    def test_descriptive_links_raise_context(
        self,
    ) -> None:
        article = """
        - [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
        """

        adjusted, _ = (
            calibrate_citation_subscores(
                article=article,
                raw_subscores=RAW_ALL_ONE,
            )
        )

        self.assertEqual(
            adjusted["link_context"],
            4,
        )

    def test_semantic_scores_are_not_inflated(
        self,
    ) -> None:
        article = """
        - [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
        """

        adjusted, _ = (
            calibrate_citation_subscores(
                article=article,
                raw_subscores=RAW_ALL_ONE,
            )
        )

        self.assertEqual(
            adjusted[
                "claim_source_alignment"
            ],
            1,
        )
        self.assertEqual(
            adjusted["citation_coverage"],
            1,
        )

    def test_no_url_forces_all_scores_to_one(
        self,
    ) -> None:
        raw = {
            name: 5
            for name in RAW_ALL_ONE
        }

        adjusted, audit = (
            calibrate_citation_subscores(
                article="# URLなし",
                raw_subscores=raw,
            )
        )

        self.assertEqual(
            adjusted,
            RAW_ALL_ONE,
        )
        self.assertEqual(
            audit["adjusted_score"],
            1.0,
        )

    def test_local_urls_are_excluded(
        self,
    ) -> None:
        article = """
        http://127.0.0.1:5000/
        http://localhost:5000/
        http://192.168.1.10:5000/
        """

        self.assertEqual(
            public_external_urls(article),
            [],
        )

    def test_single_label_internal_host_is_excluded(
        self,
    ) -> None:
        self.assertEqual(
            public_external_urls(
                "http://mlflow-server:5000"
            ),
            [],
        )


class ComparisonChecksTest(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.baseline = {
            "structure_score/mean": 1.0,
            "reproducibility_proxy/mean": 0.45,
            "technical_accuracy/mean": 4.5,
            "helpfulness/mean": 3.0,
            "reproducibility/mean": 1.4,
            "citation_quality/mean": 1.0,
            "readability_ja/mean": 3.25,
            "original_value/mean": 1.0,
        }

        self.candidate = {
            "structure_score/mean": 1.0,
            "reproducibility_proxy/mean": 1.0,
            "technical_accuracy/mean": 4.75,
            "helpfulness/mean": 4.0,
            "reproducibility/mean": 3.6,
            "citation_quality/mean": 3.0,
            "readability_ja/mean": 3.75,
            "original_value/mean": 1.0,
        }

    def test_readability_improvement_passes(
        self,
    ) -> None:
        checks = build_success_checks(
            self.baseline,
            self.candidate,
        )

        self.assertTrue(
            checks[
                "readability_not_regressed"
            ]
        )

    def test_citation_target_passes_at_three(
        self,
    ) -> None:
        checks = build_success_checks(
            self.baseline,
            self.candidate,
        )

        self.assertTrue(
            checks[
                "citation_quality_target"
            ]
        )

    def test_proxy_maximum_can_pass_against_maximum(
        self,
    ) -> None:
        baseline = dict(self.baseline)
        candidate = dict(self.candidate)
        baseline["reproducibility_proxy/mean"] = 1.0
        candidate["reproducibility_proxy/mean"] = 1.0

        checks = build_success_checks(baseline, candidate)

        self.assertTrue(
            checks["reproducibility_proxy_target_and_not_regressed"]
        )

    def test_proxy_below_target_fails(
        self,
    ) -> None:
        candidate = dict(self.candidate)
        candidate["reproducibility_proxy/mean"] = 0.75

        checks = build_success_checks(self.baseline, candidate)

        self.assertFalse(
            checks["reproducibility_proxy_target_and_not_regressed"]
        )

    def test_original_value_target_fails(
        self,
    ) -> None:
        checks = build_success_checks(
            self.baseline,
            self.candidate,
        )

        self.assertFalse(
            checks[
                "original_value_target"
            ]
        )

    def test_regression_is_detected(
        self,
    ) -> None:
        candidate = dict(self.candidate)
        candidate[
            "technical_accuracy/mean"
        ] = 4.25

        checks = build_success_checks(
            self.baseline,
            candidate,
        )

        self.assertFalse(
            checks[
                "technical_accuracy_not_regressed"
            ]
        )


class ComparableVersionsTest(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.versions = {
            "combined_version": "combined-v2.3.1",
            "judge_prompt_version": (
                "article-judge-v2.3"
            ),
            "code_scorer_version": (
                "code-scorer-v3.1"
            ),
            "citation_calibration_version": (
                "citation-calibration-v2.3.1"
            ),
            "content_calibration_version": (
                "content-calibration-v2.4"
            ),
        }

    def test_same_versions_pass(
        self,
    ) -> None:
        result = comparable_versions(
            self.versions,
            dict(self.versions),
        )

        self.assertEqual(
            result,
            self.versions,
        )

    def test_mixed_judge_versions_fail(
        self,
    ) -> None:
        candidate = dict(self.versions)
        candidate[
            "judge_prompt_version"
        ] = "article-judge-v2.2"

        with self.assertRaises(ValueError):
            comparable_versions(
                self.versions,
                candidate,
            )


if __name__ == "__main__":
    unittest.main()
