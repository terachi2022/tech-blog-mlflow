"""combined-v2.2の回帰テスト。"""

from __future__ import annotations

import unittest

from evaluation.citation_calibration import (
    calibrate_citation_subscores,
    markdown_public_links,
    public_external_urls,
)
from evaluation.comparison_checks import (
    build_success_checks,
)


RAW_ALL_ONE = {
    "source_authority": 1,
    "claim_source_alignment": 1,
    "citation_coverage": 1,
    "link_context": 1,
}


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


if __name__ == "__main__":
    unittest.main()
