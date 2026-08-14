"""combined-v2.1.1の回帰テスト。"""

from __future__ import annotations

import unittest

from evaluation.comparison_checks import (
    build_success_checks,
)
try:
    from evaluation.scorers import (
        _public_external_urls,
    )
except ModuleNotFoundError:
    _public_external_urls = None


class PublicExternalURLsTest(
    unittest.TestCase
):
    @unittest.skipIf(
        _public_external_urls is None,
        "mlflowが未導入の環境ではスキップ",
    )
    def test_public_urls_are_detected(
        self,
    ) -> None:
        article = """
        https://mlflow.org/docs/latest/
        https://github.com/mlflow/mlflow
        https://pypi.org/project/mlflow/
        """

        assert _public_external_urls is not None

        self.assertEqual(
            len(
                _public_external_urls(
                    article
                )
            ),
            3,
        )

    @unittest.skipIf(
        _public_external_urls is None,
        "mlflowが未導入の環境ではスキップ",
    )
    def test_local_urls_are_excluded(
        self,
    ) -> None:
        article = """
        http://127.0.0.1:5000/
        http://localhost:5000/
        http://192.168.1.10:5000/
        """

        assert _public_external_urls is not None

        self.assertEqual(
            _public_external_urls(article),
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
            "citation_quality/mean": 4.5,
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
