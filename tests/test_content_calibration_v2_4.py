"""Content calibration v2.4の回帰テスト。"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


try:
    import mlflow  # noqa: F401
except ModuleNotFoundError:
    mlflow_stub = types.ModuleType("mlflow")
    tracking_stub = types.ModuleType("mlflow.tracking")
    tracking_stub.MlflowClient = object
    sys.modules["mlflow"] = mlflow_stub
    sys.modules["mlflow.tracking"] = tracking_stub

from evaluation.compare_runs import comparable_versions
from evaluation.content_calibration_v2_4 import (
    aggregate_subscores,
    calibrate_content_subscores,
)


JUDGE_PROMPT_PATH = (
    PROJECT_ROOT / "prompts" / "article_judge_v2_4.md"
)


def base_subscores() -> dict[str, dict[str, int]]:
    return {
        "technical_accuracy": {
            "conceptual_correctness": 5,
            "api_command_correctness": 4,
            "internal_consistency": 5,
            "unsupported_claim_control": 5,
        },
        "helpfulness": {
            "goal_clarity": 5,
            "actionability": 4,
            "audience_fit": 3,
            "troubleshooting_value": 3,
        },
        "reproducibility": {
            "environment_specificity": 4,
        },
        "citation_quality": {
            "source_authority": 4,
        },
        "readability_ja": {
            "structure_flow": 4,
        },
        "original_value": {
            "concrete_evidence": 4,
        },
    }


def base_rationales() -> dict[str, dict[str, str]]:
    return {
        "technical_accuracy": {
            "api_command_correctness": (
                "APIとコマンドの使用法は概ね正しい。"
                "ただし、具体的なエラーハンドリングや"
                "例外処理の記述がない。"
            ),
        },
        "helpfulness": {
            "troubleshooting_value": (
                "代表的なエラーと対処が記載されている。"
            ),
        },
    }


def complete_evidence() -> dict[str, bool]:
    return {
        "has_complete_train_code": True,
        "has_required_shell_commands": True,
        "exception_handling_is_valid": True,
        "has_complete_troubleshooting": True,
        "has_specific_error_messages": True,
    }


class ContentCalibrationTest(unittest.TestCase):
    def test_exception_only_api_penalty_is_corrected(self) -> None:
        adjusted, audit = calibrate_content_subscores(
            article="# test",
            raw_subscores=base_subscores(),
            rationales=base_rationales(),
            evidence=complete_evidence(),
        )

        self.assertEqual(
            adjusted["technical_accuracy"]["api_command_correctness"],
            5,
        )
        self.assertEqual(
            audit["version"],
            "content-calibration-v2.4",
        )

    def test_actual_api_error_is_not_corrected(self) -> None:
        rationales = base_rationales()
        rationales["technical_accuracy"]["api_command_correctness"] = (
            "APIの使用法は正しい箇所もあるが、引数に誤りがあり、"
            "そのままでは実行できない。例外処理もない。"
        )

        adjusted, audit = calibrate_content_subscores(
            article="# test",
            raw_subscores=base_subscores(),
            rationales=rationales,
            evidence=complete_evidence(),
        )

        self.assertEqual(
            adjusted["technical_accuracy"]["api_command_correctness"],
            4,
        )
        self.assertFalse(
            any(
                item["subscore"] == "api_command_correctness"
                for item in audit["adjustments"]
            )
        )

    def test_incomplete_api_evidence_is_not_corrected(self) -> None:
        evidence = complete_evidence()
        evidence["has_complete_train_code"] = False

        adjusted, _ = calibrate_content_subscores(
            article="# test",
            raw_subscores=base_subscores(),
            rationales=base_rationales(),
            evidence=evidence,
        )

        self.assertEqual(
            adjusted["technical_accuracy"]["api_command_correctness"],
            4,
        )

    def test_complete_troubleshooting_is_raised_to_four(self) -> None:
        adjusted, _ = calibrate_content_subscores(
            article="# test",
            raw_subscores=base_subscores(),
            rationales=base_rationales(),
            evidence=complete_evidence(),
        )

        self.assertEqual(
            adjusted["helpfulness"]["troubleshooting_value"],
            4,
        )

    def test_incomplete_troubleshooting_is_not_corrected(self) -> None:
        evidence = complete_evidence()
        evidence["has_complete_troubleshooting"] = False

        adjusted, _ = calibrate_content_subscores(
            article="# test",
            raw_subscores=base_subscores(),
            rationales=base_rationales(),
            evidence=evidence,
        )

        self.assertEqual(
            adjusted["helpfulness"]["troubleshooting_value"],
            3,
        )

    def test_aggregate_scores_reflect_only_audited_changes(self) -> None:
        adjusted, _ = calibrate_content_subscores(
            article="# test",
            raw_subscores=base_subscores(),
            rationales=base_rationales(),
            evidence=complete_evidence(),
        )
        aggregates = aggregate_subscores(adjusted)

        self.assertEqual(aggregates["technical_accuracy"], 5.0)
        self.assertEqual(aggregates["helpfulness"], 4.0)
        self.assertEqual(
            adjusted["helpfulness"]["audience_fit"],
            3,
        )


class JudgePromptV24Test(unittest.TestCase):
    def test_prompt_has_v2_4_anchors(self) -> None:
        prompt = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")

        for fragment in (
            "例外処理を要求しない",
            "3種類以上の代表的Error",
            "4種類以上について上記5要素",
            "actionabilityとaudience_fitを分離する",
        ):
            self.assertIn(fragment, prompt)


class ComparableVersionsV24Test(unittest.TestCase):
    def setUp(self) -> None:
        self.versions = {
            "combined_version": "combined-v2.4.0",
            "judge_prompt_version": "article-judge-v2.4",
            "code_scorer_version": "code-scorer-v3.1",
            "citation_calibration_version": (
                "citation-calibration-v2.3.1"
            ),
            "content_calibration_version": (
                "content-calibration-v2.4"
            ),
        }

    def test_same_versions_pass(self) -> None:
        self.assertEqual(
            comparable_versions(self.versions, dict(self.versions)),
            self.versions,
        )

    def test_mixed_content_calibration_versions_fail(self) -> None:
        candidate = dict(self.versions)
        candidate["content_calibration_version"] = (
            "content-calibration-v2.3"
        )

        with self.assertRaises(ValueError):
            comparable_versions(self.versions, candidate)


if __name__ == "__main__":
    unittest.main()
