"""STEP 4 Skills-only実験の回帰テスト。"""

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

try:
    import mlx.core  # noqa: F401
except (ModuleNotFoundError, ImportError):
    # LinuxでPackageは存在してもlibmlx.soをLoadできない場合を含む。
    mlx_stub = types.ModuleType("mlx")
    mlx_core_stub = types.ModuleType("mlx.core")
    mlx_stub.core = mlx_core_stub
    sys.modules["mlx"] = mlx_stub
    sys.modules["mlx.core"] = mlx_core_stub

try:
    import mlx_lm  # noqa: F401
except ModuleNotFoundError:
    mlx_lm_stub = types.ModuleType("mlx_lm")
    mlx_lm_stub.generate = lambda *args, **kwargs: ""
    mlx_lm_stub.load = lambda *args, **kwargs: (None, None)
    sample_stub = types.ModuleType("mlx_lm.sample_utils")
    sample_stub.make_sampler = lambda **kwargs: None
    sys.modules["mlx_lm"] = mlx_lm_stub
    sys.modules["mlx_lm.sample_utils"] = sample_stub


from evaluation.compare_skill_experiment import (  # noqa: E402
    ADOPTED_GENERATION_RUN_ID,
    validate_controlled_change,
)
from evaluation.evaluate_skill_candidate import (  # noqa: E402
    EXPECTED_GENERATION_PARAMETERS,
    build_evaluation_command,
)
from evaluation.skill_experiment_checks import (  # noqa: E402
    build_skill_success_checks,
)
from tech_blog_mlflow.generate_with_skill import (  # noqa: E402
    ADOPTED_BASE_PROMPT_SHA256,
    SKILL_NAME,
    SKILL_VERSION,
    build_effective_prompt,
    load_skill,
)


SKILL_PATH = (
    PROJECT_ROOT
    / "skills"
    / "technical-blog-quality"
    / "SKILL.md"
)


def quality_metrics() -> dict[str, float]:
    return {
        "structure_score/mean": 1.0,
        "reproducibility_proxy/mean": 0.8,
        "has_prerequisites/mean": 1.0,
        "has_version_info/mean": 1.0,
        "has_failure_cases/mean": 1.0,
        "technical_accuracy/mean": 5.0,
        "helpfulness/mean": 4.5,
        "reproducibility/mean": 4.8,
        "citation_quality/mean": 4.5,
        "readability_ja/mean": 4.5,
        "original_value/mean": 4.0,
    }


def generation_metadata() -> tuple[dict, dict]:
    common = {
        "model": "Qwen/Qwen3-8B-MLX-4bit",
        "theme": "MLflowを使って機械学習の実験を管理する方法",
        "prompt_version": "article-v3.5.2",
        "system_prompt_sha256": "system-sha",
        "generation_parameters": dict(
            EXPECTED_GENERATION_PARAMETERS
        ),
        "all_prechecks_passed": True,
    }
    baseline = {
        **common,
        "run_id": ADOPTED_GENERATION_RUN_ID,
        "prompt_sha256": ADOPTED_BASE_PROMPT_SHA256,
    }
    candidate = {
        **common,
        "run_id": "candidate-generation-run",
        "prompt_sha256": "effective-prompt-sha",
        "base_prompt_sha256": ADOPTED_BASE_PROMPT_SHA256,
        "failed_prechecks": [],
        "skill": {
            "enabled": True,
            "name": SKILL_NAME,
            "version": SKILL_VERSION,
            "sha256": "skill-sha",
        },
        "adopted_candidate": {
            "generation_run_id": ADOPTED_GENERATION_RUN_ID,
        },
        "controlled_change": {
            "name": "skills",
            "baseline": False,
            "candidate": True,
            "base_prompt_changed": False,
            "max_tokens_changed": False,
        },
    }
    return baseline, candidate


class SkillFileTest(unittest.TestCase):
    def test_skill_has_expected_frontmatter_and_workflow(self) -> None:
        skill = SKILL_PATH.read_text(encoding="utf-8")
        self.assertTrue(skill.startswith("---\n"))
        self.assertIn(
            f"name: {SKILL_NAME}",
            skill,
        )
        for fragment in (
            "Observed",
            "Referenced",
            "Unverified",
            "実行経路を確認する",
            "Evidenceと主張を対応させる",
            "出力前に内部レビューする",
        ):
            self.assertIn(fragment, skill)

    def test_effective_prompt_adds_skill_once(self) -> None:
        base = "# Base\n\n本文\n"
        skill = SKILL_PATH.read_text(encoding="utf-8")
        effective = build_effective_prompt(base, skill)
        self.assertTrue(effective.startswith(base.rstrip()))
        self.assertEqual(
            effective.count("# Applied Skill"),
            1,
        )
        self.assertEqual(
            effective.count(f"name: {SKILL_NAME}"),
            1,
        )

    def test_duplicate_skill_marker_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_effective_prompt(
                "# Applied Skill\n",
                "---\nname: test\n---\n",
            )

    def test_project_skill_loads(self) -> None:
        original = Path.cwd()
        try:
            import os
            os.chdir(PROJECT_ROOT)
            skill = load_skill()
        finally:
            os.chdir(original)
        self.assertTrue(skill.endswith("\n"))
        self.assertIn(f"name: {SKILL_NAME}", skill)


class ControlledChangeTest(unittest.TestCase):
    def test_skills_only_metadata_passes(self) -> None:
        baseline, candidate = generation_metadata()
        control = validate_controlled_change(
            baseline,
            candidate,
        )
        self.assertEqual(
            control["changed_variable"],
            "skills",
        )
        self.assertTrue(
            control["candidate_prechecks_passed"]
        )

    def test_changed_max_tokens_is_rejected(self) -> None:
        baseline, candidate = generation_metadata()
        candidate["generation_parameters"] = {
            **candidate["generation_parameters"],
            "max_tokens": 8192,
        }
        with self.assertRaises(ValueError):
            validate_controlled_change(
                baseline,
                candidate,
            )

    def test_changed_base_prompt_is_rejected(self) -> None:
        baseline, candidate = generation_metadata()
        candidate["base_prompt_sha256"] = "changed"
        with self.assertRaises(ValueError):
            validate_controlled_change(
                baseline,
                candidate,
            )


class SkillSuccessChecksTest(unittest.TestCase):
    def test_non_regression_and_one_improvement_pass(self) -> None:
        baseline = quality_metrics()
        candidate = dict(baseline)
        candidate["original_value/mean"] = 4.25
        checks = build_skill_success_checks(
            baseline,
            candidate,
            candidate_prechecks_passed=True,
        )
        self.assertTrue(all(checks.values()))

    def test_equal_scores_do_not_prove_skill_value(self) -> None:
        baseline = quality_metrics()
        checks = build_skill_success_checks(
            baseline,
            dict(baseline),
            candidate_prechecks_passed=True,
        )
        self.assertFalse(
            checks["skill_value_demonstrated"]
        )

    def test_any_quality_regression_fails(self) -> None:
        baseline = quality_metrics()
        candidate = dict(baseline)
        candidate["helpfulness/mean"] = 4.25
        candidate["original_value/mean"] = 4.25
        checks = build_skill_success_checks(
            baseline,
            candidate,
            candidate_prechecks_passed=True,
        )
        self.assertFalse(
            checks["helpfulness_not_regressed"]
        )

    def test_failed_precheck_fails(self) -> None:
        baseline = quality_metrics()
        candidate = dict(baseline)
        candidate["original_value/mean"] = 4.25
        checks = build_skill_success_checks(
            baseline,
            candidate,
            candidate_prechecks_passed=False,
        )
        self.assertFalse(
            checks["candidate_prechecks_passed"]
        )


class EvaluationCommandTest(unittest.TestCase):
    def test_command_uses_frozen_evaluator(self) -> None:
        command = build_evaluation_command(
            metadata={
                "article_path": "articles/skill.md",
                "run_id": "generation-run",
            },
            run_name="skill-eval",
        )
        self.assertIn(
            "evaluation.evaluate_combined_v2_4",
            command,
        )
        self.assertIn("skill-v1", command)
        self.assertIn("article-v3.5.2", command)


if __name__ == "__main__":
    unittest.main()
