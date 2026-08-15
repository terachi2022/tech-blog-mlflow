from __future__ import annotations

import unittest
from types import SimpleNamespace

from tech_blog_mlflow.judge_integration import (
    REGISTERED_SCORER_NAME,
    builtin_contract_matches,
    find_scorer,
    is_expected_oss_registration_error,
    scorer_contract,
)


class JudgeIntegrationContractTest(unittest.TestCase):
    def scorer(self, **updates):
        data = {
            "name": REGISTERED_SCORER_NAME,
            "builtin_scorer_class": "ResponseLength",
            "builtin_scorer_pydantic_data": {
                "min_length": 1800,
                "max_length": 7000,
                "unit": "chars",
            },
        }
        data.update(updates)
        return SimpleNamespace(
            name=data["name"], model_dump=lambda: data
        )

    def test_contract_is_fixed(self):
        contract = scorer_contract()
        self.assertEqual(contract["name"], "article_length_guard_v1")
        self.assertEqual(contract["min_length"], 1800)
        self.assertEqual(contract["max_length"], 7000)

    def test_builtin_contract_matches(self):
        self.assertTrue(builtin_contract_matches(self.scorer()))

    def test_changed_builtin_contract_is_rejected(self):
        scorer = self.scorer()
        scorer.model_dump()["builtin_scorer_pydantic_data"]["max_length"] = 1
        self.assertFalse(builtin_contract_matches(scorer))

    def test_find_scorer_returns_exact_name(self):
        expected = self.scorer()
        self.assertIs(
            find_scorer([SimpleNamespace(name="other"), expected], expected.name),
            expected,
        )

    def test_duplicate_scorers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "複数"):
            find_scorer([self.scorer(), self.scorer()], REGISTERED_SCORER_NAME)

    def test_oss_custom_scorer_rejection_is_recognized(self):
        message = (
            "Custom scorer registration (using @scorer decorator) is "
            "not supported outside of Databricks tracking environments"
        )
        self.assertTrue(is_expected_oss_registration_error(message))


if __name__ == "__main__":
    unittest.main()
