from __future__ import annotations

import unittest
from types import SimpleNamespace

from tech_blog_mlflow.evaluation_dataset_registry import (
    dataset_tags,
    dataset_tags_match,
    find_dataset,
    manifest_sha256,
)


def records():
    return [
        {
            "inputs": {"article_markdown": "# B", "theme": "T"},
            "expectations": {"human_scores": {"a": 2}, "publishable": False},
            "tags": {"variant": "b"},
        },
        {
            "inputs": {"article_markdown": "# A", "theme": "T"},
            "expectations": {"human_scores": {"a": 3}, "publishable": False},
            "tags": {"variant": "a"},
        },
    ]


class EvaluationDatasetContractTest(unittest.TestCase):
    def test_manifest_is_independent_of_record_order(self):
        self.assertEqual(manifest_sha256(records()), manifest_sha256(list(reversed(records()))))

    def test_manifest_changes_with_expectation(self):
        changed = records()
        changed[0]["expectations"]["human_scores"]["a"] = 4
        self.assertNotEqual(manifest_sha256(records()), manifest_sha256(changed))

    def test_tags_mark_dataset_immutable(self):
        tags = dataset_tags(records())
        self.assertEqual(tags["record_count"], "2")
        self.assertEqual(tags["immutable"], "true")

    def test_mlflow_managed_dataset_tag_is_allowed(self):
        expected = dataset_tags(records())
        actual = {**expected, "mlflow.user": "tera"}
        self.assertTrue(dataset_tags_match(actual, expected))

    def test_find_dataset_returns_exact_name(self):
        expected = SimpleNamespace(name="target")
        self.assertIs(
            find_dataset([SimpleNamespace(name="other"), expected], "target"),
            expected,
        )

    def test_duplicate_dataset_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "複数"):
            find_dataset(
                [SimpleNamespace(name="target"), SimpleNamespace(name="target")],
                "target",
            )


if __name__ == "__main__":
    unittest.main()
