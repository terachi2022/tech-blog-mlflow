from __future__ import annotations

import unittest
from types import SimpleNamespace

from tech_blog_mlflow.external_model_registry import (
    ExternalModelSpec,
    find_matching_model,
    model_has_prompt_link,
    model_tags,
    spec_identity,
)


def spec() -> ExternalModelSpec:
    return ExternalModelSpec(
        role="generator",
        name="generator-model",
        huggingface_id="org/model-4bit",
        source_run_id="run-1",
        prompt_name="generator-prompt",
        prompt_version=1,
        model_type="text-generation",
        params={"runtime": "mlx-lm"},
    )


class ExternalModelContractTest(unittest.TestCase):
    def test_identity_is_deterministic(self):
        self.assertEqual(spec_identity(spec()), spec_identity(spec()))
        self.assertEqual(len(spec_identity(spec())), 64)

    def test_tags_make_external_location_explicit(self):
        tags = model_tags(spec())
        self.assertEqual(tags["weights_location"], "huggingface-cache")
        self.assertEqual(tags["weights_copied_to_mlflow"], "false")

    def test_matching_model_is_reused(self):
        item = spec()
        model = SimpleNamespace(tags={"spec_sha256": spec_identity(item)})
        client = SimpleNamespace(
            search_logged_models=lambda **_: [model]
        )
        self.assertIs(find_matching_model(client, "1", item), model)

    def test_same_name_with_changed_spec_is_rejected(self):
        item = spec()
        model = SimpleNamespace(tags={"spec_sha256": "different"})
        client = SimpleNamespace(
            search_logged_models=lambda **_: [model]
        )
        with self.assertRaisesRegex(ValueError, "上書き"):
            find_matching_model(client, "1", item)

    def test_prompt_link_is_read_from_model_tag(self):
        item = spec()
        model = SimpleNamespace(
            model_id="m-1",
            tags={
                "mlflow.linkedPrompts":
                    '[{"name":"generator-prompt","version":"1"}]'
            },
        )
        self.assertTrue(model_has_prompt_link(model, item))

    def test_invalid_prompt_link_json_is_rejected(self):
        model = SimpleNamespace(
            model_id="m-1", tags={"mlflow.linkedPrompts": "{"}
        )
        with self.assertRaisesRegex(ValueError, "JSON"):
            model_has_prompt_link(model, spec())


if __name__ == "__main__":
    unittest.main()
