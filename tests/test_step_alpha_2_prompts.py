from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tech_blog_mlflow.prompt_registry import (
    PromptSpec,
    ensure_run_link,
    find_matching_version,
    registered_variables,
    run_has_prompt_link,
    text_sha256,
    validate_source,
    version_tags,
)


def spec(path: Path) -> PromptSpec:
    return PromptSpec(
        role="generator",
        name="generator",
        source_version="v1",
        source_path=path,
        run_id="run-1",
        expected_variables=frozenset({"THEME"}),
        model_config={"model_name": "model"},
    )


class PromptRegistryContractTest(unittest.TestCase):
    def test_variable_contract_accepts_expected_marker(self):
        prompt = "Write {{THEME}} and {{ THEME }}"
        self.assertEqual(registered_variables(prompt), frozenset({"THEME"}))
        validate_source(spec(Path("prompt.md")), prompt)

    def test_variable_contract_rejects_unexpected_marker(self):
        with self.assertRaisesRegex(ValueError, "Prompt変数"):
            validate_source(spec(Path("prompt.md")), "{{ARTICLE}}")

    def test_tags_preserve_source_identity(self):
        item = spec(Path("prompts/example.md"))
        tags = version_tags(item, "{{THEME}}")
        self.assertEqual(tags["source_version"], "v1")
        self.assertEqual(tags["source_sha256"], text_sha256("{{THEME}}"))

    def test_matching_version_is_reused(self):
        template = "{{THEME}}"
        version = SimpleNamespace(
            template=template,
            tags={"source_sha256": text_sha256(template)},
        )
        client = SimpleNamespace(
            get_prompt=lambda _name: SimpleNamespace(name="generator"),
            search_prompt_versions=lambda *_args, **_kwargs: [version]
        )
        self.assertIs(find_matching_version(client, spec(Path("x")), template), version)

    def test_duplicate_matching_versions_are_rejected(self):
        template = "{{THEME}}"
        version = SimpleNamespace(
            template=template,
            tags={"source_sha256": text_sha256(template)},
        )
        client = SimpleNamespace(
            get_prompt=lambda _name: SimpleNamespace(name="generator"),
            search_prompt_versions=lambda *_args, **_kwargs: [version, version]
        )
        with self.assertRaisesRegex(ValueError, "複数"):
            find_matching_version(client, spec(Path("x")), template)

    def test_missing_prompt_is_a_new_registration(self):
        client = SimpleNamespace(get_prompt=lambda _name: None)
        self.assertIsNone(
            find_matching_version(client, spec(Path("x")), "{{THEME}}")
        )

    def test_run_link_is_idempotent(self):
        calls = []
        prompt = SimpleNamespace(
            uri="prompts:/generator/1", name="generator", version=1
        )
        client = SimpleNamespace(
            get_run=lambda _run_id: SimpleNamespace(
                data=SimpleNamespace(
                    tags={
                        "mlflow.linkedPrompts":
                            '[{"name":"generator","version":"1"}]'
                    }
                )
            ),
            link_prompt_version_to_run=lambda *args: calls.append(args),
        )
        self.assertFalse(ensure_run_link(client, "run-1", prompt))
        self.assertEqual(calls, [])

    def test_run_link_rejects_invalid_json(self):
        prompt = SimpleNamespace(name="generator", version=1)
        client = SimpleNamespace(
            get_run=lambda _run_id: SimpleNamespace(
                data=SimpleNamespace(tags={"mlflow.linkedPrompts": "{"})
            )
        )
        with self.assertRaisesRegex(ValueError, "JSON"):
            run_has_prompt_link(client, "run-1", prompt)


if __name__ == "__main__":
    unittest.main()
