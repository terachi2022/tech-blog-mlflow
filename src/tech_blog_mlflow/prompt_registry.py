"""STEP-α-2: MLflow Prompt Registryの登録と検証。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "tech-blog-generation"
WORKFLOW_VERSION = "step-alpha-2-prompts-v1.0.0"
PROMPT_ALIAS = "production"

GENERATOR_RUN_ID = "b5c925c2322b4e30b04f07e24d160a04"
EVALUATION_RUN_ID = "fdf0c239445f44a0999a6b1fe7a419b6"


@dataclass(frozen=True)
class PromptSpec:
    role: str
    name: str
    source_version: str
    source_path: Path
    run_id: str
    expected_variables: frozenset[str]
    model_config: dict[str, Any]
    response_format: dict[str, Any] | None = None


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_specs(project_root: Path = Path(".")) -> tuple[PromptSpec, ...]:
    from evaluation.judge_schema_v2 import ArticleJudgeResultV2

    return (
        PromptSpec(
            role="generator",
            name="tech-blog-article-generator",
            source_version="article-v3.5.2",
            source_path=project_root / "prompts/article_generation_v3_5_2.md",
            run_id=GENERATOR_RUN_ID,
            expected_variables=frozenset({
                "THEME",
                "MACOS_VERSION",
                "PYTHON_VERSION",
                "MLFLOW_VERSION",
                "MLX_LM_VERSION",
            }),
            model_config={
                "model_name": "Qwen/Qwen3-8B-MLX-4bit",
                "temperature": 0.0,
                "max_tokens": 4096,
            },
        ),
        PromptSpec(
            role="judge",
            name="tech-blog-article-judge",
            source_version="article-judge-v2.4",
            source_path=project_root / "prompts/article_judge_v2_4.md",
            run_id=EVALUATION_RUN_ID,
            expected_variables=frozenset({"ARTICLE"}),
            model_config={
                "model_name": "mlx-community/gemma-3-text-27b-it-4bit",
                "temperature": 0.0,
                "max_tokens": 3600,
            },
            response_format=ArticleJudgeResultV2.model_json_schema(),
        ),
    )


def read_template(spec: PromptSpec) -> str:
    if not spec.source_path.is_file():
        raise FileNotFoundError(f"Promptがありません: {spec.source_path}")
    return spec.source_path.read_text(encoding="utf-8")


def registered_variables(template: str) -> frozenset[str]:
    import re

    return frozenset(re.findall(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}", template))


def validate_source(spec: PromptSpec, template: str) -> None:
    actual = registered_variables(template)
    if actual != spec.expected_variables:
        raise ValueError(
            "Prompt変数がContractと一致しません: "
            f"name={spec.name}, expected={sorted(spec.expected_variables)}, "
            f"actual={sorted(actual)}"
        )


def version_tags(spec: PromptSpec, template: str) -> dict[str, str]:
    return {
        "workflow_version": WORKFLOW_VERSION,
        "prompt_role": spec.role,
        "source_version": spec.source_version,
        "source_path": str(spec.source_path),
        "source_sha256": text_sha256(template),
    }


def find_matching_version(
    client: Any,
    spec: PromptSpec,
    template: str,
) -> Any | None:
    # MLflow OSSのsearch_prompt_versions()は、初回登録前には空Listでは
    # なくRESOURCE_DOES_NOT_EXISTを返す。先にPromptの存在を確認する。
    if client.get_prompt(spec.name) is None:
        return None
    expected_hash = text_sha256(template)
    versions = list(client.search_prompt_versions(spec.name, max_results=100))
    matches = [
        version
        for version in versions
        if version.tags.get("source_sha256") == expected_hash
        and version.template == template
    ]
    if len(matches) > 1:
        raise ValueError(
            f"同一内容のPrompt Versionが複数あります: name={spec.name}"
        )
    return matches[0] if matches else None


def run_has_prompt_link(client: Any, run_id: str, prompt: Any) -> bool:
    """Run側のMLflow標準TagからPrompt Linkを確認する。

    MLflow 3.15.1 OSSではlink_prompt_version_to_run()がこのTagを更新する
    一方、list_logged_prompts()は別Tagを検索するため、後者は検証に使わない。
    """

    raw = client.get_run(run_id).data.tags.get("mlflow.linkedPrompts", "[]")
    try:
        links = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Runのmlflow.linkedPromptsがJSONではありません: run_id={run_id}"
        ) from exc
    return any(
        item.get("name") == prompt.name
        and str(item.get("version")) == str(prompt.version)
        for item in links
    )


def ensure_run_link(client: Any, run_id: str, prompt: Any) -> bool:
    if run_has_prompt_link(client, run_id, prompt):
        return False
    client.link_prompt_version_to_run(run_id, prompt)
    return True


def setup_prompt_registry(
    *,
    tracking_uri: str = TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
    project_root: Path = Path("."),
    dry_run: bool = False,
) -> dict[str, Any]:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name=experiment_name)
    client = MlflowClient()
    specs = prompt_specs(project_root)
    plans = []
    for spec in specs:
        template = read_template(spec)
        validate_source(spec, template)
        plans.append({
            **asdict(spec),
            "source_path": str(spec.source_path),
            "expected_variables": sorted(spec.expected_variables),
            "source_sha256": text_sha256(template),
            "template_chars": len(template),
        })

    base = {
        "workflow_version": WORKFLOW_VERSION,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment.name,
        "experiment_id": experiment.experiment_id,
        "alias": PROMPT_ALIAS,
        "plans": plans,
    }
    if dry_run:
        return {**base, "mode": "dry-run"}

    registrations = []
    for spec in specs:
        template = read_template(spec)
        prompt = find_matching_version(client, spec, template)
        created = prompt is None
        if created:
            prompt = client.register_prompt(
                name=spec.name,
                template=template,
                commit_message=f"Register {spec.source_version} via {WORKFLOW_VERSION}",
                tags=version_tags(spec, template),
                response_format=spec.response_format,
                model_config=spec.model_config,
            )
        client.set_prompt_alias(spec.name, PROMPT_ALIAS, prompt.version)
        linked = ensure_run_link(client, spec.run_id, prompt)
        registrations.append({
            "role": spec.role,
            "name": prompt.name,
            "version": prompt.version,
            "uri": prompt.uri,
            "alias": PROMPT_ALIAS,
            "source_version": spec.source_version,
            "source_sha256": text_sha256(template),
            "run_id": spec.run_id,
            "created": created,
            "run_link_created": linked,
        })
    return {**base, "mode": "apply", "registrations": registrations}


def validate_prompt_registry(
    *,
    tracking_uri: str = TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
    project_root: Path = Path("."),
) -> dict[str, Any]:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name=experiment_name)
    client = MlflowClient()
    rows = []
    for spec in prompt_specs(project_root):
        template = read_template(spec)
        validate_source(spec, template)
        prompt = client.get_prompt_version_by_alias(spec.name, PROMPT_ALIAS)
        row = {
            "role": spec.role,
            "name": spec.name,
            "version": prompt.version,
            "uri": prompt.uri,
            "alias": PROMPT_ALIAS,
            "source_version": spec.source_version,
            "source_sha256": text_sha256(template),
            "template_matches": prompt.template == template,
            "variables_match": frozenset(prompt.variables) == spec.expected_variables,
            "model_config_matches": prompt.model_config == spec.model_config,
            "response_format_matches": prompt.response_format == spec.response_format,
            "run_id": spec.run_id,
            "run_linked": run_has_prompt_link(client, spec.run_id, prompt),
        }
        row["valid"] = all([
            row["template_matches"],
            row["variables_match"],
            row["model_config_matches"],
            row["response_format_matches"],
            row["run_linked"],
        ])
        rows.append(row)
    return {
        "workflow_version": WORKFLOW_VERSION,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment.name,
        "experiment_id": experiment.experiment_id,
        "validation_status": "validated" if all(row["valid"] for row in rows) else "invalid",
        "prompts": rows,
    }
