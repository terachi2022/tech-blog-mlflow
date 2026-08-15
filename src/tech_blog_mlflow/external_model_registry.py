"""STEP-α-3: MLXモデルをMLflow External Modelとして管理する。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "tech-blog-generation"
WORKFLOW_VERSION = "step-alpha-3-models-v1.0.0"


@dataclass(frozen=True)
class ExternalModelSpec:
    role: str
    name: str
    huggingface_id: str
    source_run_id: str
    prompt_name: str
    prompt_version: int
    model_type: str
    params: dict[str, str]


def model_specs() -> tuple[ExternalModelSpec, ...]:
    return (
        ExternalModelSpec(
            role="generator",
            name="tech-blog-generator-qwen3-8b-mlx-4bit",
            huggingface_id="Qwen/Qwen3-8B-MLX-4bit",
            source_run_id="b5c925c2322b4e30b04f07e24d160a04",
            prompt_name="tech-blog-article-generator",
            prompt_version=1,
            model_type="text-generation",
            params={
                "runtime": "mlx-lm",
                "quantization": "4bit",
                "max_tokens": "4096",
                "temperature": "0.0",
                "seed": "42",
                "enable_thinking": "false",
                "generation_config_version": "generation-v3.5.2",
            },
        ),
        ExternalModelSpec(
            role="judge",
            name="tech-blog-judge-gemma3-27b-mlx-4bit",
            huggingface_id="mlx-community/gemma-3-text-27b-it-4bit",
            source_run_id="fdf0c239445f44a0999a6b1fe7a419b6",
            prompt_name="tech-blog-article-judge",
            prompt_version=1,
            model_type="llm-judge",
            params={
                "runtime": "mlx-lm",
                "quantization": "4bit",
                "max_tokens": "3600",
                "temperature": "0.0",
                "judge_prompt_version": "article-judge-v2.4",
                "judge_schema": "25-subscore-v2",
            },
        ),
    )


def spec_identity(spec: ExternalModelSpec) -> str:
    payload = json.dumps(asdict(spec), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_tags(spec: ExternalModelSpec) -> dict[str, str]:
    return {
        "workflow_version": WORKFLOW_VERSION,
        "model_role": spec.role,
        "huggingface_id": spec.huggingface_id,
        "weights_location": "huggingface-cache",
        "weights_copied_to_mlflow": "false",
        "spec_sha256": spec_identity(spec),
    }


def find_matching_model(client: Any, experiment_id: str, spec: ExternalModelSpec) -> Any | None:
    models = list(client.search_logged_models(
        experiment_ids=[experiment_id],
        filter_string=f"name = '{spec.name}'",
        max_results=100,
    ))
    matches = [
        model for model in models
        if model.tags.get("spec_sha256") == spec_identity(spec)
    ]
    if len(matches) > 1:
        raise ValueError(f"同一仕様のExternal Modelが複数あります: {spec.name}")
    if models and not matches:
        raise ValueError(
            "同名External Modelの仕様が異なります。既存Modelを上書きしません: "
            f"{spec.name}"
        )
    return matches[0] if matches else None


def model_has_prompt_link(model: Any, spec: ExternalModelSpec) -> bool:
    raw = model.tags.get("mlflow.linkedPrompts", "[]")
    try:
        links = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Modelのmlflow.linkedPromptsがJSONではありません: {model.model_id}"
        ) from exc
    return any(
        item.get("name") == spec.prompt_name
        and str(item.get("version")) == str(spec.prompt_version)
        for item in links
    )


def setup_external_models(
    *,
    tracking_uri: str = TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
    dry_run: bool = False,
) -> dict[str, Any]:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name=experiment_name)
    specs = model_specs()
    base = {
        "workflow_version": WORKFLOW_VERSION,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment.name,
        "experiment_id": experiment.experiment_id,
        "plans": [
            {**asdict(spec), "spec_sha256": spec_identity(spec)} for spec in specs
        ],
    }
    if dry_run:
        return {**base, "mode": "dry-run"}

    client = MlflowClient()
    registrations = []
    for spec in specs:
        # Source RunとPrompt Versionを先に解決し、孤立Modelを作らない。
        client.get_run(spec.source_run_id)
        client.get_prompt_version(spec.prompt_name, spec.prompt_version)
        model = find_matching_model(client, experiment.experiment_id, spec)
        created = model is None
        if created:
            model = mlflow.create_external_model(
                name=spec.name,
                source_run_id=spec.source_run_id,
                tags=model_tags(spec),
                params=spec.params,
                model_type=spec.model_type,
                experiment_id=experiment.experiment_id,
            )
        prompt_link_created = not model_has_prompt_link(model, spec)
        if prompt_link_created:
            client.link_prompt_version_to_model(
                spec.prompt_name,
                str(spec.prompt_version),
                model.model_id,
            )
            model = client.get_logged_model(model.model_id)
        registrations.append({
            "role": spec.role,
            "name": model.name,
            "model_id": model.model_id,
            "model_uri": model.model_uri,
            "model_type": model.model_type,
            "source_run_id": model.source_run_id,
            "huggingface_id": spec.huggingface_id,
            "prompt_uri": f"prompts:/{spec.prompt_name}/{spec.prompt_version}",
            "created": created,
            "prompt_link_created": prompt_link_created,
        })
    return {**base, "mode": "apply", "registrations": registrations}


def _status_value(status: Any) -> str:
    return str(getattr(status, "name", getattr(status, "value", status))).upper()


def validate_external_models(
    *,
    tracking_uri: str = TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
) -> dict[str, Any]:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name=experiment_name)
    client = MlflowClient()
    rows = []
    for spec in model_specs():
        model = find_matching_model(client, experiment.experiment_id, spec)
        if model is None:
            rows.append({"role": spec.role, "name": spec.name, "valid": False})
            continue
        artifacts = client.list_logged_model_artifacts(model.model_id)
        artifact_paths = sorted(item.path for item in artifacts)
        row = {
            "role": spec.role,
            "name": model.name,
            "model_id": model.model_id,
            "model_uri": model.model_uri,
            "status": _status_value(model.status),
            "model_type_matches": model.model_type == spec.model_type,
            "source_run_matches": model.source_run_id == spec.source_run_id,
            "tags_match": all(model.tags.get(k) == v for k, v in model_tags(spec).items()),
            "params_match": model.params == spec.params,
            "prompt_linked": model_has_prompt_link(model, spec),
            "artifact_paths": artifact_paths,
            "metadata_artifact_only": artifact_paths == ["MLmodel"],
        }
        row["valid"] = all([
            row["status"] == "READY",
            row["model_type_matches"],
            row["source_run_matches"],
            row["tags_match"],
            row["params_match"],
            row["prompt_linked"],
            row["metadata_artifact_only"],
        ])
        rows.append(row)
    return {
        "workflow_version": WORKFLOW_VERSION,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment.name,
        "experiment_id": experiment.experiment_id,
        "validation_status": "validated" if all(row["valid"] for row in rows) else "invalid",
        "models": rows,
    }
