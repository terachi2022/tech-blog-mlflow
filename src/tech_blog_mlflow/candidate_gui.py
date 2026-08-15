"""Candidate pipelineの結果をMLflow GUIで扱えるEntityへ公開する。"""

from __future__ import annotations

from dataclasses import asdict
import re
from typing import Any

from tech_blog_mlflow.candidate_models import GENERATOR, INDEPENDENT_JUDGE, PIPELINE_VERSION, PRIMARY_JUDGE, REVIEWER
from tech_blog_mlflow.review_workflow import (
    ensure_label_schemas, ensure_review_presentation, ensure_review_queue,
    extract_article_payload, resolve_single_trace, review_item_dict, schema_contract,
)

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "tech-blog-generation"
REVIEW_QUEUE_NAME = "candidate-multi-model-human-review-v1"
ROLE_RUNS = {
    "generator": ("candidate-generation", None, GENERATOR),
    "reviewer": ("candidate-review", None, REVIEWER),
    "primary-judge": ("evaluation-calibration", "primary", PRIMARY_JUDGE),
    "independent-judge": ("evaluation-calibration", "independent", INDEPENDENT_JUDGE),
}


def latest_candidate_run_ids(client: Any, experiment_id: str) -> dict[str, str]:
    """役割ごとの最新Candidate Runを取得する。"""
    result: dict[str, str] = {}
    for role, (stage, judge_role, _model) in ROLE_RUNS.items():
        clauses = [f"tags.stage = '{stage}'"]
        if role in {"generator", "reviewer"}:
            clauses.append(f"params.pipeline_version = '{PIPELINE_VERSION}'")
        else:
            clauses.append("params.generator_model LIKE '%GPT-OSS-Swallow-120B%'")
        if judge_role:
            clauses.append(f"params.judge_role = '{judge_role}'")
        runs = client.search_runs(
            [experiment_id], filter_string=" AND ".join(clauses),
            order_by=["attributes.start_time DESC"], max_results=1,
        )
        if not runs:
            raise ValueError(f"Candidate Runがありません: role={role}")
        result[role] = runs[0].info.run_id
    return result


def publish_candidate_models(*, mlflow_module: Any, client: Any, experiment_id: str, run_ids: dict[str, str]) -> list[dict[str, Any]]:
    """各役割をMLflow Logged Model（External Model）として冪等登録する。"""
    published = []
    for role, (_stage, _judge_role, model) in ROLE_RUNS.items():
        run_id = run_ids[role]
        client.get_run(run_id)
        model_slug = re.sub(r"[^A-Za-z0-9_-]+", "-", model.model_id.rsplit("/", 1)[-1])
        name = f"candidate-{role}-{model_slug}"
        matches = list(client.search_logged_models(
            experiment_ids=[experiment_id], filter_string=f"name = '{name}'", max_results=100,
        ))
        existing = next((
            item for item in matches
            if item.source_run_id == run_id
            and item.tags.get("candidate_model_role") == role
            and str(getattr(item.status, "name", item.status)).upper() == "READY"
        ), None)
        created = existing is None
        logged_model = existing or mlflow_module.create_external_model(
            name=name, source_run_id=run_id, experiment_id=experiment_id,
            model_type="llm-judge" if "judge" in role else "text-generation",
            tags={
                "candidate_model_role": role, "huggingface_id": model.model_id,
                "pipeline_version": PIPELINE_VERSION, "weights_location": "huggingface-cache",
                "weights_copied_to_mlflow": "false",
            },
            params={
                "runtime": model.runtime, "quantization": model.quantization,
                "max_tokens": str(model.max_tokens), "temperature": str(model.temperature),
            },
        )
        published.append({
            "role": role, "name": logged_model.name, "model_id": logged_model.model_id,
            "model_uri": logged_model.model_uri, "source_run_id": run_id, "created": created,
        })
    return published


def publish_candidate_review(*, mlflow_module: Any, experiment_id: str, evaluation_run_id: str) -> dict[str, Any]:
    """Primary Judgeの記事TraceをHuman Review Queueへ冪等追加する。"""
    from mlflow.genai.review_queues import add_items_to_review_queue

    target = resolve_single_trace(
        mlflow_module.search_traces, experiment_id=experiment_id,
        evaluation_run_id=evaluation_run_id, label="candidate-multi-model",
    )
    source_trace = mlflow_module.get_trace(target.trace_id, flush=True)
    if source_trace is None:
        raise ValueError(f"評価Traceを取得できません: {target.trace_id}")
    extract_article_payload(source_trace)
    schemas, created_schema_names = ensure_label_schemas(experiment_id=experiment_id)
    queue, queue_created = ensure_review_queue(
        experiment_id=experiment_id, queue_name=REVIEW_QUEUE_NAME,
        schema_ids=[str(schema.schema_id) for schema in schemas],
    )
    presentation = ensure_review_presentation(
        mlflow_module=mlflow_module, experiment_id=experiment_id,
        target=target, source_trace=source_trace,
    )
    items = add_items_to_review_queue(queue.queue_id, item_ids=[presentation.review_trace_id])
    return {
        "queue_id": queue.queue_id, "queue_name": queue.name, "queue_created": queue_created,
        "schemas_created": created_schema_names,
        "schemas": [schema_contract(schema) for schema in schemas],
        "presentation": asdict(presentation), "items": [review_item_dict(item) for item in items],
    }
