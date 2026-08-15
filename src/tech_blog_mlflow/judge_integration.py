"""STEP-α-5: OSS Judges登録とLocal MLX Judgeの境界を検証する。"""

from __future__ import annotations

from typing import Any


TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "tech-blog-generation"
WORKFLOW_VERSION = "step-alpha-5-judges-v1.0.0"
REGISTERED_SCORER_NAME = "article_length_guard_v1"
LOCAL_JUDGE_SCORER_NAME = "local_article_judge_v2_4"


def scorer_contract() -> dict[str, Any]:
    return {
        "name": REGISTERED_SCORER_NAME,
        "builtin_scorer_class": "ResponseLength",
        "min_length": 1800,
        "max_length": 7000,
        "unit": "chars",
    }


def find_scorer(scorers: list[Any], name: str) -> Any | None:
    matches = [scorer for scorer in scorers if scorer.name == name]
    if len(matches) > 1:
        raise ValueError(f"同名Scorerが複数あります: {name}")
    return matches[0] if matches else None


def builtin_contract_matches(scorer: Any) -> bool:
    dumped = scorer.model_dump()
    data = dumped.get("builtin_scorer_pydantic_data") or {}
    expected = scorer_contract()
    return (
        dumped.get("name") == expected["name"]
        and dumped.get("builtin_scorer_class") == expected["builtin_scorer_class"]
        and data.get("min_length") == expected["min_length"]
        and data.get("max_length") == expected["max_length"]
        and data.get("unit") == expected["unit"]
    )


def _status_value(value: Any) -> str:
    return str(getattr(value, "name", getattr(value, "value", value))).upper()


def is_expected_oss_registration_error(message: str) -> bool:
    return (
        "Custom scorer registration" in message
        and "not supported outside of Databricks" in message
    )


def verify_local_judge_registration_boundary(experiment_id: str) -> dict[str, Any]:
    """実ScorerがOSS登録拒否されることをAPI自身で確認する。"""

    from mlflow.exceptions import MlflowException

    from evaluation.llm_scorer_v2_4 import build_llm_judge_v2_4_scorer
    from evaluation.local_judge_v2_4 import LocalArticleJudgeV2_4

    local_scorer = build_llm_judge_v2_4_scorer(LocalArticleJudgeV2_4())
    try:
        local_scorer.register(
            name=LOCAL_JUDGE_SCORER_NAME,
            experiment_id=experiment_id,
        )
    except MlflowException as exc:
        message = str(exc)
        if not is_expected_oss_registration_error(message):
            raise
        return {
            "scorer_name": LOCAL_JUDGE_SCORER_NAME,
            "scorer_kind": str(local_scorer.kind),
            "registration_supported": False,
            "expected_rejection_observed": True,
            "reason": "decorator_scorer_registration_not_supported_on_oss",
        }
    raise ValueError("Local MLX decorator Scorerが想定外にOSSへ登録されました。")


def setup_judge_integration(
    *,
    tracking_uri: str = TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
    dry_run: bool = False,
) -> dict[str, Any]:
    import mlflow
    from mlflow.genai.scorers import ResponseLength, list_scorers

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name=experiment_name)
    boundary = verify_local_judge_registration_boundary(experiment.experiment_id)
    base = {
        "workflow_version": WORKFLOW_VERSION,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment.name,
        "experiment_id": experiment.experiment_id,
        "local_judge_boundary": boundary,
        "registered_scorer_plan": scorer_contract(),
        "local_judge_evidence": {
            "external_model": "models:/m-57625c5d614f4b9382aa9a243abb340c",
            "prompt": "prompts:/tech-blog-article-judge/1",
            "evaluation_run_id": "fdf0c239445f44a0999a6b1fe7a419b6",
            "evaluation_dataset_id": "d-f21d22043d7749a387cf34bc06fcffd5",
            "review_queue": "article-quality-human-review-v2",
        },
    }
    if dry_run:
        return {**base, "mode": "dry-run"}

    scorer = find_scorer(
        list_scorers(experiment_id=experiment.experiment_id),
        REGISTERED_SCORER_NAME,
    )
    created = scorer is None
    if created:
        scorer = ResponseLength(
            name=REGISTERED_SCORER_NAME,
            min_length=1800,
            max_length=7000,
            unit="chars",
        ).register(experiment_id=experiment.experiment_id)
    elif not builtin_contract_matches(scorer):
        raise ValueError("既存ScorerのContractが異なります。上書きしません。")
    return {
        **base,
        "mode": "apply",
        "registered_scorer": {
            "name": scorer.name,
            "kind": str(scorer.kind),
            "status": _status_value(scorer.status),
            "sample_rate": scorer.sample_rate,
            "filter_string": scorer.filter_string,
            "created": created,
            "contract_matches": builtin_contract_matches(scorer),
        },
    }


def validate_judge_integration(
    *,
    tracking_uri: str = TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
) -> dict[str, Any]:
    import mlflow
    from mlflow import MlflowClient
    from mlflow.genai.datasets import get_dataset
    from mlflow.genai.scorers import list_scorers

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name=experiment_name)
    boundary = verify_local_judge_registration_boundary(experiment.experiment_id)
    scorers = list_scorers(experiment_id=experiment.experiment_id)
    scorer = find_scorer(scorers, REGISTERED_SCORER_NAME)
    client = MlflowClient()
    evidence_checks = {
        "judge_model_retrievable": client.get_logged_model(
            "m-57625c5d614f4b9382aa9a243abb340c"
        ).model_id == "m-57625c5d614f4b9382aa9a243abb340c",
        "judge_prompt_retrievable": client.get_prompt_version(
            "tech-blog-article-judge", 1
        ).version == 1,
        "evaluation_run_retrievable": client.get_run(
            "fdf0c239445f44a0999a6b1fe7a419b6"
        ).info.run_id == "fdf0c239445f44a0999a6b1fe7a419b6",
        "calibration_dataset_retrievable": get_dataset(
            dataset_id="d-f21d22043d7749a387cf34bc06fcffd5"
        ).dataset_id == "d-f21d22043d7749a387cf34bc06fcffd5",
    }
    checks = {
        "builtin_scorer_registered": scorer is not None,
        "builtin_contract_matches": scorer is not None and builtin_contract_matches(scorer),
        "builtin_not_sampling": scorer is not None and scorer.sample_rate is None,
        "local_registration_rejected_as_expected": boundary[
            "expected_rejection_observed"
        ],
        "local_judge_not_misrepresented": find_scorer(
            scorers, LOCAL_JUDGE_SCORER_NAME
        ) is None,
        **evidence_checks,
    }
    return {
        "workflow_version": WORKFLOW_VERSION,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment.name,
        "experiment_id": experiment.experiment_id,
        "validation_status": "validated" if all(checks.values()) else "invalid",
        "checks": checks,
        "registered_scorers": [
            {
                "name": item.name,
                "kind": str(item.kind),
                "status": _status_value(item.status),
                "sample_rate": item.sample_rate,
            }
            for item in scorers
        ],
        "local_judge_boundary": boundary,
    }
