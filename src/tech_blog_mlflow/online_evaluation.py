"""STEP 5-DのOffline/Online Joinを検証して構築する。

Publication Registryを唯一のIdentity Layerとして、採用Generation、
Offline Evaluation、GA4 Observation、GSC Observationを結合する。
Online指標からOffline品質点を上書きしたり、合成Scoreを自動生成したりしない。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from tech_blog_mlflow.online_registry import (
    load_registry,
    require_run_id,
    require_sha256,
    sha256_file,
    validate_offline_reference,
)


JOIN_SCHEMA_VERSION = "offline-online-join-v1"
ONLINE_EVALUATION_VERSION = "online-evaluation-v1"
OBSERVATION_SCHEMA_VERSION = "online-observation-v1"
SUPPORTED_COLLECTORS = {
    "ga4": "ga4-data-api-v1.2",
    "gsc": "gsc-search-analytics-v1.1",
}

IDENTIFIER_PATTERN = re.compile(r"^[0-9a-f]{24}$")


def canonical_json_sha256(value: Any) -> str:
    """JSON値を決定的に直列化してSHA-256を返す。"""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONがありません: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON RootはObjectである必要があります: {path}")
    return value


def load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONLがありません: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL {line_number}行目がObjectではありません: {path}")
        records.append(value)
    return records


def select_unique_record(
    records: list[dict[str, Any]],
    *,
    key: str,
    value: str,
    label: str,
) -> dict[str, Any]:
    matches = [record for record in records if record.get(key) == value]
    if not matches:
        raise KeyError(f"{label}が見つかりません: {value}")
    if len(matches) != 1:
        raise ValueError(f"{label}が重複しています: {value}")
    return matches[0]


def resolve_run_json(directory: Path, run_id: str, label: str) -> tuple[Path, dict[str, Any]]:
    """Run IDを持つJSONをDirectory配下から一意に解決する。"""
    required_run_id = require_run_id(run_id, label)
    if not directory.is_dir():
        raise FileNotFoundError(f"{label}のDirectoryがありません: {directory}")
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(directory.glob("*.json")):
        value = load_json_object(path)
        if value.get("run_id") == required_run_id:
            matches.append((path, value))
    if not matches:
        raise FileNotFoundError(
            f"{label}={required_run_id}を含むJSONがありません: {directory}"
        )
    if len(matches) != 1:
        paths = [str(path) for path, _ in matches]
        raise ValueError(f"{label}のJSONが重複しています: {paths}")
    return matches[0]


def _project_relative(project_root: Path, path: Path) -> str:
    root = project_root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"Source ArtifactがProject外です: {path}") from error


def validate_observation(
    observation: Mapping[str, Any],
    *,
    publication: Mapping[str, Any],
    source: str,
) -> None:
    if source not in SUPPORTED_COLLECTORS:
        raise ValueError(f"未対応のOnline Sourceです: {source}")
    if observation.get("schema_version") != OBSERVATION_SCHEMA_VERSION:
        raise ValueError(
            f"{source} Observation Schemaが違います: "
            f"{observation.get('schema_version')!r}"
        )
    observation_id = str(observation.get("observation_id") or "")
    if not IDENTIFIER_PATTERN.fullmatch(observation_id):
        raise ValueError(f"{source} Observation IDが不正です。")
    if observation.get("publication_id") != publication.get("publication_id"):
        raise ValueError(f"{source} Observationのpublication_idが一致しません。")
    if observation.get("source") != source:
        raise ValueError(f"Observation Sourceが一致しません: expected={source}")
    expected_collector = SUPPORTED_COLLECTORS[source]
    if observation.get("collector_version") != expected_collector:
        raise ValueError(
            f"{source} Collector Versionが採用Versionではありません: "
            f"actual={observation.get('collector_version')!r}, "
            f"expected={expected_collector!r}"
        )
    date_range = observation.get("date_range")
    if not isinstance(date_range, Mapping):
        raise KeyError(f"{source} Observationにdate_rangeがありません。")
    if not date_range.get("start_date") or not date_range.get("end_date"):
        raise KeyError(f"{source} Observationの期間が不完全です。")
    metrics = observation.get("metrics")
    if not isinstance(metrics, Mapping):
        raise KeyError(f"{source} Observationにmetricsがありません。")
    require_sha256(observation.get("raw_response_sha256"), f"{source} Raw SHA-256")

    dimensions = observation.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise KeyError(f"{source} Observationにdimensionsがありません。")
    online = publication["online"]
    if source == "ga4":
        if dimensions.get("page_location") != online["published_url"]:
            raise ValueError("GA4 Observationのpage_locationがCanonical URLと違います。")
        if str(dimensions.get("ga4_property_id")) != str(online["ga4_property_id"]):
            raise ValueError("GA4 ObservationのProperty IDがRegistryと違います。")
        window_state = dimensions.get("collection_window_state")
        if window_state not in {"includes_today", "closed_period"}:
            raise ValueError("GA4 Observationのcollection_window_stateが不正です。")
        if window_state == "includes_today" and observation.get("is_partial") is not True:
            raise ValueError(
                "GA4 includes_today Observationはpartialである必要があります。"
            )
        expected_cta = online["cta_tracking"].get("enabled") is True
        if metrics.get("cta_tracking_enabled") is not expected_cta:
            raise ValueError("GA4 ObservationのCTA計測状態がRegistryと違います。")
        if not expected_cta and any(
            metrics.get(name) is not None
            for name in ("cta_event_count", "cta_rate", "cta_rate_denominator")
        ):
            raise ValueError("CTA未実装時のCTA指標はnullである必要があります。")
    else:
        if dimensions.get("page") != online["published_url"]:
            raise ValueError("GSC ObservationのpageがCanonical URLと違います。")
        if dimensions.get("gsc_site_url") != online["gsc_site_url"]:
            raise ValueError("GSC ObservationのPropertyがRegistryと違います。")
        if dimensions.get("data_state") not in {"all", "final"}:
            raise ValueError("GSC Observationのdata_stateが不正です。")
        if metrics.get("impressions") == 0:
            if metrics.get("clicks") != 0:
                raise ValueError("GSCはimpressions=0なのにclicks>0です。")
            if metrics.get("ctr") is not None or metrics.get("position") is not None:
                raise ValueError(
                    "GSC impressions=0ではctrとpositionをnullにしてください。"
                )


def resolve_raw_artifact(
    raw_dir: Path,
    observation: Mapping[str, Any],
) -> Path:
    source = str(observation["source"])
    date_range = observation["date_range"]
    raw_sha256 = str(observation["raw_response_sha256"])
    filename = (
        f"{source}_{observation['publication_id']}_"
        f"{date_range['start_date']}_{date_range['end_date']}_"
        f"{raw_sha256[:12]}.json"
    )
    path = raw_dir / filename
    raw = load_json_object(path)
    if raw.get("raw_response_sha256") != raw_sha256:
        raise ValueError(f"Raw Artifact内部のSHA-256がObservationと違います: {path}")
    return path


def _source_descriptor(project_root: Path, path: Path) -> dict[str, str]:
    return {
        "path": _project_relative(project_root, path),
        "file_sha256": sha256_file(path),
    }


def _nullable_metric_paths(observation: Mapping[str, Any]) -> list[str]:
    source = str(observation["source"])
    return sorted(
        f"online/{source}/{name}"
        for name, value in observation["metrics"].items()
        if value is None
    )


def build_join_record(
    *,
    project_root: Path,
    publication: dict[str, Any],
    generation: dict[str, Any],
    generation_path: Path,
    evaluation: dict[str, Any],
    evaluation_path: Path,
    ga4: dict[str, Any],
    ga4_raw_path: Path,
    gsc: dict[str, Any],
    gsc_raw_path: Path,
    created_at: str,
) -> dict[str, Any]:
    """検証済みSourceから、Scoreを混合しないJoin Recordを構築する。"""
    expected_offline = validate_offline_reference(project_root, generation, evaluation)
    if publication.get("offline") != expected_offline:
        raise ValueError("Publication RegistryのOffline参照をSource JSONから再現できません。")
    validate_observation(ga4, publication=publication, source="ga4")
    validate_observation(gsc, publication=publication, source="gsc")

    ga4_range = dict(ga4["date_range"])
    gsc_range = dict(gsc["date_range"])
    if ga4_range != gsc_range:
        raise ValueError(
            "GA4とGSCのDate Label範囲が一致しません: "
            f"ga4={ga4_range}, gsc={gsc_range}"
        )

    gsc_data_state = str(gsc["dimensions"]["data_state"])
    partial_reasons: list[str] = []
    if ga4.get("is_partial") is True:
        partial_reasons.append("GA4 Observationがpartialです。")
    if gsc.get("is_partial") is True:
        partial_reasons.append("GSC Observationがpartialです。")
    if gsc_data_state != "final":
        partial_reasons.append("GSC dataStateがfinalではありません。")
    data_status = "provisional" if partial_reasons else "final"

    identity = {
        "schema_version": JOIN_SCHEMA_VERSION,
        "publication_id": publication["publication_id"],
        "generation_run_id": publication["offline"]["generation_run_id"],
        "evaluation_run_id": publication["offline"]["evaluation_run_id"],
        "ga4_observation_id": ga4["observation_id"],
        "gsc_observation_id": gsc["observation_id"],
    }
    join_id = canonical_json_sha256(identity)[:24]
    nullable_metrics = sorted(
        _nullable_metric_paths(ga4) + _nullable_metric_paths(gsc)
    )
    return {
        "schema_version": JOIN_SCHEMA_VERSION,
        "online_evaluation_version": ONLINE_EVALUATION_VERSION,
        "join_id": join_id,
        "created_at": created_at,
        "online_evaluation_run_id": None,
        "data_status": data_status,
        "partial_reasons": partial_reasons,
        "publication": publication,
        "offline": {
            "article": {
                "path": publication["offline"]["article_path"],
                "sha256": publication["offline"]["article_sha256"],
            },
            "generation": {
                **_source_descriptor(project_root, generation_path),
                "run_id": generation["run_id"],
                "model": generation["model"],
                "prompt_version": generation["prompt_version"],
                "generation_config_version": generation[
                    "generation_config_version"
                ],
                "metrics": generation["metrics"],
            },
            "evaluation": {
                **_source_descriptor(project_root, evaluation_path),
                "run_id": evaluation["run_id"],
                "combined_version": evaluation["combined_version"],
                "judge_model": evaluation["judge"]["model"],
                "judge_prompt_version": evaluation["judge"]["prompt_version"],
                "metrics": evaluation["metrics"],
            },
        },
        "online": {
            "date_alignment": {
                "same_date_labels": True,
                "start_date": ga4_range["start_date"],
                "end_date": ga4_range["end_date"],
                "ga4_timezone": publication["online"]["measurement_timezone"],
                "gsc_timezone": str(gsc["dimensions"]["date_timezone"]),
                "same_instant_window": False,
            },
            "ga4": {
                "source_artifact": _source_descriptor(project_root, ga4_raw_path),
                "observation": ga4,
            },
            "gsc": {
                "source_artifact": _source_descriptor(project_root, gsc_raw_path),
                "observation": gsc,
            },
            "nullable_metrics": nullable_metrics,
        },
        "decision_policy": {
            "automatic_promotion": False,
            "online_thresholds_applied": False,
            "human_review_required": True,
            "offline_scores_overwritten": False,
            "composite_offline_online_score_created": False,
            "cta_tracking_enabled": publication["online"]["cta_tracking"][
                "enabled"
            ],
        },
    }


def metric_map(record: Mapping[str, Any]) -> dict[str, float]:
    """MLflowへ保存できる非null数値だけをNamespace付きで返す。"""
    metrics: dict[str, float] = {}

    def add(prefix: str, values: Mapping[str, Any]) -> None:
        for name, value in values.items():
            if isinstance(value, bool) or value is None:
                continue
            if not isinstance(value, (int, float)):
                continue
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"Metricが有限値ではありません: {prefix}/{name}")
            metrics[f"{prefix}/{name}"] = number

    add("offline/generation", record["offline"]["generation"]["metrics"])
    add("offline/evaluation", record["offline"]["evaluation"]["metrics"])
    add("online/ga4", record["online"]["ga4"]["observation"]["metrics"])
    add("online/gsc", record["online"]["gsc"]["observation"]["metrics"])
    return metrics


def parameter_map(record: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
    publication = record["publication"]
    ga4 = record["online"]["ga4"]["observation"]
    gsc = record["online"]["gsc"]["observation"]
    return {
        "evaluation_stage": "step-5-d",
        "evaluation_type": "offline-online-join",
        "join_schema_version": str(record["schema_version"]),
        "online_evaluation_version": str(record["online_evaluation_version"]),
        "join_id": str(record["join_id"]),
        "publication_id": str(publication["publication_id"]),
        "article_sha256": str(publication["offline"]["article_sha256"]),
        "generation_run_id": str(publication["offline"]["generation_run_id"]),
        "offline_evaluation_run_id": str(
            publication["offline"]["evaluation_run_id"]
        ),
        "ga4_observation_id": str(ga4["observation_id"]),
        "ga4_collector_version": str(ga4["collector_version"]),
        "ga4_collection_window_state": str(
            ga4["dimensions"]["collection_window_state"]
        ),
        "gsc_observation_id": str(gsc["observation_id"]),
        "gsc_collector_version": str(gsc["collector_version"]),
        "gsc_data_state": str(gsc["dimensions"]["data_state"]),
        "start_date": str(ga4["date_range"]["start_date"]),
        "end_date": str(ga4["date_range"]["end_date"]),
        "data_status": str(record["data_status"]),
        "ga4_is_partial": bool(ga4.get("is_partial", False)),
        "gsc_is_partial": bool(gsc.get("is_partial", False)),
        "cta_tracking_enabled": bool(
            publication["online"]["cta_tracking"]["enabled"]
        ),
    }


def prepare_join_from_project(
    *,
    project_root: Path,
    registry_path: Path,
    observations_path: Path,
    publication_id: str,
    ga4_observation_id: str,
    gsc_observation_id: str,
    generation_results_dir: Path,
    evaluation_results_dir: Path,
    ga4_raw_dir: Path,
    gsc_raw_dir: Path,
    created_at: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    publication = select_unique_record(
        load_registry(registry_path),
        key="publication_id",
        value=publication_id,
        label="Publication",
    )
    observations = load_jsonl_objects(observations_path)
    ga4 = select_unique_record(
        observations,
        key="observation_id",
        value=ga4_observation_id,
        label="GA4 Observation",
    )
    gsc = select_unique_record(
        observations,
        key="observation_id",
        value=gsc_observation_id,
        label="GSC Observation",
    )
    offline = publication["offline"]
    generation_path, generation = resolve_run_json(
        generation_results_dir,
        str(offline["generation_run_id"]),
        "Generation Run ID",
    )
    evaluation_path, evaluation = resolve_run_json(
        evaluation_results_dir,
        str(offline["evaluation_run_id"]),
        "Evaluation Run ID",
    )
    validate_observation(ga4, publication=publication, source="ga4")
    validate_observation(gsc, publication=publication, source="gsc")
    if dict(ga4["date_range"]) != dict(gsc["date_range"]):
        raise ValueError(
            "GA4とGSCのDate Label範囲が一致しません: "
            f"ga4={ga4['date_range']}, gsc={gsc['date_range']}"
        )
    ga4_raw_path = resolve_raw_artifact(ga4_raw_dir, ga4)
    gsc_raw_path = resolve_raw_artifact(gsc_raw_dir, gsc)
    record = build_join_record(
        project_root=project_root,
        publication=publication,
        generation=generation,
        generation_path=generation_path,
        evaluation=evaluation,
        evaluation_path=evaluation_path,
        ga4=ga4,
        ga4_raw_path=ga4_raw_path,
        gsc=gsc,
        gsc_raw_path=gsc_raw_path,
        created_at=created_at,
    )
    return record, {
        "generation": generation_path,
        "evaluation": evaluation_path,
        "ga4_raw": ga4_raw_path,
        "gsc_raw": gsc_raw_path,
    }


def write_join_record(path: Path, record: Mapping[str, Any]) -> None:
    """Join RecordをAtomicに書き込む。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(serialized)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
