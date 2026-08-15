"""STEP 5-BのGA4 Data API収集とOnline Observation生成。

Publication Registryは公開対象の不変なIdentity、Online Observationは
期間付きの観測値として分離する。認証情報は入力・出力・Logへ保存しない。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import unquote, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from tech_blog_mlflow.online_registry import load_registry


COLLECTOR_VERSION = "ga4-data-api-v1.2"
RAW_SCHEMA_VERSION = "ga4-raw-response-v1.2"
OBSERVATION_SCHEMA_VERSION = "online-observation-v1"
SUPPORTED_REGISTRY_SCHEMA = "online-publication-registry-v1.2"

MAIN_DIMENSIONS = ("pageLocation",)
MAIN_METRICS = (
    "screenPageViews",
    "activeUsers",
    "userEngagementDuration",
)
CTA_METRICS = ("eventCount",)


@dataclass(frozen=True)
class CollectionResult:
    status: str
    observation: dict[str, Any]
    raw_path: Path
    observation_path: Path


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(text)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def select_publication(
    registry_path: Path,
    publication_id: str | None = None,
) -> dict[str, Any]:
    """指定されたRegistryファイルだけから対象Publicationを一意に選ぶ。"""
    records = load_registry(registry_path)
    if not records:
        raise ValueError(f"Publication Registryが空です: {registry_path}")

    for index, record in enumerate(records, 1):
        if record.get("schema_version") != SUPPORTED_REGISTRY_SCHEMA:
            raise ValueError(
                f"Registry {index}行目のSchemaが未対応です: "
                f"{record.get('schema_version')!r}"
            )

    if publication_id is None:
        if len(records) != 1:
            raise ValueError(
                "Registryに複数Recordがあります。--publication-idを指定してください。"
            )
        record = records[0]
    else:
        matches = [
            record
            for record in records
            if record.get("publication_id") == publication_id
        ]
        if len(matches) != 1:
            raise ValueError(
                "publication_idを一意に特定できません: "
                f"{publication_id}, matches={len(matches)}"
            )
        record = matches[0]

    online = record.get("online")
    offline = record.get("offline")
    if not isinstance(online, dict) or not isinstance(offline, dict):
        raise ValueError("Registry Recordにonline/offline Objectがありません。")
    required_online = {
        "published_url",
        "published_at",
        "measurement_timezone",
        "ga4_property_id",
        "cta_tracking",
    }
    missing = sorted(required_online.difference(online))
    if missing:
        raise ValueError(f"Registry onlineに必須Keyがありません: {missing}")
    if not isinstance(online["cta_tracking"], dict):
        raise ValueError("online.cta_trackingはObjectである必要があります。")
    return record


def resolve_date_range(
    publication: Mapping[str, Any],
    start_date: str | None,
    end_date: str | None,
    *,
    today: date | None = None,
) -> tuple[str, str, bool]:
    online = publication["online"]
    timezone = ZoneInfo(str(online["measurement_timezone"]))
    published = datetime.fromisoformat(str(online["published_at"]))
    published_local_date = published.astimezone(timezone).date()
    current_date = today or datetime.now(timezone).date()

    try:
        start = date.fromisoformat(start_date) if start_date else published_local_date
        end = date.fromisoformat(end_date) if end_date else current_date
    except ValueError as error:
        raise ValueError("start-date/end-dateはYYYY-MM-DD形式で指定してください。") from error

    if start < published_local_date:
        raise ValueError(
            "start-dateは公開日以降にしてください: "
            f"published={published_local_date}, start={start}"
        )
    if end < start:
        raise ValueError("end-dateはstart-date以降にしてください。")
    if end > current_date:
        raise ValueError(
            f"end-dateに未来日は指定できません: today={current_date}, end={end}"
        )
    includes_today = end == current_date
    return start.isoformat(), end.isoformat(), includes_today


def _exact_filter(field_name: str, value: str) -> dict[str, Any]:
    return {
        "filter": {
            "field_name": field_name,
            "string_filter": {
                "match_type": "EXACT",
                "value": value,
                "case_sensitive": False,
            },
        }
    }


def page_location_aliases(value: str) -> tuple[str, ...]:
    """Canonical URIとGA4が保持し得るUnicode IRIを同一対象として返す。

    Registryは再現可能なURIとしてPercent Encodingを保持する。一方、GA4の
    pageLocationはBrowserから受けた日本語PathをUnicodeのまま返すことがある。
    HostやSchemeは変更せず、PathだけをUTF-8として復号する。
    """
    parsed = urlsplit(value)
    try:
        decoded_path = unquote(
            parsed.path,
            encoding="utf-8",
            errors="strict",
        )
    except UnicodeDecodeError as error:
        raise ValueError("Canonical URL PathをUTF-8として復号できません。") from error
    unicode_iri = urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            decoded_path,
            parsed.query,
            parsed.fragment,
        )
    )
    return tuple(dict.fromkeys((value, unicode_iri)))


def _page_location_filter(page_location: str) -> dict[str, Any]:
    """Percent EncodingとUnicode表現を完全一致の候補として扱う。"""
    aliases = page_location_aliases(page_location)
    return {
        "filter": {
            "field_name": "pageLocation",
            "in_list_filter": {
                "values": list(aliases),
                "case_sensitive": False,
            },
        }
    }


def build_request_specs(
    publication: Mapping[str, Any],
    start_date: str,
    end_date: str,
) -> dict[str, dict[str, Any] | None]:
    online = publication["online"]
    page_location = str(online["published_url"])
    property_name = f"properties/{online['ga4_property_id']}"
    main = {
        "property": property_name,
        "date_ranges": [{"start_date": start_date, "end_date": end_date}],
        "dimensions": [{"name": name} for name in MAIN_DIMENSIONS],
        "metrics": [{"name": name} for name in MAIN_METRICS],
        "dimension_filter": _page_location_filter(page_location),
        "keep_empty_rows": True,
        "limit": 10000,
    }

    cta_tracking = online["cta_tracking"]
    if cta_tracking.get("enabled") is not True:
        cta = None
    else:
        event_name = cta_tracking.get("event_name")
        if not event_name:
            raise ValueError("CTA Tracking有効時はevent_nameが必要です。")
        cta = {
            "property": property_name,
            "date_ranges": [{"start_date": start_date, "end_date": end_date}],
            "dimensions": [{"name": "pageLocation"}],
            "metrics": [{"name": name} for name in CTA_METRICS],
            "dimension_filter": {
                "and_group": {
                    "expressions": [
                        _page_location_filter(page_location),
                        _exact_filter("eventName", str(event_name)),
                    ]
                }
            },
            "keep_empty_rows": True,
            "limit": 10000,
        }
    return {"main": main, "cta": cta}


def request_spec_for_display(spec: Mapping[str, Any]) -> dict[str, Any]:
    """SDK入力のsnake_caseをREST相当の読みやすいObjectとして返す。"""
    return json.loads(json.dumps(spec, ensure_ascii=False))


def _import_ga4_client() -> tuple[type[Any], type[Any]]:
    try:
        from google.analytics.data_v1beta import BetaAnalyticsDataClient
        from google.analytics.data_v1beta.types import RunReportRequest
    except ImportError as error:
        raise RuntimeError(
            "google-analytics-dataがありません。"
            "`uv sync`または`uv add google-analytics-data==0.23.0`を実行してください。"
        ) from error
    return BetaAnalyticsDataClient, RunReportRequest


def _response_to_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return json.loads(json.dumps(response, ensure_ascii=False))
    response_type = type(response)
    try:
        return json.loads(
            response_type.to_json(
                response,
                use_integers_for_enums=False,
            )
        )
    except (AttributeError, TypeError, json.JSONDecodeError) as error:
        raise TypeError("GA4 ResponseをJSONへ変換できません。") from error


def execute_reports(
    specs: Mapping[str, Mapping[str, Any] | None],
    *,
    client_factory: Callable[[], Any] | None = None,
) -> dict[str, dict[str, Any] | None]:
    BetaAnalyticsDataClient, RunReportRequest = _import_ga4_client()
    factory = client_factory or BetaAnalyticsDataClient
    client = factory()
    responses: dict[str, dict[str, Any] | None] = {}
    try:
        for name in ("main", "cta"):
            spec = specs[name]
            if spec is None:
                responses[name] = None
                continue
            request = RunReportRequest(**dict(spec))
            responses[name] = _response_to_dict(client.run_report(request=request))
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    return responses


def _header_names(response: Mapping[str, Any], key: str) -> list[str]:
    values = response.get(key, [])
    if not isinstance(values, list):
        raise ValueError(f"GA4 Responseの{key}がListではありません。")
    return [str(item.get("name", "")) for item in values]


def _decimal(value: Any, label: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"GA4 Metric {label}が数値ではありません: {value!r}") from error
    if not number.is_finite() or number < 0:
        raise ValueError(f"GA4 Metric {label}は有限の非負値である必要があります。")
    return number


def _aggregate_response(
    response: Mapping[str, Any],
    expected_dimensions: tuple[str, ...],
    expected_metrics: tuple[str, ...],
) -> tuple[dict[str, Decimal], bool]:
    dimensions = _header_names(response, "dimensionHeaders")
    metrics = _header_names(response, "metricHeaders")
    if dimensions != list(expected_dimensions):
        raise ValueError(
            f"GA4 Dimension Headerが不一致です: {dimensions}, "
            f"expected={list(expected_dimensions)}"
        )
    if metrics != list(expected_metrics):
        raise ValueError(
            f"GA4 Metric Headerが不一致です: {metrics}, expected={list(expected_metrics)}"
        )

    totals = {name: Decimal("0") for name in expected_metrics}
    rows = response.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("GA4 ResponseのrowsがListではありません。")
    for row_number, row in enumerate(rows, 1):
        metric_values = row.get("metricValues", [])
        if len(metric_values) != len(expected_metrics):
            raise ValueError(f"GA4 Row {row_number}のMetric数が不一致です。")
        for name, item in zip(expected_metrics, metric_values, strict=True):
            totals[name] += _decimal(item.get("value"), name)

    row_count = int(response.get("rowCount", len(rows)))
    metadata = response.get("metadata", {})
    data_loss = bool(
        isinstance(metadata, dict) and metadata.get("dataLossFromOtherRow", False)
    )
    partial = data_loss or row_count > len(rows)
    return totals, partial


def _json_number(value: Decimal, *, integer: bool = False) -> int | float:
    if integer:
        if value != value.to_integral_value():
            raise ValueError(f"整数Metricに小数が返されました: {value}")
        return int(value)
    return float(value)


def build_observation(
    publication: Mapping[str, Any],
    start_date: str,
    end_date: str,
    responses: Mapping[str, Mapping[str, Any] | None],
    raw_response_sha256: str,
    collected_at: str,
    *,
    includes_today: bool,
) -> dict[str, Any]:
    main_response = responses.get("main")
    if not isinstance(main_response, Mapping):
        raise ValueError("GA4 Main Responseがありません。")
    main_totals, main_partial = _aggregate_response(
        main_response,
        MAIN_DIMENSIONS,
        MAIN_METRICS,
    )

    page_views = _json_number(main_totals["screenPageViews"], integer=True)
    active_users = _json_number(main_totals["activeUsers"], integer=True)
    engagement = _json_number(main_totals["userEngagementDuration"])
    average_engagement = (
        engagement / active_users if active_users > 0 else None
    )

    cta_tracking = publication["online"]["cta_tracking"]
    cta_enabled = cta_tracking.get("enabled") is True
    cta_partial = False
    if cta_enabled:
        cta_response = responses.get("cta")
        if not isinstance(cta_response, Mapping):
            raise ValueError("CTA Tracking有効時はGA4 CTA Responseが必要です。")
        cta_totals, cta_partial = _aggregate_response(
            cta_response,
            ("pageLocation",),
            CTA_METRICS,
        )
        cta_count: int | None = _json_number(cta_totals["eventCount"], integer=True)
        cta_rate: float | None = (
            cta_count / page_views if page_views > 0 else None
        )
        cta_denominator: str | None = "screen_page_views"
    else:
        cta_count = None
        cta_rate = None
        cta_denominator = None

    partial = includes_today or main_partial or cta_partial
    notes = [
        "average_engagement_time_secはuserEngagementDuration/activeUsersの派生値です。"
    ]
    if includes_today:
        notes.append("期間に収集日当日を含むため、GA4値は後日更新される可能性があります。")
    if main_partial or cta_partial:
        notes.append("GA4 ResponseにData Lossまたは未取得Rowの兆候があります。")
    if not cta_enabled:
        notes.append("CTA計測未実装のため、CTA指標は0ではなくnullです。")

    collection_window_state = (
        "includes_today" if includes_today else "closed_period"
    )
    identity = {
        "publication_id": publication["publication_id"],
        "source": "ga4",
        "collector_version": COLLECTOR_VERSION,
        "start_date": start_date,
        "end_date": end_date,
        "collection_window_state": collection_window_state,
        "raw_response_sha256": raw_response_sha256,
    }
    observation_id = sha256_json(identity)[:24]
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_id": observation_id,
        "publication_id": publication["publication_id"],
        "source": "ga4",
        "collector_version": COLLECTOR_VERSION,
        "collected_at": collected_at,
        "date_range": {
            "start_date": start_date,
            "end_date": end_date,
        },
        "dimensions": {
            "page_location": publication["online"]["published_url"],
            "ga4_property_id": publication["online"]["ga4_property_id"],
            "collection_window_state": collection_window_state,
        },
        "metrics": {
            "screen_page_views": page_views,
            "active_users": active_users,
            "user_engagement_duration_sec": engagement,
            "average_engagement_time_sec": average_engagement,
            "cta_tracking_enabled": cta_enabled,
            "cta_event_count": cta_count,
            "cta_rate": cta_rate,
            "cta_rate_denominator": cta_denominator,
        },
        "raw_response_sha256": raw_response_sha256,
        "is_partial": partial,
        "notes": notes,
    }


def append_observation(path: Path, observation: dict[str, Any]) -> str:
    records: list[dict[str, Any]] = []
    if path.exists():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise TypeError(f"Observation {line_number}行目がObjectではありません。")
            records.append(item)

    for existing in records:
        if existing.get("observation_id") == observation["observation_id"]:
            if existing == observation:
                return "unchanged"
            raise ValueError("同じobservation_idに異なる内容があります。")
    records.append(observation)
    serialized = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
        for item in records
    )
    _atomic_write_text(path, serialized)
    return "created"


def persist_collection(
    *,
    publication: Mapping[str, Any],
    start_date: str,
    end_date: str,
    specs: Mapping[str, Mapping[str, Any] | None],
    responses: Mapping[str, Mapping[str, Any] | None],
    raw_dir: Path,
    observation_path: Path,
    collected_at: str,
    includes_today: bool,
) -> CollectionResult:
    hashed_payload = {
        "collector_version": COLLECTOR_VERSION,
        "collection_context": {
            "includes_today": includes_today,
            "collection_window_state": (
                "includes_today" if includes_today else "closed_period"
            ),
        },
        "requests": specs,
        "responses": responses,
    }
    raw_sha256 = sha256_json(hashed_payload)
    raw_name = (
        f"ga4_{publication['publication_id']}_{start_date}_{end_date}_"
        f"{raw_sha256[:12]}.json"
    )
    raw_path = raw_dir / raw_name
    if raw_path.exists():
        existing_raw = json.loads(raw_path.read_text(encoding="utf-8"))
        if existing_raw.get("raw_response_sha256") != raw_sha256:
            raise ValueError(f"既存Raw ArtifactのHashが不一致です: {raw_path}")
        if existing_raw.get("payload") != hashed_payload:
            raise ValueError(f"既存Raw Artifactの内容が不一致です: {raw_path}")
        effective_collected_at = str(existing_raw["collected_at"])
    else:
        effective_collected_at = collected_at
        raw_artifact = {
            "schema_version": RAW_SCHEMA_VERSION,
            "collector_version": COLLECTOR_VERSION,
            "publication_id": publication["publication_id"],
            "collected_at": collected_at,
            "raw_response_sha256": raw_sha256,
            "payload": hashed_payload,
        }
        _atomic_write_text(
            raw_path,
            json.dumps(raw_artifact, ensure_ascii=False, indent=2) + "\n",
        )

    observation = build_observation(
        publication,
        start_date,
        end_date,
        responses,
        raw_sha256,
        effective_collected_at,
        includes_today=includes_today,
    )
    status = append_observation(observation_path, observation)
    return CollectionResult(
        status=status,
        observation=observation,
        raw_path=raw_path,
        observation_path=observation_path,
    )


def canonical_page_path(publication: Mapping[str, Any]) -> str:
    """将来のPath Dimension検証用にCanonical URLのPathを返す。"""
    return urlsplit(str(publication["online"]["published_url"])).path
