"""STEP 5-CのGoogle Search Console Search Analytics収集。

Publication Registryを唯一のAttribution入力とし、記事単位の集計値を
Online Observationへ、Query/Device/Country明細を別Artifactへ保存する。
認証情報は入力・出力・Logへ保存しない。
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from tech_blog_mlflow.ga4_collector import (
    append_observation,
    page_location_aliases,
    select_publication,
    sha256_json,
)


COLLECTOR_VERSION = "gsc-search-analytics-v1.1"
RAW_SCHEMA_VERSION = "gsc-raw-response-v1"
DETAIL_SCHEMA_VERSION = "gsc-search-details-v1"
OBSERVATION_SCHEMA_VERSION = "online-observation-v1"
GSC_SCOPE = "https://www.googleapis.com/auth/webmasters.readonly"
GSC_TIMEZONE = "America/Los_Angeles"
DETAIL_DIMENSIONS = ("query", "device", "country")
DEFAULT_ROW_LIMIT = 25_000
MAX_ROW_LIMIT = 25_000


@dataclass(frozen=True)
class GSCExecutionResult:
    selected_page_expression: str
    aggregate_attempts: list[dict[str, Any]]
    aggregate_response: dict[str, Any]
    detail_requests: list[dict[str, Any]]
    detail_responses: list[dict[str, Any]]


@dataclass(frozen=True)
class GSCCollectionResult:
    status: str
    observation: dict[str, Any]
    raw_path: Path
    detail_path: Path
    observation_path: Path


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


def resolve_gsc_date_range(
    publication: Mapping[str, Any],
    start_date: str | None,
    end_date: str | None,
    *,
    today: date | None = None,
) -> tuple[str, str, bool]:
    """Search Analytics仕様どおりPTの日付として期間を検証する。"""
    timezone = ZoneInfo(GSC_TIMEZONE)
    published = datetime.fromisoformat(str(publication["online"]["published_at"]))
    published_date = published.astimezone(timezone).date()
    current_date = today or datetime.now(timezone).date()
    try:
        start = date.fromisoformat(start_date) if start_date else published_date
        end = date.fromisoformat(end_date) if end_date else current_date
    except ValueError as error:
        raise ValueError("start-date/end-dateはYYYY-MM-DD形式で指定してください。") from error
    if start < published_date:
        raise ValueError(
            "start-dateはGSCのPT基準の公開日以降にしてください: "
            f"published={published_date}, start={start}"
        )
    if end < start:
        raise ValueError("end-dateはstart-date以降にしてください。")
    if end > current_date:
        raise ValueError(
            f"end-dateにPT基準の未来日は指定できません: "
            f"today={current_date}, end={end}"
        )
    return start.isoformat(), end.isoformat(), end == current_date


def _page_filter(expression: str) -> list[dict[str, Any]]:
    return [
        {
            "groupType": "and",
            "filters": [
                {
                    "dimension": "page",
                    "operator": "equals",
                    "expression": expression,
                }
            ],
        }
    ]


def build_aggregate_request(
    page_expression: str,
    start_date: str,
    end_date: str,
    *,
    data_state: str = "all",
) -> dict[str, Any]:
    if data_state not in {"all", "final"}:
        raise ValueError("data-stateはallまたはfinalです。")
    return {
        "startDate": start_date,
        "endDate": end_date,
        "type": "web",
        "dimensionFilterGroups": _page_filter(page_expression),
        "aggregationType": "auto",
        "dataState": data_state,
        "rowLimit": 1,
        "startRow": 0,
    }


def build_detail_request(
    page_expression: str,
    start_date: str,
    end_date: str,
    *,
    data_state: str = "all",
    row_limit: int = DEFAULT_ROW_LIMIT,
    start_row: int = 0,
) -> dict[str, Any]:
    if not 1 <= row_limit <= MAX_ROW_LIMIT:
        raise ValueError(f"row-limitは1〜{MAX_ROW_LIMIT}で指定してください。")
    if start_row < 0:
        raise ValueError("start-rowは0以上で指定してください。")
    request = build_aggregate_request(
        page_expression,
        start_date,
        end_date,
        data_state=data_state,
    )
    request.update(
        {
            "dimensions": list(DETAIL_DIMENSIONS),
            "rowLimit": row_limit,
            "startRow": start_row,
        }
    )
    return request


def build_request_plan(
    publication: Mapping[str, Any],
    start_date: str,
    end_date: str,
    *,
    data_state: str = "all",
    row_limit: int = DEFAULT_ROW_LIMIT,
) -> dict[str, Any]:
    online = publication["online"]
    site_url = online.get("gsc_site_url")
    if not isinstance(site_url, str) or not site_url:
        raise ValueError("Registry online.gsc_site_urlがありません。")
    aliases = page_location_aliases(str(online["published_url"]))
    return {
        "site_url": site_url,
        "page_expressions": list(aliases),
        "aggregate_requests": [
            build_aggregate_request(
                alias,
                start_date,
                end_date,
                data_state=data_state,
            )
            for alias in aliases
        ],
        "detail_request_template": build_detail_request(
            aliases[0],
            start_date,
            end_date,
            data_state=data_state,
            row_limit=row_limit,
        ),
        "detail_dimensions": list(DETAIL_DIMENSIONS),
        "date_timezone": GSC_TIMEZONE,
    }


def _import_gsc_dependencies() -> tuple[Callable[..., Any], Callable[..., Any]]:
    try:
        import google.auth
        from googleapiclient.discovery import build
    except ImportError as error:
        raise RuntimeError(
            "google-api-python-clientがありません。"
            "`uv sync`または`uv add google-api-python-client==2.198.0`を"
            "実行してください。"
        ) from error
    return google.auth.default, build


def make_query_executor() -> Callable[[str, Mapping[str, Any]], dict[str, Any]]:
    default_credentials, build = _import_gsc_dependencies()
    credentials, _ = default_credentials(scopes=[GSC_SCOPE])
    service = build(
        "searchconsole",
        "v1",
        credentials=credentials,
        cache_discovery=False,
    )

    def execute(site_url: str, body: Mapping[str, Any]) -> dict[str, Any]:
        response = service.searchanalytics().query(
            siteUrl=site_url,
            body=dict(body),
        ).execute()
        if not isinstance(response, dict):
            raise TypeError("Search Analytics ResponseがObjectではありません。")
        return json.loads(json.dumps(response, ensure_ascii=False))

    return execute


def _rows(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = response.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("Search Analytics ResponseのrowsがListではありません。")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("Search Analytics ResponseのrowがObjectではありません。")
    return rows


def _metric_tuple(response: Mapping[str, Any]) -> tuple[Decimal, ...] | None:
    rows = _rows(response)
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError("DimensionなしAggregate Responseは最大1 Rowです。")
    row = rows[0]
    return tuple(
        _decimal(row.get(name), name)
        for name in ("clicks", "impressions", "ctr", "position")
    )


def execute_search_analytics(
    plan: Mapping[str, Any],
    *,
    query_executor: Callable[[str, Mapping[str, Any]], dict[str, Any]] | None = None,
    row_limit: int = DEFAULT_ROW_LIMIT,
) -> GSCExecutionResult:
    executor = query_executor or make_query_executor()
    site_url = str(plan["site_url"])
    aliases = [str(value) for value in plan["page_expressions"]]
    aggregate_requests = list(plan["aggregate_requests"])
    attempts: list[dict[str, Any]] = []
    nonempty: list[tuple[str, dict[str, Any], tuple[Decimal, ...]]] = []
    for alias, request in zip(aliases, aggregate_requests, strict=True):
        response = executor(site_url, request)
        attempts.append(
            {
                "page_expression": alias,
                "request": request,
                "response": response,
            }
        )
        metrics = _metric_tuple(response)
        if metrics is not None:
            nonempty.append((alias, response, metrics))

    if len(nonempty) > 1:
        metric_sets = {item[2] for item in nonempty}
        if len(metric_sets) > 1:
            raise ValueError(
                "Canonical URIとUnicode IRIの両方が異なるGSC値を返しました。"
                "二重計上を避けるため収集を中止します。"
            )
    if nonempty:
        selected_alias, aggregate_response, _ = nonempty[0]
    else:
        selected_alias = aliases[0]
        aggregate_response = attempts[0]["response"]

    template = dict(plan["detail_request_template"])
    template["dimensionFilterGroups"] = _page_filter(selected_alias)
    detail_requests: list[dict[str, Any]] = []
    detail_responses: list[dict[str, Any]] = []
    start_row = 0
    while True:
        request = dict(template)
        request["rowLimit"] = row_limit
        request["startRow"] = start_row
        response = executor(site_url, request)
        rows = _rows(response)
        detail_requests.append(request)
        detail_responses.append(response)
        if len(rows) < row_limit:
            break
        start_row += row_limit

    return GSCExecutionResult(
        selected_page_expression=selected_alias,
        aggregate_attempts=attempts,
        aggregate_response=aggregate_response,
        detail_requests=detail_requests,
        detail_responses=detail_responses,
    )


def _decimal(value: Any, label: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"GSC Metric {label}が数値ではありません: {value!r}") from error
    if not number.is_finite() or number < 0:
        raise ValueError(f"GSC Metric {label}は有限の非負値である必要があります。")
    return number


def _json_number(value: Decimal) -> int | float:
    return int(value) if value == value.to_integral_value() else float(value)


def aggregate_metrics(response: Mapping[str, Any]) -> dict[str, int | float | None]:
    rows = _rows(response)
    if not rows:
        return {"clicks": 0, "impressions": 0, "ctr": None, "position": None}
    if len(rows) != 1:
        raise ValueError("DimensionなしAggregate Responseは最大1 Rowです。")
    row = rows[0]
    clicks = _decimal(row.get("clicks"), "clicks")
    impressions = _decimal(row.get("impressions"), "impressions")
    ctr = _decimal(row.get("ctr"), "ctr")
    position = _decimal(row.get("position"), "position")
    if ctr > 1:
        raise ValueError("GSC Metric ctrは0〜1である必要があります。")
    if impressions == 0:
        if clicks != 0:
            raise ValueError(
                "GSC Aggregateはimpressions=0なのにclicks>0です。"
                "不整合なResponseのためObservationを作成しません。"
            )
        return {
            "clicks": 0,
            "impressions": 0,
            "ctr": None,
            "position": None,
        }
    return {
        "clicks": _json_number(clicks),
        "impressions": _json_number(impressions),
        "ctr": float(ctr),
        "position": float(position),
    }


def build_detail_records(responses: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for response in responses:
        for row in _rows(response):
            keys = row.get("keys", [])
            if not isinstance(keys, list) or len(keys) != len(DETAIL_DIMENSIONS):
                raise ValueError("GSC Detail Rowのkeys数がDimension数と一致しません。")
            metrics = aggregate_metrics({"rows": [row]})
            records.append(
                {
                    **dict(zip(DETAIL_DIMENSIONS, map(str, keys), strict=True)),
                    **metrics,
                }
            )
    return records


def _response_is_partial(response: Mapping[str, Any]) -> bool:
    metadata = response.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return False
    return bool(
        metadata.get("firstIncompleteDate")
        or metadata.get("first_incomplete_date")
        or metadata.get("firstIncompleteHour")
        or metadata.get("first_incomplete_hour")
    )


def build_observation(
    publication: Mapping[str, Any],
    start_date: str,
    end_date: str,
    execution: GSCExecutionResult,
    raw_response_sha256: str,
    detail_sha256: str,
    collected_at: str,
    *,
    includes_today: bool,
    data_state: str,
) -> dict[str, Any]:
    metrics = aggregate_metrics(execution.aggregate_response)
    response_partial = _response_is_partial(execution.aggregate_response) or any(
        _response_is_partial(response) for response in execution.detail_responses
    )
    partial = includes_today or response_partial
    notes = [
        "Search Analyticsの日付はAmerica/Los_Angeles（PT）基準です。",
        "Query/Device/Country明細は別Artifactに保存し、記事集計値へ混在させません。",
        "Search Analytics APIは内部制限により全明細Rowを保証せず、上位Rowを返します。",
    ]
    if data_state == "all":
        notes.append("dataState=allのためFresh Dataを含み、値が後から変わる可能性があります。")
    if includes_today:
        notes.append("期間にPT基準の収集日当日を含みます。")
    if response_partial:
        notes.append("GSC Response Metadataが未確定期間を示しています。")
    aggregate_row_count = len(_rows(execution.aggregate_response))
    if aggregate_row_count == 0:
        notes.append("APIは正常応答しましたが対象記事のAggregate Rowは0件でした。")
    elif metrics["impressions"] == 0:
        notes.append(
            "Aggregate Responseはゼロ指標Rowでした。"
            "分母と掲載実績がないためCTRとPositionはnullへ正規化しました。"
        )

    identity = {
        "publication_id": publication["publication_id"],
        "source": "gsc",
        "collector_version": COLLECTOR_VERSION,
        "start_date": start_date,
        "end_date": end_date,
        "raw_response_sha256": raw_response_sha256,
    }
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_id": sha256_json(identity)[:24],
        "publication_id": publication["publication_id"],
        "source": "gsc",
        "collector_version": COLLECTOR_VERSION,
        "collected_at": collected_at,
        "date_range": {"start_date": start_date, "end_date": end_date},
        "dimensions": {
            "page": publication["online"]["published_url"],
            "matched_page_expression": execution.selected_page_expression,
            "gsc_site_url": publication["online"]["gsc_site_url"],
            "search_type": "web",
            "data_state": data_state,
            "date_timezone": GSC_TIMEZONE,
            "detail_artifact_sha256": detail_sha256,
        },
        "metrics": metrics,
        "raw_response_sha256": raw_response_sha256,
        "is_partial": partial,
        "notes": notes,
    }


def persist_collection(
    *,
    publication: Mapping[str, Any],
    start_date: str,
    end_date: str,
    plan: Mapping[str, Any],
    execution: GSCExecutionResult,
    raw_dir: Path,
    detail_dir: Path,
    observation_path: Path,
    collected_at: str,
    includes_today: bool,
    data_state: str,
) -> GSCCollectionResult:
    payload = {
        "collector_version": COLLECTOR_VERSION,
        "request_plan": plan,
        "selected_page_expression": execution.selected_page_expression,
        "aggregate_attempts": execution.aggregate_attempts,
        "detail_requests": execution.detail_requests,
        "detail_responses": execution.detail_responses,
        "collection_context": {
            "includes_today": includes_today,
            "data_state": data_state,
            "date_timezone": GSC_TIMEZONE,
        },
    }
    raw_sha256 = sha256_json(payload)
    raw_path = raw_dir / (
        f"gsc_{publication['publication_id']}_{start_date}_{end_date}_"
        f"{raw_sha256[:12]}.json"
    )

    details = build_detail_records(execution.detail_responses)
    detail_payload = {
        "schema_version": DETAIL_SCHEMA_VERSION,
        "collector_version": COLLECTOR_VERSION,
        "publication_id": publication["publication_id"],
        "date_range": {"start_date": start_date, "end_date": end_date},
        "page_expression": execution.selected_page_expression,
        "dimensions": list(DETAIL_DIMENSIONS),
        "row_count": len(details),
        "rows": details,
        "limitations": [
            "Search Analytics APIは全Rowではなく上位Rowを返す場合があります。",
            "Privacy保護により一部Queryが省略される場合があります。",
        ],
    }
    detail_sha256 = sha256_json(detail_payload)
    detail_payload["detail_sha256"] = detail_sha256
    detail_path = detail_dir / (
        f"gsc_details_{publication['publication_id']}_{start_date}_{end_date}_"
        f"{detail_sha256[:12]}.json"
    )

    if raw_path.exists():
        existing_raw = json.loads(raw_path.read_text(encoding="utf-8"))
        if existing_raw.get("raw_response_sha256") != raw_sha256:
            raise ValueError(f"既存Raw ArtifactのHashが不一致です: {raw_path}")
        if existing_raw.get("payload") != payload:
            raise ValueError(f"既存Raw Artifactの内容が不一致です: {raw_path}")
        effective_collected_at = str(existing_raw["collected_at"])
    else:
        effective_collected_at = collected_at
        _atomic_write_text(
            raw_path,
            json.dumps(
                {
                    "schema_version": RAW_SCHEMA_VERSION,
                    "collector_version": COLLECTOR_VERSION,
                    "publication_id": publication["publication_id"],
                    "collected_at": collected_at,
                    "raw_response_sha256": raw_sha256,
                    "payload": payload,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )

    if detail_path.exists():
        existing_detail = json.loads(detail_path.read_text(encoding="utf-8"))
        if existing_detail != detail_payload:
            raise ValueError(f"既存Detail Artifactの内容が不一致です: {detail_path}")
    else:
        _atomic_write_text(
            detail_path,
            json.dumps(detail_payload, ensure_ascii=False, indent=2) + "\n",
        )

    observation = build_observation(
        publication,
        start_date,
        end_date,
        execution,
        raw_sha256,
        detail_sha256,
        effective_collected_at,
        includes_today=includes_today,
        data_state=data_state,
    )
    status = append_observation(observation_path, observation)
    return GSCCollectionResult(
        status=status,
        observation=observation,
        raw_path=raw_path,
        detail_path=detail_path,
        observation_path=observation_path,
    )
