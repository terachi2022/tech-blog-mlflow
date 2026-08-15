"""STEP 5の公開記事Registryを検証・更新する。

Offline評価値と公開後のOnline指標は別々に保存する。このModuleは
両者を結合するための不変なIdentityだけを登録し、PVなどの値は扱わない。
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


REGISTRY_SCHEMA_VERSION = "online-publication-registry-v1.2"
OFFLINE_REFERENCE_VERSION = "offline-reference-v1"
ONLINE_TARGET_VERSION = "online-target-v1.2"

RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GA4_PROPERTY_PATTERN = re.compile(r"^(?:properties/)?([0-9]+)$")
CTA_EVENT_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,39}$")

BLOCKED_HOST_SUFFIXES = (
    ".internal",
    ".lan",
    ".local",
    ".localhost",
    ".home",
)


def load_json_object(path: Path) -> dict[str, Any]:
    """JSON Objectを読み込み、存在とRoot型を検証する。"""
    if not path.is_file():
        raise FileNotFoundError(f"JSONがありません: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON RootはObjectである必要があります: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_run_id(value: Any, label: str) -> str:
    text = str(value or "")
    if not RUN_ID_PATTERN.fullmatch(text):
        raise ValueError(f"{label}は32桁の小文字16進数である必要があります。")
    return text


def require_sha256(value: Any, label: str) -> str:
    text = str(value or "")
    if not SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{label}は64桁の小文字SHA-256である必要があります。")
    return text


def _public_hostname(hostname: str) -> str:
    try:
        host = hostname.rstrip(".").encode("idna").decode("ascii").lower()
    except UnicodeError as error:
        raise ValueError("公開URLのDomainをIDNAへ変換できません。") from error
    if not host or host == "localhost" or host.endswith(BLOCKED_HOST_SUFFIXES):
        raise ValueError("公開URLにLocal/Internal Hostは使用できません。")

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if "." not in host:
            raise ValueError("公開URLには完全修飾Domainを指定してください。") from None
    else:
        if not address.is_global:
            raise ValueError("公開URLにPrivate/Reserved IPは使用できません。")
    return host


def normalize_public_article_url(value: str) -> str:
    """公開記事URLを検証し、Scheme/Hostだけ正規化する。"""
    if value != value.strip() or any(character.isspace() for character in value):
        raise ValueError("公開記事URLに空白を含めることはできません。")
    parsed = urlsplit(value)
    if parsed.scheme.lower() != "https":
        raise ValueError("公開記事URLはHTTPSで指定してください。")
    if parsed.username or parsed.password:
        raise ValueError("公開記事URLに認証情報を含めることはできません。")
    if parsed.query or parsed.fragment:
        raise ValueError("Canonical URLにはQuery/Fragmentを含めないでください。")
    if not parsed.hostname:
        raise ValueError("公開記事URLにHostがありません。")

    host = _public_hostname(parsed.hostname)
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("公開記事URLのPortが不正です。") from error
    if port not in (None, 443):
        raise ValueError("公開記事URLは標準HTTPS Portを使用してください。")

    path = parsed.path or "/"
    if not path.startswith("/"):
        raise ValueError("公開記事URLのPathが不正です。")
    # IRIの日本語Pathと既存のPercent Encodingを同じCanonical URIへ揃える。
    invalid_percent = re.search(r"%(?![0-9A-Fa-f]{2})", path)
    if invalid_percent:
        raise ValueError("公開記事URLのPercent Encodingが不正です。")
    path = quote(
        path,
        safe="/:@-._~!$&'()*+,;=%",
    )
    path = re.sub(
        r"%[0-9A-Fa-f]{2}",
        lambda match: match.group(0).lower(),
        path,
    )
    netloc = host if port is None else f"{host}:443"
    return urlunsplit(("https", netloc, path, "", ""))


def normalize_published_at(value: str) -> str:
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        timestamp = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError("published-atはISO 8601形式で指定してください。") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("published-atにはTimezone Offsetが必要です。")
    return timestamp.isoformat(timespec="seconds")


def normalize_ga4_property_id(value: str) -> str:
    match = GA4_PROPERTY_PATTERN.fullmatch(value.strip())
    if not match:
        raise ValueError("GA4 Property IDは数字、またはproperties/<数字>で指定してください。")
    return match.group(1)


def normalize_gsc_site_url(value: str) -> str:
    text = value.strip()
    if text.startswith("sc-domain:"):
        domain = text.removeprefix("sc-domain:").strip().lower().rstrip(".")
        if not domain or "/" in domain or ":" in domain:
            raise ValueError("Search Console Domain Propertyが不正です。")
        _public_hostname(domain)
        return f"sc-domain:{domain}"

    normalized = normalize_public_article_url(text)
    parsed = urlsplit(normalized)
    if parsed.path != "/" and not parsed.path.endswith("/"):
        normalized = urlunsplit(
            (parsed.scheme, parsed.netloc, f"{parsed.path}/", "", "")
        )
    return normalized


def normalize_cta_event_name(value: str) -> str:
    text = value.strip()
    if not CTA_EVENT_PATTERN.fullmatch(text):
        raise ValueError(
            "CTA Event名は英字で始まる40文字以内の英数字/Underscoreにしてください。"
        )
    return text


def normalize_measurement_timezone(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("measurement-timezoneは空にできません。")
    try:
        ZoneInfo(text)
    except ZoneInfoNotFoundError as error:
        raise ValueError("measurement-timezoneはIANA Time Zoneで指定してください。") from error
    return text


def _project_file(project_root: Path, relative_value: Any) -> tuple[str, Path]:
    relative = Path(str(relative_value or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("article_pathはProject内の相対Pathである必要があります。")
    root = project_root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("article_pathがProject外を参照しています。") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"記事がありません: {relative}")
    return relative.as_posix(), resolved


def validate_offline_reference(
    project_root: Path,
    generation: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Generation/Evaluation JSONと記事実体のIdentityを照合する。"""
    required_generation = {
        "run_id",
        "article_path",
        "article_sha256",
        "model",
        "prompt_version",
        "generation_config_version",
        "all_prechecks_passed",
    }
    missing_generation = sorted(required_generation.difference(generation))
    if missing_generation:
        raise KeyError(f"Generation JSONの必須Keyがありません: {missing_generation}")
    if generation["all_prechecks_passed"] is not True:
        raise ValueError("事前検査を通過していないGeneration Runは公開登録できません。")

    generation_run_id = require_run_id(generation["run_id"], "Generation Run ID")
    article_sha256 = require_sha256(generation["article_sha256"], "Article SHA-256")
    article_path, article_file = _project_file(project_root, generation["article_path"])
    actual_sha256 = sha256_file(article_file)
    if actual_sha256 != article_sha256:
        raise ValueError(
            "記事SHA-256がGeneration Metadataと一致しません: "
            f"metadata={article_sha256}, actual={actual_sha256}"
        )

    evaluation_run_id = require_run_id(evaluation.get("run_id"), "Evaluation Run ID")
    article = evaluation.get("article")
    if not isinstance(article, dict):
        raise KeyError("Evaluation JSONにarticle Objectがありません。")
    expected_article = {
        "path": article_path,
        "sha256": article_sha256,
        "source_run_id": generation_run_id,
        "generator_prompt_version": generation["prompt_version"],
    }
    for name, expected in expected_article.items():
        if article.get(name) != expected:
            raise ValueError(
                "Evaluation JSONとGeneration JSONの対応が不一致です: "
                f"article.{name}={article.get(name)!r}, expected={expected!r}"
            )

    combined_version = str(evaluation.get("combined_version") or "")
    judge = evaluation.get("judge")
    if not combined_version or not isinstance(judge, dict):
        raise KeyError("Evaluation VersionまたはJudge Metadataがありません。")
    judge_prompt_version = str(judge.get("prompt_version") or "")
    judge_model = str(judge.get("model") or "")
    if not judge_prompt_version or not judge_model:
        raise KeyError("Judge Model/Prompt Versionがありません。")

    skill = generation.get("skill")
    skills_enabled = bool(skill.get("enabled")) if isinstance(skill, dict) else False
    return {
        "version": OFFLINE_REFERENCE_VERSION,
        "article_path": article_path,
        "article_sha256": article_sha256,
        "generation_run_id": generation_run_id,
        "generation_model": str(generation["model"]),
        "generator_prompt_version": str(generation["prompt_version"]),
        "generation_config_version": str(generation["generation_config_version"]),
        "evaluation_run_id": evaluation_run_id,
        "combined_evaluation_version": combined_version,
        "judge_model": judge_model,
        "judge_prompt_version": judge_prompt_version,
        "skills_enabled": skills_enabled,
    }


def build_publication_record(
    *,
    project_root: Path,
    generation: dict[str, Any],
    evaluation: dict[str, Any],
    published_url: str,
    published_at: str,
    ga4_property_id: str,
    gsc_site_url: str,
    cta_event_name: str | None,
    measurement_timezone: str,
) -> dict[str, Any]:
    offline = validate_offline_reference(project_root, generation, evaluation)
    online = {
        "version": ONLINE_TARGET_VERSION,
        "published_url": normalize_public_article_url(published_url),
        "published_at": normalize_published_at(published_at),
        "measurement_timezone": normalize_measurement_timezone(measurement_timezone),
        "ga4_property_id": normalize_ga4_property_id(ga4_property_id),
        "gsc_site_url": normalize_gsc_site_url(gsc_site_url),
        "cta_tracking": {
            "enabled": cta_event_name is not None,
            "event_name": (
                normalize_cta_event_name(cta_event_name)
                if cta_event_name is not None
                else None
            ),
        },
    }
    identity = "\n".join(
        (
            offline["article_sha256"],
            offline["generation_run_id"],
            online["published_url"],
        )
    )
    publication_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "publication_id": publication_id,
        "offline": offline,
        "online": online,
    }


def load_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"Registry {line_number}行目がObjectではありません。")
        records.append(value)
    return records


def append_publication_record(path: Path, record: dict[str, Any]) -> str:
    """RecordをAtomicに追記する。同一Recordの再実行は変更しない。"""
    records = load_registry(path)
    for existing in records:
        if existing.get("publication_id") == record["publication_id"]:
            if existing == record:
                return "unchanged"
            raise ValueError("同じpublication_idに異なる内容が登録されています。")
        if existing.get("online", {}).get("published_url") == record["online"]["published_url"]:
            raise ValueError("同じ公開記事URLが別Recordとして登録されています。")
        if existing.get("offline", {}).get("article_sha256") == record["offline"]["article_sha256"]:
            raise ValueError("同じArticle SHA-256が別URLとして登録されています。")

    path.parent.mkdir(parents=True, exist_ok=True)
    records.append(record)
    serialized = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
        for item in records
    )
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
    return "created"
