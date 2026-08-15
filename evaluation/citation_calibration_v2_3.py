"""Citationサブスコアの決定論的な校正処理。"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse


CITATION_SUBSCORES = (
    "source_authority",
    "claim_source_alignment",
    "citation_coverage",
    "link_context",
)

URL_PATTERN = re.compile(
    r"https?://[^\s)>\"']+",
    flags=re.IGNORECASE,
)

MARKDOWN_LINK_PATTERN = re.compile(
    r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
    flags=re.IGNORECASE,
)

GENERIC_LINK_LABELS = {
    "url",
    "link",
    "リンク",
    "こちら",
    "ここ",
    "詳細",
    "参考",
    "公式",
    "公式サイト",
}


def _clean_url(url: str) -> str:
    return url.rstrip(
        ".,;:!?。、）]}>"
    )


def _is_public_external_url(
    url: str,
) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname

    if not host:
        return False

    normalized_host = host.lower()

    if normalized_host == "localhost":
        return False

    try:
        ip = ipaddress.ip_address(
            normalized_host
        )

        return not (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
        )

    except ValueError:
        return (
            "." in normalized_host
            and not normalized_host.endswith(
                (".local", ".internal", ".localhost")
            )
        )


def public_external_urls(
    article: str,
) -> list[str]:
    """記事中の公開外部URLを登場順に返す。"""
    urls = [
        _clean_url(url)
        for url in URL_PATTERN.findall(
            article
        )
    ]

    filtered = [
        url
        for url in urls
        if _is_public_external_url(url)
    ]

    return list(
        dict.fromkeys(filtered)
    )


def markdown_public_links(
    article: str,
) -> list[dict[str, str]]:
    """Markdown形式の公開外部リンクを返す。"""
    links: list[dict[str, str]] = []

    for label, raw_url in (
        MARKDOWN_LINK_PATTERN.findall(
            article
        )
    ):
        url = _clean_url(raw_url)

        if not _is_public_external_url(url):
            continue

        links.append(
            {
                "label": label.strip(),
                "url": url,
            }
        )

    return links


def _is_official_primary_source(
    url: str,
) -> bool:
    """本プロジェクトで扱う公式・一次情報を判定する。"""
    parsed = urlparse(url)
    host = (
        parsed.hostname or ""
    ).lower()
    path = parsed.path.lower()

    if (
        host == "mlflow.org"
        or host.endswith(".mlflow.org")
    ):
        return True

    if (
        host == "python.org"
        or host.endswith(".python.org")
    ):
        return True

    if (
        host == "scikit-learn.org"
        or host.endswith(
            ".scikit-learn.org"
        )
    ):
        return True

    if (
        host == "docs.astral.sh"
        or host.endswith(".docs.astral.sh")
    ):
        return True

    if (
        host == "github.com"
        and (
            path.startswith("/mlflow/")
            or path.startswith("/ml-explore/")
        )
    ):
        return True

    return False


def _is_descriptive_label(
    label: str,
    url: str,
) -> bool:
    normalized = label.strip()

    if not normalized:
        return False

    if normalized.lower() in (
        GENERIC_LINK_LABELS
    ):
        return False

    if normalized == url:
        return False

    return len(normalized) >= 4


def calibrate_citation_subscores(
    *,
    article: str,
    raw_subscores: Mapping[str, int],
) -> tuple[
    dict[str, int],
    dict[str, Any],
]:
    """Citationの客観的要素だけを決定論的に校正する。

    意味的な対応関係と網羅性はLLM Judgeの判定を保持する。
    情報源の種類とMarkdownリンク文脈だけを機械的に補正する。
    """
    adjusted = {
        name: int(raw_subscores[name])
        for name in CITATION_SUBSCORES
    }

    urls = public_external_urls(article)
    markdown_links = markdown_public_links(
        article
    )

    audit: dict[str, Any] = {
        "version": (
            "citation-calibration-v2.3.1"
        ),
        "public_external_urls": urls,
        "public_external_url_count": len(
            urls
        ),
        "markdown_links": markdown_links,
        "raw_subscores": dict(
            raw_subscores
        ),
        "adjustments": [],
    }

    if not urls:
        adjusted = {
            name: 1
            for name in CITATION_SUBSCORES
        }

        audit["adjustments"].append(
            {
                "rule": "no-public-source",
                "reason": (
                    "公開URLがないためCitationの"
                    "全サブスコアを1に固定"
                ),
            }
        )
    else:
        official_urls = [
            url
            for url in urls
            if _is_official_primary_source(
                url
            )
        ]

        if len(official_urls) == len(urls):
            previous = adjusted[
                "source_authority"
            ]
            adjusted[
                "source_authority"
            ] = max(previous, 5)

            if previous != 5:
                audit["adjustments"].append(
                    {
                        "rule": (
                            "all-sources-official"
                        ),
                        "subscore": (
                            "source_authority"
                        ),
                        "from": previous,
                        "to": 5,
                    }
                )

        elif official_urls:
            previous = adjusted[
                "source_authority"
            ]
            adjusted[
                "source_authority"
            ] = max(previous, 4)

            if previous < 4:
                audit["adjustments"].append(
                    {
                        "rule": (
                            "official-source-present"
                        ),
                        "subscore": (
                            "source_authority"
                        ),
                        "from": previous,
                        "to": 4,
                    }
                )

        descriptive_urls = {
            link["url"]
            for link in markdown_links
            if _is_descriptive_label(
                link["label"],
                link["url"],
            )
        }

        if (
            urls
            and set(urls).issubset(
                descriptive_urls
            )
        ):
            previous = adjusted[
                "link_context"
            ]
            adjusted[
                "link_context"
            ] = max(previous, 4)

            if previous < 4:
                audit["adjustments"].append(
                    {
                        "rule": (
                            "all-links-descriptive"
                        ),
                        "subscore": (
                            "link_context"
                        ),
                        "from": previous,
                        "to": 4,
                    }
                )

        elif descriptive_urls:
            previous = adjusted[
                "link_context"
            ]
            adjusted[
                "link_context"
            ] = max(previous, 3)

            if previous < 3:
                audit["adjustments"].append(
                    {
                        "rule": (
                            "descriptive-link-present"
                        ),
                        "subscore": (
                            "link_context"
                        ),
                        "from": previous,
                        "to": 3,
                    }
                )

    audit["adjusted_subscores"] = dict(
        adjusted
    )
    audit["adjusted_score"] = round(
        sum(adjusted.values())
        / len(CITATION_SUBSCORES),
        2,
    )

    return adjusted, audit
