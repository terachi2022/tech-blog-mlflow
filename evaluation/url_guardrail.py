"""Citation評価で使用するURL抽出・Guardrail処理。

記事本文からHTTP/HTTPS URLを抽出し、既存Code Scorerと同じ
_public_external_urls()を利用して、公開外部URLだけを残す。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from evaluation.scorers import _public_external_urls


URL_PATTERN = re.compile(
    r"https?://[^\s<>\[\]{}\"']+",
    flags=re.IGNORECASE,
)

URL_TRAILING_PUNCTUATION = (
    ".,;:!?、。）」』】}>"
)


def normalize_article_text(
    value: Any,
) -> str:
    """MLflow scorerへ渡された出力を記事本文の文字列へ正規化する。

    Args:
        value:
            str、dict、その他のMLflow outputs。

    Returns:
        記事本文。
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, Mapping):
        article_keys = (
            "article",
            "response",
            "output",
            "text",
            "content",
        )

        for key in article_keys:
            candidate = value.get(key)

            if isinstance(candidate, str):
                return candidate

    return str(value)


def extract_url_candidates(
    article: str,
) -> list[str]:
    """記事本文からHTTP/HTTPS URL候補を抽出する。"""
    normalized_article = normalize_article_text(
        article
    )

    candidates: list[str] = []

    for matched_url in URL_PATTERN.findall(
        normalized_article
    ):
        cleaned_url = matched_url.rstrip(
            URL_TRAILING_PUNCTUATION
        )

        if cleaned_url:
            candidates.append(
                cleaned_url
            )

    return candidates


def extract_public_external_urls(
    article: str,
) -> list[str]:
    """公開外部URLのみを抽出する。

    `_public_external_urls()`には記事本文ではなく、
    抽出済みURLのリストを渡す。
    """
    candidates = extract_url_candidates(
        article
    )

    public_urls = _public_external_urls(
        candidates
    )

    # URLの登場順を維持しながら重複を除外する
    return list(
        dict.fromkeys(public_urls)
    )


def apply_citation_guardrail(
    *,
    article: str,
    aggregate_scores: Mapping[str, float],
    judge_record: dict[str, Any],
) -> dict[str, float]:
    """URL有無によるCitation Guardrailを適用する。

    公開外部URLがない記事は、citation_qualityを1.0に固定する。
    URLがある場合はLLM Judgeのスコアを変更しない。

    Args:
        article:
            評価対象の記事本文。
        aggregate_scores:
            LLM Judgeが算出した6評価軸の集約値。
        judge_record:
            JSONへ保存するJudge実行記録。

    Returns:
        Guardrail適用後の集約スコア。
    """
    public_urls = extract_public_external_urls(
        article
    )

    adjusted_scores = {
        key: float(value)
        for key, value in aggregate_scores.items()
    }

    citation_original_score = float(
        adjusted_scores.get(
            "citation_quality",
            1.0,
        )
    )

    citation_override = (
        len(public_urls) == 0
    )

    if citation_override:
        adjusted_scores[
            "citation_quality"
        ] = 1.0

    judge_record["guardrails"] = {
        "version": (
            "citation-url-guardrail-v2"
        ),
        "public_external_urls": (
            public_urls
        ),
        "public_external_url_count": (
            len(public_urls)
        ),
        "citation_override": (
            citation_override
        ),
        "citation_original_score": (
            citation_original_score
        ),
        "citation_adjusted_score": float(
            adjusted_scores[
                "citation_quality"
            ]
        ),
        "reason": (
            "No public external URL"
            if citation_override
            else (
                "Public external URLs found; "
                "LLM Judge score retained"
            )
        ),
    }

    judge_record[
        "adjusted_aggregate_scores"
    ] = dict(adjusted_scores)

    return adjusted_scores

