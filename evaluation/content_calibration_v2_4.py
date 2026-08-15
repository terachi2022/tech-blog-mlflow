"""LLM Judgeの内容サブスコアを決定論的に監査・校正する。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from tech_blog_mlflow.article_v3_checks import (
    article_checks,
)


CONTENT_CALIBRATION_VERSION = "content-calibration-v2.4"

API_CORRECT_PATTERN = re.compile(
    r"(?:API|コマンド|使用法).*(?:正しい|適切)",
)

OUT_OF_SCOPE_EXCEPTION_PATTERN = re.compile(
    r"(?:エラーハンドリング|例外処理).*(?:ない|不足|示されていない)",
)

ACTUAL_API_ERROR_PATTERN = re.compile(
    r"(?:API|コマンド|引数).*(?:誤り|不正|間違|実行できない|存在しない)",
)


def _score(
    values: Mapping[str, Mapping[str, int]],
    dimension: str,
    subscore: str,
) -> int:
    try:
        value = int(values[dimension][subscore])
    except KeyError as exc:
        raise KeyError(
            "Content calibrationに必要なSubscoreがありません: "
            f"{dimension}.{subscore}"
        ) from exc

    if value not in {1, 2, 3, 4, 5}:
        raise ValueError(
            "Subscoreは1〜5の整数である必要があります: "
            f"{dimension}.{subscore}={value}"
        )

    return value


def calibrate_content_subscores(
    *,
    article: str,
    raw_subscores: Mapping[str, Mapping[str, int]],
    rationales: Mapping[str, Mapping[str, str]],
    evidence: Mapping[str, bool] | None = None,
) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    """評価ルール違反と決定論的な証拠見落としだけを補正する。

    記事の弱点を隠す一般的なBottom-up補正は行わない。補正対象は、
    APIが正しいとJudge自身が認めながら評価対象外の例外処理だけで
    減点した場合と、5要素を備えたTroubleshootingを3点以下へ
    見落とした場合に限定する。
    """
    adjusted = deepcopy(
        {
            dimension: dict(items)
            for dimension, items in raw_subscores.items()
        }
    )
    observed = (
        dict(evidence)
        if evidence is not None
        else article_checks(article)
    )
    adjustments: list[dict[str, Any]] = []

    api_score = _score(
        raw_subscores,
        "technical_accuracy",
        "api_command_correctness",
    )
    api_rationale = str(
        rationales
        .get("technical_accuracy", {})
        .get("api_command_correctness", "")
    )
    api_evidence_complete = all(
        observed.get(name, False)
        for name in (
            "has_complete_train_code",
            "has_required_shell_commands",
            "exception_handling_is_valid",
        )
    )
    exception_only_penalty = bool(
        API_CORRECT_PATTERN.search(api_rationale)
        and OUT_OF_SCOPE_EXCEPTION_PATTERN.search(api_rationale)
        and not ACTUAL_API_ERROR_PATTERN.search(api_rationale)
    )

    if (
        api_score < 5
        and api_evidence_complete
        and exception_only_penalty
    ):
        adjusted[
            "technical_accuracy"
        ]["api_command_correctness"] = 5
        adjustments.append(
            {
                "dimension": "technical_accuracy",
                "subscore": "api_command_correctness",
                "raw_score": api_score,
                "adjusted_score": 5,
                "reason": (
                    "Judge confirmed API/command correctness but "
                    "deducted only for optional exception handling; "
                    "deterministic code and shell checks passed"
                ),
            }
        )

    troubleshooting_score = _score(
        raw_subscores,
        "helpfulness",
        "troubleshooting_value",
    )
    troubleshooting_evidence_complete = all(
        observed.get(name, False)
        for name in (
            "has_complete_troubleshooting",
            "has_specific_error_messages",
        )
    )

    if (
        troubleshooting_score < 4
        and troubleshooting_evidence_complete
    ):
        adjusted[
            "helpfulness"
        ]["troubleshooting_value"] = 4
        adjustments.append(
            {
                "dimension": "helpfulness",
                "subscore": "troubleshooting_value",
                "raw_score": troubleshooting_score,
                "adjusted_score": 4,
                "reason": (
                    "Four deterministic troubleshooting sections "
                    "contain symptom, check, cause, action, recheck, "
                    "required commands, and expected evidence"
                ),
            }
        )

    audit = {
        "version": CONTENT_CALIBRATION_VERSION,
        "evidence": {
            "has_complete_train_code": bool(
                observed.get("has_complete_train_code", False)
            ),
            "has_required_shell_commands": bool(
                observed.get("has_required_shell_commands", False)
            ),
            "exception_handling_is_valid": bool(
                observed.get("exception_handling_is_valid", False)
            ),
            "has_complete_troubleshooting": bool(
                observed.get("has_complete_troubleshooting", False)
            ),
            "has_specific_error_messages": bool(
                observed.get("has_specific_error_messages", False)
            ),
        },
        "raw_subscores": {
            dimension: dict(items)
            for dimension, items in raw_subscores.items()
        },
        "adjustments": adjustments,
        "adjusted_subscores": adjusted,
    }
    return adjusted, audit


def aggregate_subscores(
    subscores: Mapping[str, Mapping[str, int]],
) -> dict[str, float]:
    """Dimensionごとの単純平均を返す。"""
    aggregates: dict[str, float] = {}

    for dimension, items in subscores.items():
        if not items:
            raise ValueError(
                f"Subscoreが空です: {dimension}"
            )
        aggregates[dimension] = round(
            sum(int(value) for value in items.values())
            / len(items),
            2,
        )

    return aggregates

