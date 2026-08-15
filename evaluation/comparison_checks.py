"""BaselineとCandidateの成功条件を判定する純粋関数。"""

from __future__ import annotations

from collections.abc import Mapping


FLOAT_TOLERANCE = 1e-9
REPRODUCIBILITY_PROXY_TARGET = 0.8


def metric_value(
    metrics: Mapping[str, float],
    name: str,
) -> float:
    """指定したMLflow Metricをfloatとして返す。"""
    if name not in metrics:
        raise KeyError(
            f"Metricがありません: {name}"
        )

    return float(metrics[name])


def not_regressed(
    baseline: float,
    candidate: float,
) -> bool:
    """浮動小数点誤差を考慮して非悪化を判定する。"""
    return (
        candidate + FLOAT_TOLERANCE
        >= baseline
    )


def improved(
    baseline: float,
    candidate: float,
) -> bool:
    """浮動小数点誤差を考慮して改善を判定する。"""
    return (
        candidate
        > baseline + FLOAT_TOLERANCE
    )


def target_met_without_regression(
    baseline: float,
    candidate: float,
    target: float,
) -> bool:
    """目標値を満たし、Baselineから悪化していないか判定する。

    Baselineが満点の場合にもCandidateの満点を成功とする。
    厳密なimprovedだけを使うと、Baseline=1.0では成功不能になる。
    """
    return (
        candidate + FLOAT_TOLERANCE >= target
        and not_regressed(baseline, candidate)
    )


def build_success_checks(
    baseline_metrics: Mapping[str, float],
    candidate_metrics: Mapping[str, float],
) -> dict[str, bool]:
    """実験の成功条件をBaseline/Candidateから判定する。

    `*_not_regressed`は固定値ではなくBaselineとの比較、
    `*_improved`はBaselineを上回ったかで判定する。
    `*_target`だけが絶対値の目標である。
    """
    baseline_structure = metric_value(
        baseline_metrics,
        "structure_score/mean",
    )
    candidate_structure = metric_value(
        candidate_metrics,
        "structure_score/mean",
    )

    baseline_proxy = metric_value(
        baseline_metrics,
        "reproducibility_proxy/mean",
    )
    candidate_proxy = metric_value(
        candidate_metrics,
        "reproducibility_proxy/mean",
    )

    baseline_accuracy = metric_value(
        baseline_metrics,
        "technical_accuracy/mean",
    )
    candidate_accuracy = metric_value(
        candidate_metrics,
        "technical_accuracy/mean",
    )

    baseline_readability = metric_value(
        baseline_metrics,
        "readability_ja/mean",
    )
    candidate_readability = metric_value(
        candidate_metrics,
        "readability_ja/mean",
    )

    return {
        "structure_not_regressed": (
            not_regressed(
                baseline_structure,
                candidate_structure,
            )
        ),
        "reproducibility_proxy_target_and_not_regressed": (
            target_met_without_regression(
                baseline_proxy,
                candidate_proxy,
                REPRODUCIBILITY_PROXY_TARGET,
            )
        ),
        "technical_accuracy_not_regressed": (
            not_regressed(
                baseline_accuracy,
                candidate_accuracy,
            )
        ),
        "helpfulness_target": (
            metric_value(
                candidate_metrics,
                "helpfulness/mean",
            )
            >= 4.0
        ),
        "reproducibility_target": (
            metric_value(
                candidate_metrics,
                "reproducibility/mean",
            )
            >= 3.0
        ),
        "citation_quality_target": (
            metric_value(
                candidate_metrics,
                "citation_quality/mean",
            )
            >= 3.0
        ),
        "readability_not_regressed": (
            not_regressed(
                baseline_readability,
                candidate_readability,
            )
        ),
        "original_value_target": (
            metric_value(
                candidate_metrics,
                "original_value/mean",
            )
            >= 3.0
        ),
    }
