"""STEP 4 Skills-only実験の成功条件を判定する純粋関数。"""

from __future__ import annotations

from collections.abc import Mapping


FLOAT_TOLERANCE = 1e-9
MINIMUM_SKILL_IMPROVEMENT = 0.25

LLM_JUDGE_METRICS = (
    "technical_accuracy/mean",
    "helpfulness/mean",
    "reproducibility/mean",
    "citation_quality/mean",
    "readability_ja/mean",
    "original_value/mean",
)

CODE_GUARD_METRICS = (
    "structure_score/mean",
    "reproducibility_proxy/mean",
    "has_prerequisites/mean",
    "has_version_info/mean",
    "has_failure_cases/mean",
)


def metric_value(
    metrics: Mapping[str, float],
    name: str,
) -> float:
    """必須Metricをfloatとして返す。"""
    if name not in metrics:
        raise KeyError(
            f"Metricがありません: {name}"
        )

    return float(metrics[name])


def not_regressed(
    baseline: float,
    candidate: float,
) -> bool:
    """浮動小数点誤差を考慮して非劣化を判定する。"""
    return (
        candidate + FLOAT_TOLERANCE
        >= baseline
    )


def llm_judge_mean(
    metrics: Mapping[str, float],
) -> float:
    """6軸の単純平均を返す。"""
    return sum(
        metric_value(metrics, name)
        for name in LLM_JUDGE_METRICS
    ) / len(LLM_JUDGE_METRICS)


def build_skill_success_checks(
    baseline_metrics: Mapping[str, float],
    candidate_metrics: Mapping[str, float],
    *,
    candidate_prechecks_passed: bool,
) -> dict[str, bool]:
    """Skills-only実験の採用条件を返す。

    採用済みSTEP 3候補は絶対目標を満たしているため、STEP 4では
    全品質軸と構造Guardを非劣化条件にする。そのうえで、Skillを
    採用する根拠として6軸のいずれかに0.25以上の改善を要求する。
    """
    checks: dict[str, bool] = {
        "candidate_prechecks_passed": (
            candidate_prechecks_passed
        ),
    }

    for name in CODE_GUARD_METRICS:
        short_name = name.removesuffix(
            "/mean"
        )
        checks[f"{short_name}_not_regressed"] = (
            not_regressed(
                metric_value(
                    baseline_metrics,
                    name,
                ),
                metric_value(
                    candidate_metrics,
                    name,
                ),
            )
        )

    improvements: list[float] = []

    for name in LLM_JUDGE_METRICS:
        short_name = name.removesuffix(
            "/mean"
        )
        baseline = metric_value(
            baseline_metrics,
            name,
        )
        candidate = metric_value(
            candidate_metrics,
            name,
        )
        improvements.append(
            candidate - baseline
        )
        checks[f"{short_name}_not_regressed"] = (
            not_regressed(
                baseline,
                candidate,
            )
        )

    baseline_mean = llm_judge_mean(
        baseline_metrics
    )
    candidate_mean = llm_judge_mean(
        candidate_metrics
    )

    checks["llm_judge_mean_not_regressed"] = (
        not_regressed(
            baseline_mean,
            candidate_mean,
        )
    )
    checks["skill_value_demonstrated"] = (
        max(improvements)
        + FLOAT_TOLERANCE
        >= MINIMUM_SKILL_IMPROVEMENT
    )

    return checks
