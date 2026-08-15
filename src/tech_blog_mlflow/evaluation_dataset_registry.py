"""STEP-α-4: 記事と人手評価をMLflow Evaluation Datasetへ固定する。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "tech-blog-generation"
WORKFLOW_VERSION = "step-alpha-4-dataset-v1.0.0"
DATASET_NAME = "tech-blog-article-quality-calibration-v1"
THEME = "MLflowを使って機械学習の実験を管理する方法"


@dataclass(frozen=True)
class ArticleRecordSpec:
    variant: str
    article_path: Path
    article_sha256: str
    generation_run_id: str
    evaluation_run_id: str
    evaluation_trace_id: str
    generator_prompt_version: str
    human_scores: dict[str, int]
    publishable: bool
    review_notes: str


def article_specs(project_root: Path = Path(".")) -> tuple[ArticleRecordSpec, ...]:
    return (
        ArticleRecordSpec(
            variant="baseline",
            article_path=project_root / "articles/baseline_20260814_004017.md",
            article_sha256="181ba7b4c01e3d4743cdf35f9f3380d87ffeecf04fc13b9100dffcc30ff3c013",
            generation_run_id="b7dfd7ec5d0c4439873da3684fc2c5b2",
            evaluation_run_id="4f56c781fdfb4e95805c6b957302373f",
            evaluation_trace_id="tr-72032620757663af5d9912a3aebb7e58",
            generator_prompt_version="baseline-v1",
            human_scores={
                "technical_accuracy": 3,
                "helpfulness": 3,
                "reproducibility": 2,
                "citation_quality": 2,
                "readability_ja": 3,
                "original_value": 2,
            },
            publishable=False,
            review_notes="非常にシンプルで読みやすいが情報量が圧倒的に少なすぎる。",
        ),
        ArticleRecordSpec(
            variant="prompt-v3.5.2",
            article_path=project_root / "articles/prompt_v3_5_2_20260814_164216.md",
            article_sha256="20aec80f03005b30fa85896267bdb81efb122a36ca52c5ef680c70ba2343f824",
            generation_run_id="b5c925c2322b4e30b04f07e24d160a04",
            evaluation_run_id="fdf0c239445f44a0999a6b1fe7a419b6",
            evaluation_trace_id="tr-4c2e4985a03dd19d214bd64a2094d00a",
            generator_prompt_version="article-v3.5.2",
            human_scores={
                "technical_accuracy": 3,
                "helpfulness": 3,
                "reproducibility": 4,
                "citation_quality": 3,
                "readability_ja": 3,
                "original_value": 3,
            },
            publishable=False,
            review_notes=(
                "最初にしては、最低限出来ているが、どんな課題を解決するために、"
                "この記事があるのかがわからない。\n"
                "課題・目的・概要を全体的に入れることと、全体的なフローの明確化もすること。"
            ),
        ),
    )


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_records(project_root: Path = Path(".")) -> list[dict[str, Any]]:
    records = []
    for spec in article_specs(project_root):
        if not spec.article_path.is_file():
            raise FileNotFoundError(f"記事がありません: {spec.article_path}")
        article = spec.article_path.read_text(encoding="utf-8")
        actual_sha = text_sha256(article)
        if actual_sha != spec.article_sha256:
            raise ValueError(
                "記事SHA-256が固定値と一致しません: "
                f"variant={spec.variant}, expected={spec.article_sha256}, actual={actual_sha}"
            )
        records.append({
            "inputs": {
                "article_markdown": article,
                "theme": THEME,
            },
            "expectations": {
                "human_scores": spec.human_scores,
                "publishable": spec.publishable,
                "review_notes": spec.review_notes,
            },
            "tags": {
                "variant": spec.variant,
                "article_path": str(spec.article_path),
                "article_sha256": spec.article_sha256,
                "generation_run_id": spec.generation_run_id,
                "evaluation_run_id": spec.evaluation_run_id,
                "evaluation_trace_id": spec.evaluation_trace_id,
                "generator_prompt_version": spec.generator_prompt_version,
                "review_source": "article-quality-human-review-v2",
            },
        })
    return records


def manifest_sha256(records: list[dict[str, Any]]) -> str:
    normalized = sorted(records, key=lambda item: item["tags"]["variant"])
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def find_dataset(datasets: list[Any], name: str) -> Any | None:
    matches = [dataset for dataset in datasets if dataset.name == name]
    if len(matches) > 1:
        raise ValueError(f"同名Datasetが複数あります: {name}")
    return matches[0] if matches else None


def _normalize_dataset_rows(dataset: Any) -> list[dict[str, Any]]:
    rows = []
    for row in dataset.to_df().to_dict(orient="records"):
        tags = dict(row["tags"] or {})
        tags.pop("mlflow.user", None)
        rows.append({
            "inputs": row["inputs"],
            "expectations": row["expectations"],
            "tags": tags,
        })
    return sorted(rows, key=lambda item: item["tags"]["variant"])


def dataset_matches(dataset: Any, expected_records: list[dict[str, Any]]) -> bool:
    expected = sorted(expected_records, key=lambda item: item["tags"]["variant"])
    return _normalize_dataset_rows(dataset) == expected


def dataset_tags(records: list[dict[str, Any]]) -> dict[str, str]:
    return {
        "workflow_version": WORKFLOW_VERSION,
        "dataset_version": "1",
        "purpose": "human-calibrated-article-quality-evaluation",
        "record_count": str(len(records)),
        "manifest_sha256": manifest_sha256(records),
        "immutable": "true",
    }


def dataset_tags_match(actual: dict[str, Any], expected: dict[str, str]) -> bool:
    """MLflowが追加する管理Tagを許容し、固定Tagだけを厳密比較する。"""

    return all(actual.get(key) == value for key, value in expected.items())


def setup_evaluation_dataset(
    *,
    tracking_uri: str = TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
    project_root: Path = Path("."),
    dry_run: bool = False,
) -> dict[str, Any]:
    import mlflow
    from mlflow.genai.datasets import create_dataset, get_dataset, search_datasets

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name=experiment_name)
    records = build_records(project_root)
    tags = dataset_tags(records)
    base = {
        "workflow_version": WORKFLOW_VERSION,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment.name,
        "experiment_id": experiment.experiment_id,
        "dataset_name": DATASET_NAME,
        "record_count": len(records),
        "manifest_sha256": tags["manifest_sha256"],
        "records": [
            {
                "variant": row["tags"]["variant"],
                "article_path": row["tags"]["article_path"],
                "article_sha256": row["tags"]["article_sha256"],
                "generation_run_id": row["tags"]["generation_run_id"],
                "evaluation_run_id": row["tags"]["evaluation_run_id"],
                "human_scores": row["expectations"]["human_scores"],
                "publishable": row["expectations"]["publishable"],
            }
            for row in records
        ],
    }
    if dry_run:
        return {**base, "mode": "dry-run"}

    datasets = search_datasets(experiment_ids=[experiment.experiment_id], max_results=100)
    dataset = find_dataset(datasets, DATASET_NAME)
    created = dataset is None
    if created:
        dataset = create_dataset(
            name=DATASET_NAME,
            experiment_id=experiment.experiment_id,
            tags=tags,
        )
        dataset.merge_records(records)
        # merge_records後のDigest/ProfileをServerから再取得する。
        dataset = get_dataset(dataset_id=dataset.dataset_id)
    else:
        if not dataset_tags_match(dataset.tags, tags):
            raise ValueError("既存DatasetのTagが固定仕様と一致しません。上書きしません。")
        if not dataset_matches(dataset, records):
            raise ValueError("既存DatasetのRecordが固定仕様と一致しません。上書きしません。")
    return {
        **base,
        "mode": "apply",
        "dataset_id": dataset.dataset_id,
        "digest": dataset.digest,
        "created": created,
        "records_match": dataset_matches(dataset, records),
    }


def validate_evaluation_dataset(
    *,
    tracking_uri: str = TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
    project_root: Path = Path("."),
) -> dict[str, Any]:
    import mlflow
    from mlflow.genai.datasets import search_datasets

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name=experiment_name)
    records = build_records(project_root)
    tags = dataset_tags(records)
    dataset = find_dataset(
        search_datasets(experiment_ids=[experiment.experiment_id], max_results=100),
        DATASET_NAME,
    )
    if dataset is None:
        return {
            "workflow_version": WORKFLOW_VERSION,
            "validation_status": "invalid",
            "dataset_name": DATASET_NAME,
            "reason": "dataset_not_found",
        }
    rows = _normalize_dataset_rows(dataset)
    variants = [row["tags"]["variant"] for row in rows]
    checks = {
        "experiment_linked": experiment.experiment_id in dataset.experiment_ids,
        "tags_match": dataset_tags_match(dataset.tags, tags),
        "record_count_matches": len(rows) == len(records),
        "records_match": rows == sorted(records, key=lambda item: item["tags"]["variant"]),
        "variants_unique": len(variants) == len(set(variants)),
        "human_expectations_complete": all(
            len(row["expectations"]["human_scores"]) == 6
            and isinstance(row["expectations"]["publishable"], bool)
            for row in rows
        ),
    }
    return {
        "workflow_version": WORKFLOW_VERSION,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment.name,
        "experiment_id": experiment.experiment_id,
        "dataset_name": dataset.name,
        "dataset_id": dataset.dataset_id,
        "digest": dataset.digest,
        "record_count": len(rows),
        "variants": variants,
        "checks": checks,
        "validation_status": "validated" if all(checks.values()) else "invalid",
    }
