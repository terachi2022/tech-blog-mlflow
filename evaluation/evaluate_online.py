"""STEP 5-D: Offline品質とOnline観測をMLflow Runで結合する。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tech_blog_mlflow.online_evaluation import (
    metric_map,
    parameter_map,
    prepare_join_from_project,
    write_join_record,
)


DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT = "tech-blog-generation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publication Registryを結合キーに、採用Offline Runと明示指定した"
            "GA4/GSC Observationを1つの監査Runへ保存する。"
        )
    )
    parser.add_argument("--publication-id", required=True)
    parser.add_argument("--ga4-observation-id", required=True)
    parser.add_argument("--gsc-observation-id", required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("datasets/publication_registry.jsonl"),
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path("datasets/online_metrics.jsonl"),
    )
    parser.add_argument(
        "--generation-results-dir",
        type=Path,
        default=Path("generation_results"),
    )
    parser.add_argument(
        "--evaluation-results-dir",
        type=Path,
        default=Path("evaluation_results"),
    )
    parser.add_argument(
        "--ga4-raw-dir",
        type=Path,
        default=Path("datasets/raw/ga4"),
    )
    parser.add_argument(
        "--gsc-raw-dir",
        type=Path,
        default=Path("datasets/raw/gsc"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results"),
    )
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--run-name", default="online-evaluation-step5-d-v1")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Source検証とJoin Previewだけを行い、MLflow/Fileへ書き込まない。",
    )
    return parser.parse_args()


def _root_path(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    created_at = now.isoformat(timespec="seconds")
    record, sources = prepare_join_from_project(
        project_root=root,
        registry_path=_root_path(root, args.registry),
        observations_path=_root_path(root, args.observations),
        publication_id=args.publication_id,
        ga4_observation_id=args.ga4_observation_id,
        gsc_observation_id=args.gsc_observation_id,
        generation_results_dir=_root_path(root, args.generation_results_dir),
        evaluation_results_dir=_root_path(root, args.evaluation_results_dir),
        ga4_raw_dir=_root_path(root, args.ga4_raw_dir),
        gsc_raw_dir=_root_path(root, args.gsc_raw_dir),
        created_at=created_at,
    )

    print("=" * 60)
    print("STEP 5-D Online Evaluation / Offline-Online Join")
    print("=" * 60)
    print(f"Mode             : {'dry-run' if args.dry_run else 'run'}")
    print(f"Join ID          : {record['join_id']}")
    print(f"Publication ID   : {record['publication']['publication_id']}")
    print(f"Generation Run   : {record['offline']['generation']['run_id']}")
    print(f"Offline Eval Run : {record['offline']['evaluation']['run_id']}")
    print(
        "GA4 Observation  :",
        record["online"]["ga4"]["observation"]["observation_id"],
    )
    print(
        "GSC Observation  :",
        record["online"]["gsc"]["observation"]["observation_id"],
    )
    print(f"Data Status      : {record['data_status']}")

    if args.dry_run:
        print()
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return

    import mlflow

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        record["online_evaluation_run_id"] = run_id
        record["mlflow"] = {
            "tracking_uri": args.tracking_uri,
            "experiment_name": args.experiment_name,
            "run_name": args.run_name,
            "run_id": run_id,
        }
        mlflow.log_params(parameter_map(record))
        mlflow.set_tags(
            {
                "stage": "step-5-d",
                "evaluation_type": "offline-online-join",
                "publication_id": record["publication"]["publication_id"],
                "data_status": record["data_status"],
                "automatic_promotion": "false",
            }
        )
        mlflow.log_metrics(metric_map(record))

        timestamp = now.strftime("%Y%m%d_%H%M%S")
        output_path = _root_path(root, args.output_dir) / (
            f"online_evaluation_{timestamp}_{run_id}.json"
        )
        write_join_record(output_path, record)
        mlflow.log_artifact(str(output_path), artifact_path="online_evaluation")
        for name, source_path in sources.items():
            mlflow.log_artifact(
                str(source_path),
                artifact_path=f"online_evaluation/sources/{name}",
            )

    print(f"Online Eval Run  : {run_id}")
    print(f"Result JSON      : {output_path}")
    print(f"Tracking URI     : {args.tracking_uri}")
    print()
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
