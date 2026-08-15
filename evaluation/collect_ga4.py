"""STEP 5-B: Publication Registryを入力にGA4 Data APIを収集する。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tech_blog_mlflow.ga4_collector import (
    COLLECTOR_VERSION,
    build_request_specs,
    execute_reports,
    persist_collection,
    request_spec_for_display,
    resolve_date_range,
    select_publication,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "正規Publication Registryの公開記事をpageLocation完全一致で収集し、"
            "Raw ResponseとOnline Observationを分離保存する。"
        )
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("datasets/publication_registry.jsonl"),
        help="正規Registryファイル。mistake_*は指定しない。",
    )
    parser.add_argument("--publication-id", default=None)
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD。既定は公開日。")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD。既定は収集日。")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("datasets/raw/ga4"),
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path("datasets/online_metrics.jsonl"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="認証/API呼出し/ファイル書込みを行わずRequestだけ表示する。",
    )
    return parser.parse_args()


def _root_path(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    registry_path = _root_path(root, args.registry)
    publication = select_publication(registry_path, args.publication_id)
    start_date, end_date, includes_today = resolve_date_range(
        publication,
        args.start_date,
        args.end_date,
    )
    specs = build_request_specs(publication, start_date, end_date)

    print("=" * 60)
    print("STEP 5-B GA4 Data API Collector")
    print("=" * 60)
    print(f"Mode           : {'dry-run' if args.dry_run else 'collect'}")
    print(f"Collector      : {COLLECTOR_VERSION}")
    print(f"Publication ID : {publication['publication_id']}")
    print(f"GA4 Property   : {publication['online']['ga4_property_id']}")
    print(f"Page Location  : {publication['online']['published_url']}")
    print(f"Date Range     : {start_date} .. {end_date}")
    print(f"Includes Today : {includes_today}")

    if args.dry_run:
        print()
        print(json.dumps(
            {
                "main": request_spec_for_display(specs["main"]),
                "cta": (
                    request_spec_for_display(specs["cta"])
                    if specs["cta"] is not None
                    else None
                ),
                "writes": [],
                "credentials_loaded": False,
            },
            ensure_ascii=False,
            indent=2,
        ))
        return

    responses = execute_reports(specs)
    timezone = ZoneInfo(str(publication["online"]["measurement_timezone"]))
    collected_at = datetime.now(timezone).isoformat(timespec="seconds")
    result = persist_collection(
        publication=publication,
        start_date=start_date,
        end_date=end_date,
        specs=specs,
        responses=responses,
        raw_dir=_root_path(root, args.raw_dir),
        observation_path=_root_path(root, args.observations),
        collected_at=collected_at,
        includes_today=includes_today,
    )
    print(f"Status         : {result.status}")
    print(f"Observation ID : {result.observation['observation_id']}")
    print(f"Raw Artifact   : {result.raw_path}")
    print(f"Observations   : {result.observation_path}")
    print()
    print(json.dumps(result.observation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
