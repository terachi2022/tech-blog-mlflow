"""STEP 5-C: Search Console Search Analytics APIを収集する。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tech_blog_mlflow.gsc_collector import (
    COLLECTOR_VERSION,
    DEFAULT_ROW_LIMIT,
    GSC_TIMEZONE,
    build_request_plan,
    execute_search_analytics,
    persist_collection,
    resolve_gsc_date_range,
    select_publication,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "正規Publication Registryの公開記事をpage完全一致で収集し、"
            "記事集計、Query明細、Raw Responseを分離保存する。"
        )
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("datasets/publication_registry.jsonl"),
        help="正規Registryファイル。mistake_*は指定しない。",
    )
    parser.add_argument("--publication-id", default=None)
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD（PT）。")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD（PT）。")
    parser.add_argument(
        "--data-state",
        choices=("all", "final"),
        default="all",
        help="allはFresh Dataを含む。finalは確定済みDataだけ。",
    )
    parser.add_argument("--row-limit", type=int, default=DEFAULT_ROW_LIMIT)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--raw-dir", type=Path, default=Path("datasets/raw/gsc"))
    parser.add_argument(
        "--details-dir",
        type=Path,
        default=Path("datasets/gsc_details"),
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
    publication = select_publication(
        _root_path(root, args.registry),
        args.publication_id,
    )
    start_date, end_date, includes_today = resolve_gsc_date_range(
        publication,
        args.start_date,
        args.end_date,
    )
    plan = build_request_plan(
        publication,
        start_date,
        end_date,
        data_state=args.data_state,
        row_limit=args.row_limit,
    )

    print("=" * 60)
    print("STEP 5-C Search Console Search Analytics Collector")
    print("=" * 60)
    print(f"Mode           : {'dry-run' if args.dry_run else 'collect'}")
    print(f"Collector      : {COLLECTOR_VERSION}")
    print(f"Publication ID : {publication['publication_id']}")
    print(f"GSC Property   : {publication['online']['gsc_site_url']}")
    print(f"Page           : {publication['online']['published_url']}")
    print(f"Date Range PT  : {start_date} .. {end_date}")
    print(f"Data State     : {args.data_state}")
    print(f"Includes Today : {includes_today}")

    if args.dry_run:
        print()
        print(
            json.dumps(
                {
                    "request_plan": plan,
                    "writes": [],
                    "credentials_loaded": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    execution = execute_search_analytics(plan, row_limit=args.row_limit)
    collected_at = datetime.now(ZoneInfo(GSC_TIMEZONE)).isoformat(timespec="seconds")
    result = persist_collection(
        publication=publication,
        start_date=start_date,
        end_date=end_date,
        plan=plan,
        execution=execution,
        raw_dir=_root_path(root, args.raw_dir),
        detail_dir=_root_path(root, args.details_dir),
        observation_path=_root_path(root, args.observations),
        collected_at=collected_at,
        includes_today=includes_today,
        data_state=args.data_state,
    )
    print(f"Status         : {result.status}")
    print(f"Observation ID : {result.observation['observation_id']}")
    print(f"Matched Page   : {execution.selected_page_expression}")
    print(f"Raw Artifact   : {result.raw_path}")
    print(f"Detail Artifact: {result.detail_path}")
    print(f"Observations   : {result.observation_path}")
    print()
    print(json.dumps(result.observation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
