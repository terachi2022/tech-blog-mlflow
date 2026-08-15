"""STEP 5-A: 公開記事とOffline RunをRegistryへ登録する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tech_blog_mlflow.online_registry import (
    append_publication_record,
    build_publication_record,
    load_json_object,
)


ADOPTED_GENERATION_RUN_ID = "b5c925c2322b4e30b04f07e24d160a04"
ADOPTED_EVALUATION_RUN_ID = "fdf0c239445f44a0999a6b1fe7a419b6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "公開記事URLを、採用済みGeneration/Evaluation Runと対応付ける。"
            "Online指標そのものは記録しない。"
        )
    )
    parser.add_argument("--published-url", required=True)
    parser.add_argument("--published-at", required=True)
    parser.add_argument("--ga4-property-id", required=True)
    parser.add_argument("--gsc-site-url", required=True)
    cta_group = parser.add_mutually_exclusive_group(required=True)
    cta_group.add_argument("--cta-event-name")
    cta_group.add_argument("--cta-not-implemented", action="store_true")
    parser.add_argument("--measurement-timezone", default="Asia/Tokyo")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--generation-json", type=Path, default=None)
    parser.add_argument("--evaluation-json", type=Path, default=None)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("datasets/publication_registry.jsonl"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def find_single(root: Path, directory: str, pattern: str, label: str) -> Path:
    matches = sorted((root / directory).glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{label}を一意に特定できません: pattern={pattern}, matches={len(matches)}"
        )
    return matches[0]


def resolve_metadata_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    root = args.project_root.resolve()
    generation_json = args.generation_json or find_single(
        root,
        "generation_results",
        f"*{ADOPTED_GENERATION_RUN_ID}.json",
        "採用Generation JSON",
    )
    evaluation_json = args.evaluation_json or find_single(
        root,
        "evaluation_results",
        f"*{ADOPTED_EVALUATION_RUN_ID}.json",
        "採用Evaluation JSON",
    )
    if not generation_json.is_absolute():
        generation_json = root / generation_json
    if not evaluation_json.is_absolute():
        evaluation_json = root / evaluation_json
    return generation_json, evaluation_json


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    generation_json, evaluation_json = resolve_metadata_paths(args)
    record = build_publication_record(
        project_root=root,
        generation=load_json_object(generation_json),
        evaluation=load_json_object(evaluation_json),
        published_url=args.published_url,
        published_at=args.published_at,
        ga4_property_id=args.ga4_property_id,
        gsc_site_url=args.gsc_site_url,
        cta_event_name=(
            None if args.cta_not_implemented else args.cta_event_name
        ),
        measurement_timezone=args.measurement_timezone,
    )

    if args.dry_run:
        status = "dry-run"
    else:
        registry = args.registry if args.registry.is_absolute() else root / args.registry
        status = append_publication_record(registry, record)

    print("=" * 60)
    print("STEP 5-A Publication Registry")
    print("=" * 60)
    print(f"Status         : {status}")
    print(f"Publication ID : {record['publication_id']}")
    print(f"Article SHA-256: {record['offline']['article_sha256']}")
    print(f"Generation Run : {record['offline']['generation_run_id']}")
    print(f"Evaluation Run : {record['offline']['evaluation_run_id']}")
    print(f"Published URL  : {record['online']['published_url']}")
    print()
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
