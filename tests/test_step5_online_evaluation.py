from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tech_blog_mlflow.online_evaluation import (
    metric_map,
    parameter_map,
    prepare_join_from_project,
    write_join_record,
)
from tech_blog_mlflow.online_registry import validate_offline_reference


PUBLICATION_ID = "c" * 24
GENERATION_RUN_ID = "d" * 32
EVALUATION_RUN_ID = "e" * 32
GA4_OBSERVATION_ID = "1" * 24
GSC_OBSERVATION_ID = "2" * 24
PUBLISHED_URL = "https://www.lmdev.org/wpblog/test/"


class ProjectFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.registry_path = root / "datasets/publication_registry.jsonl"
        self.observations_path = root / "datasets/online_metrics.jsonl"
        self.generation_dir = root / "generation_results"
        self.evaluation_dir = root / "evaluation_results"
        self.ga4_raw_dir = root / "datasets/raw/ga4"
        self.gsc_raw_dir = root / "datasets/raw/gsc"
        for directory in (
            self.registry_path.parent,
            self.generation_dir,
            self.evaluation_dir,
            self.ga4_raw_dir,
            self.gsc_raw_dir,
            root / "articles",
        ):
            directory.mkdir(parents=True, exist_ok=True)

        article_path = root / "articles/adopted.md"
        article_path.write_text("# Test\n", encoding="utf-8")
        article_sha256 = hashlib.sha256(article_path.read_bytes()).hexdigest()
        self.generation = {
            "run_id": GENERATION_RUN_ID,
            "article_path": "articles/adopted.md",
            "article_sha256": article_sha256,
            "model": "Qwen/Qwen3-8B-MLX-4bit",
            "prompt_version": "article-v3.5.2",
            "generation_config_version": "generation-v3.5.2",
            "all_prechecks_passed": True,
            "metrics": {
                "generation_time_sec": 33.762,
                "output_tokens": 3030,
            },
        }
        self.evaluation = {
            "run_id": EVALUATION_RUN_ID,
            "combined_version": "combined-v2.4.0",
            "article": {
                "path": "articles/adopted.md",
                "sha256": article_sha256,
                "variant": "prompt-v3.5.2",
                "source_run_id": GENERATION_RUN_ID,
                "generator_prompt_version": "article-v3.5.2",
            },
            "judge": {
                "model": "mlx-community/gemma-3-text-27b-it-4bit",
                "prompt_version": "article-judge-v2.4",
            },
            "metrics": {
                "technical_accuracy/mean": 5.0,
                "original_value/mean": 4.0,
            },
        }
        self.generation_path = self.generation_dir / "generation.json"
        self.evaluation_path = self.evaluation_dir / "evaluation.json"
        self._write_json(self.generation_path, self.generation)
        self._write_json(self.evaluation_path, self.evaluation)

        offline = validate_offline_reference(root, self.generation, self.evaluation)
        self.publication = {
            "schema_version": "online-publication-registry-v1.2",
            "publication_id": PUBLICATION_ID,
            "offline": offline,
            "online": {
                "version": "online-target-v1.2",
                "published_url": PUBLISHED_URL,
                "published_at": "2026-08-13T22:43:00+09:00",
                "measurement_timezone": "Asia/Tokyo",
                "ga4_property_id": "549810344",
                "gsc_site_url": "https://www.lmdev.org/wpblog/",
                "cta_tracking": {"enabled": False, "event_name": None},
            },
        }
        self.registry_path.write_text(
            json.dumps(self.publication, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        self.ga4_sha = "a" * 64
        self.gsc_sha = "b" * 64
        self.ga4 = {
            "schema_version": "online-observation-v1",
            "observation_id": GA4_OBSERVATION_ID,
            "publication_id": PUBLICATION_ID,
            "source": "ga4",
            "collector_version": "ga4-data-api-v1.2",
            "collected_at": "2026-08-14T22:14:47+09:00",
            "date_range": {"start_date": "2026-08-13", "end_date": "2026-08-14"},
            "dimensions": {
                "page_location": PUBLISHED_URL,
                "ga4_property_id": "549810344",
                "collection_window_state": "includes_today",
            },
            "metrics": {
                "screen_page_views": 2,
                "active_users": 2,
                "user_engagement_duration_sec": 0.0,
                "average_engagement_time_sec": 0.0,
                "cta_tracking_enabled": False,
                "cta_event_count": None,
                "cta_rate": None,
                "cta_rate_denominator": None,
            },
            "raw_response_sha256": self.ga4_sha,
            "is_partial": True,
            "notes": [],
        }
        self.gsc = {
            "schema_version": "online-observation-v1",
            "observation_id": GSC_OBSERVATION_ID,
            "publication_id": PUBLICATION_ID,
            "source": "gsc",
            "collector_version": "gsc-search-analytics-v1.1",
            "collected_at": "2026-08-14T07:36:31-07:00",
            "date_range": {"start_date": "2026-08-13", "end_date": "2026-08-14"},
            "dimensions": {
                "page": PUBLISHED_URL,
                "matched_page_expression": PUBLISHED_URL,
                "gsc_site_url": "https://www.lmdev.org/wpblog/",
                "search_type": "web",
                "data_state": "all",
                "date_timezone": "America/Los_Angeles",
                "detail_artifact_sha256": "f" * 64,
            },
            "metrics": {
                "clicks": 0,
                "impressions": 0,
                "ctr": None,
                "position": None,
            },
            "raw_response_sha256": self.gsc_sha,
            "is_partial": True,
            "notes": [],
        }
        self.write_observations()
        self.ga4_raw_path = self._raw_path(self.ga4_raw_dir, self.ga4)
        self.gsc_raw_path = self._raw_path(self.gsc_raw_dir, self.gsc)
        self._write_json(
            self.ga4_raw_path,
            {"raw_response_sha256": self.ga4_sha, "payload": {}},
        )
        self._write_json(
            self.gsc_raw_path,
            {"raw_response_sha256": self.gsc_sha, "payload": {}},
        )

    @staticmethod
    def _write_json(path: Path, value: dict) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _raw_path(directory: Path, observation: dict) -> Path:
        date_range = observation["date_range"]
        sha = observation["raw_response_sha256"]
        return directory / (
            f"{observation['source']}_{PUBLICATION_ID}_"
            f"{date_range['start_date']}_{date_range['end_date']}_{sha[:12]}.json"
        )

    def write_observations(self, extra: list[dict] | None = None) -> None:
        values = [self.ga4, self.gsc, *(extra or [])]
        self.observations_path.write_text(
            "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
            encoding="utf-8",
        )

    def prepare(self) -> tuple[dict, dict[str, Path]]:
        return prepare_join_from_project(
            project_root=self.root,
            registry_path=self.registry_path,
            observations_path=self.observations_path,
            publication_id=PUBLICATION_ID,
            ga4_observation_id=GA4_OBSERVATION_ID,
            gsc_observation_id=GSC_OBSERVATION_ID,
            generation_results_dir=self.generation_dir,
            evaluation_results_dir=self.evaluation_dir,
            ga4_raw_dir=self.ga4_raw_dir,
            gsc_raw_dir=self.gsc_raw_dir,
            created_at="2026-08-14T23:00:00+09:00",
        )


class OnlineEvaluationJoinTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = ProjectFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_join_is_provisional_and_keeps_namespaces_separate(self) -> None:
        record, sources = self.fixture.prepare()
        self.assertEqual(record["data_status"], "provisional")
        self.assertEqual(len(record["partial_reasons"]), 3)
        self.assertFalse(record["decision_policy"]["automatic_promotion"])
        self.assertFalse(record["decision_policy"]["offline_scores_overwritten"])
        self.assertEqual(sources["ga4_raw"], self.fixture.ga4_raw_path)

        metrics = metric_map(record)
        self.assertEqual(metrics["offline/evaluation/technical_accuracy/mean"], 5.0)
        self.assertEqual(metrics["online/ga4/screen_page_views"], 2.0)
        self.assertEqual(metrics["online/gsc/impressions"], 0.0)
        self.assertNotIn("online/gsc/ctr", metrics)
        self.assertNotIn("online/ga4/cta_event_count", metrics)
        self.assertIn("online/gsc/position", record["online"]["nullable_metrics"])

        params = parameter_map(record)
        self.assertEqual(params["evaluation_stage"], "step-5-d")
        self.assertEqual(params["gsc_data_state"], "all")
        self.assertTrue(params["ga4_is_partial"])

    def test_final_requires_both_nonpartial_and_gsc_final(self) -> None:
        self.fixture.ga4["is_partial"] = False
        self.fixture.ga4["dimensions"]["collection_window_state"] = "closed_period"
        self.fixture.gsc["is_partial"] = False
        self.fixture.gsc["dimensions"]["data_state"] = "final"
        self.fixture.write_observations()
        record, _ = self.fixture.prepare()
        self.assertEqual(record["data_status"], "final")
        self.assertEqual(record["partial_reasons"], [])

    def test_legacy_collector_is_rejected(self) -> None:
        self.fixture.gsc["collector_version"] = "gsc-search-analytics-v1"
        self.fixture.write_observations()
        with self.assertRaisesRegex(ValueError, "採用Version"):
            self.fixture.prepare()

    def test_ga4_v1_1_is_rejected_after_window_identity_fix(self) -> None:
        self.fixture.ga4["collector_version"] = "ga4-data-api-v1.1"
        self.fixture.write_observations()
        with self.assertRaisesRegex(ValueError, "採用Version"):
            self.fixture.prepare()

    def test_ga4_includes_today_cannot_be_nonpartial(self) -> None:
        self.fixture.ga4["is_partial"] = False
        self.fixture.write_observations()
        with self.assertRaisesRegex(ValueError, "includes_today"):
            self.fixture.prepare()

    def test_different_date_labels_are_rejected(self) -> None:
        self.fixture.gsc["date_range"]["end_date"] = "2026-08-13"
        self.fixture.write_observations()
        with self.assertRaisesRegex(ValueError, "Date Label"):
            self.fixture.prepare()

    def test_duplicate_observation_id_is_rejected(self) -> None:
        self.fixture.write_observations(extra=[dict(self.fixture.ga4)])
        with self.assertRaisesRegex(ValueError, "重複"):
            self.fixture.prepare()

    def test_raw_artifact_hash_mismatch_is_rejected(self) -> None:
        self.fixture._write_json(
            self.fixture.ga4_raw_path,
            {"raw_response_sha256": "0" * 64, "payload": {}},
        )
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.fixture.prepare()

    def test_atomic_write_round_trip(self) -> None:
        record, _ = self.fixture.prepare()
        output = self.fixture.root / "evaluation_results/join.json"
        write_join_record(output, record)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), record)


if __name__ == "__main__":
    unittest.main()
