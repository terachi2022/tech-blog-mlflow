"""STEP 5-A Publication Registryの回帰テスト。"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from tech_blog_mlflow.online_registry import (  # noqa: E402
    append_publication_record,
    build_publication_record,
    normalize_gsc_site_url,
    normalize_measurement_timezone,
    normalize_public_article_url,
    sha256_file,
)


GENERATION_RUN_ID = "b5c925c2322b4e30b04f07e24d160a04"
EVALUATION_RUN_ID = "fdf0c239445f44a0999a6b1fe7a419b6"


def adopted_json(directory: str, run_id: str) -> Path:
    matches = list((PROJECT_ROOT / directory).glob(f"*{run_id}.json"))
    if len(matches) != 1:
        raise AssertionError(f"Fixtureを一意に特定できません: {matches}")
    return matches[0]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class URLValidationTest(unittest.TestCase):
    def test_public_https_url_is_retained(self) -> None:
        self.assertEqual(
            normalize_public_article_url("https://www.lmdev.org/wpblog/mlflow/"),
            "https://www.lmdev.org/wpblog/mlflow/",
        )

    def test_unicode_and_percent_encoded_paths_have_same_canonical_url(self) -> None:
        unicode_url = (
            "https://www.lmdev.org/wpblog/"
            "apple-m5-maxでmlflow公式hyperparameter-tuningを実践する-"
            "optunaでハイパーパラメ/"
        )
        encoded_url = (
            "https://www.lmdev.org/wpblog/"
            "apple-m5-max%e3%81%a7mlflow%e5%85%ac%e5%bc%8f"
            "hyperparameter-tuning%e3%82%92%e5%ae%9f%e8%b7%b5"
            "%e3%81%99%e3%82%8b-optuna%e3%81%a7%e3%83%8f"
            "%e3%82%a4%e3%83%91%e3%83%bc%e3%83%91%e3%83%a9"
            "%e3%83%a1/"
        )
        self.assertEqual(
            normalize_public_article_url(unicode_url),
            normalize_public_article_url(encoded_url),
        )

    def test_local_and_query_urls_are_rejected(self) -> None:
        for value in (
            "http://www.lmdev.org/wpblog/mlflow/",
            "https://127.0.0.1/article/",
            "https://localhost/article/",
            "https://www.lmdev.org/article/?preview=1",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_public_article_url(value)

    def test_search_console_properties_are_normalized(self) -> None:
        self.assertEqual(
            normalize_gsc_site_url("sc-domain:LMDEV.ORG"),
            "sc-domain:lmdev.org",
        )
        self.assertEqual(
            normalize_gsc_site_url("https://www.lmdev.org/wpblog"),
            "https://www.lmdev.org/wpblog/",
        )

    def test_measurement_timezone_must_be_iana_name(self) -> None:
        self.assertEqual(normalize_measurement_timezone("Asia/Tokyo"), "Asia/Tokyo")
        with self.assertRaises(ValueError):
            normalize_measurement_timezone("JST")


class OfflineLinkageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generation = load(
            adopted_json("generation_results", GENERATION_RUN_ID)
        )
        cls.evaluation = load(
            adopted_json("evaluation_results", EVALUATION_RUN_ID)
        )

    def build_record(self) -> dict:
        return build_publication_record(
            project_root=PROJECT_ROOT,
            generation=copy.deepcopy(self.generation),
            evaluation=copy.deepcopy(self.evaluation),
            published_url="https://www.lmdev.org/wpblog/mlflow-experiment/",
            published_at="2026-08-14T18:00:00+09:00",
            ga4_property_id="properties/123456789",
            gsc_site_url="sc-domain:lmdev.org",
            cta_event_name="article_cta_click",
            measurement_timezone="Asia/Tokyo",
        )

    def build_record_without_cta(self) -> dict:
        return build_publication_record(
            project_root=PROJECT_ROOT,
            generation=copy.deepcopy(self.generation),
            evaluation=copy.deepcopy(self.evaluation),
            published_url="https://www.lmdev.org/wpblog/mlflow-experiment/",
            published_at="2026-08-13T22:43:00+09:00",
            ga4_property_id="549810344",
            gsc_site_url="https://www.lmdev.org/wpblog/",
            cta_event_name=None,
            measurement_timezone="Asia/Tokyo",
        )

    def test_adopted_article_and_runs_are_linked(self) -> None:
        record = self.build_record()
        offline = record["offline"]
        self.assertEqual(offline["generation_run_id"], GENERATION_RUN_ID)
        self.assertEqual(offline["evaluation_run_id"], EVALUATION_RUN_ID)
        self.assertEqual(
            offline["article_sha256"],
            sha256_file(PROJECT_ROOT / offline["article_path"]),
        )
        self.assertFalse(offline["skills_enabled"])

    def test_unimplemented_cta_is_explicitly_null(self) -> None:
        record = self.build_record_without_cta()
        self.assertEqual(
            record["online"]["cta_tracking"],
            {"enabled": False, "event_name": None},
        )

    def test_confirmed_unicode_url_maps_to_committed_publication_id(self) -> None:
        record = build_publication_record(
            project_root=PROJECT_ROOT,
            generation=copy.deepcopy(self.generation),
            evaluation=copy.deepcopy(self.evaluation),
            published_url=(
                "https://www.lmdev.org/wpblog/"
                "apple-m5-maxでmlflow公式hyperparameter-tuningを実践する-"
                "optunaでハイパーパラメ/"
            ),
            published_at="2026-08-13T22:43:00+09:00",
            ga4_property_id="549810344",
            gsc_site_url="https://www.lmdev.org/wpblog/",
            cta_event_name=None,
            measurement_timezone="Asia/Tokyo",
        )
        self.assertEqual(record["publication_id"], "65c0b6e43f5dedf39e0011e8")
        self.assertNotRegex(record["online"]["published_url"], r"[^\x00-\x7f]")

    def test_tampered_article_sha_is_rejected(self) -> None:
        generation = copy.deepcopy(self.generation)
        generation["article_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            build_publication_record(
                project_root=PROJECT_ROOT,
                generation=generation,
                evaluation=copy.deepcopy(self.evaluation),
                published_url="https://www.lmdev.org/wpblog/mlflow-experiment/",
                published_at="2026-08-14T18:00:00+09:00",
                ga4_property_id="123456789",
                gsc_site_url="sc-domain:lmdev.org",
                cta_event_name="article_cta_click",
                measurement_timezone="Asia/Tokyo",
            )

    def test_mismatched_evaluation_run_is_rejected(self) -> None:
        evaluation = copy.deepcopy(self.evaluation)
        evaluation["article"]["source_run_id"] = "0" * 32
        with self.assertRaises(ValueError):
            build_publication_record(
                project_root=PROJECT_ROOT,
                generation=copy.deepcopy(self.generation),
                evaluation=evaluation,
                published_url="https://www.lmdev.org/wpblog/mlflow-experiment/",
                published_at="2026-08-14T18:00:00+09:00",
                ga4_property_id="123456789",
                gsc_site_url="sc-domain:lmdev.org",
                cta_event_name="article_cta_click",
                measurement_timezone="Asia/Tokyo",
            )

    def test_append_is_idempotent_and_conflicts_fail(self) -> None:
        record = self.build_record()
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory) / "publication_registry.jsonl"
            self.assertEqual(append_publication_record(registry, record), "created")
            self.assertEqual(append_publication_record(registry, record), "unchanged")
            self.assertEqual(len(registry.read_text(encoding="utf-8").splitlines()), 1)

            conflict = copy.deepcopy(record)
            conflict["publication_id"] = "f" * 24
            with self.assertRaises(ValueError):
                append_publication_record(registry, conflict)


class SchemaTest(unittest.TestCase):
    def test_registry_and_observation_schemas_are_separate(self) -> None:
        registry_schema = load(
            PROJECT_ROOT / "datasets" / "publication_registry.schema.json"
        )
        observation_schema = load(
            PROJECT_ROOT / "datasets" / "online_metrics.schema.json"
        )
        self.assertEqual(
            registry_schema["properties"]["schema_version"]["const"],
            "online-publication-registry-v1.2",
        )
        self.assertEqual(
            observation_schema["properties"]["schema_version"]["const"],
            "online-observation-v1",
        )
        self.assertNotIn("metrics", registry_schema["properties"])
        self.assertNotIn("offline", observation_schema["properties"])

    def test_committed_publication_record_uses_confirmed_values(self) -> None:
        records = [
            json.loads(line)
            for line in (
                PROJECT_ROOT / "datasets" / "publication_registry.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(
            record["schema_version"],
            "online-publication-registry-v1.2",
        )
        self.assertEqual(record["publication_id"], "65c0b6e43f5dedf39e0011e8")
        self.assertEqual(record["online"]["published_at"], "2026-08-13T22:43:00+09:00")
        self.assertEqual(record["online"]["ga4_property_id"], "549810344")
        self.assertEqual(
            record["online"]["gsc_site_url"],
            "https://www.lmdev.org/wpblog/",
        )
        self.assertEqual(
            record["online"]["cta_tracking"],
            {"enabled": False, "event_name": None},
        )


if __name__ == "__main__":
    unittest.main()
