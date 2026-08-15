from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch
from urllib.parse import unquote

from tech_blog_mlflow.ga4_collector import (
    COLLECTOR_VERSION,
    build_observation,
    build_request_specs,
    execute_reports,
    page_location_aliases,
    persist_collection,
    resolve_date_range,
    select_publication,
    sha256_json,
)


PUBLICATION_ID = "65c0b6e43f5dedf39e0011e8"
PUBLISHED_URL = (
    "https://www.lmdev.org/wpblog/"
    "apple-m5-max%e3%81%a7mlflow%e5%85%ac%e5%bc%8f/"
)


def publication(*, cta_enabled: bool = False) -> dict:
    return {
        "schema_version": "online-publication-registry-v1.2",
        "publication_id": PUBLICATION_ID,
        "offline": {"article_sha256": "a" * 64},
        "online": {
            "version": "online-target-v1.2",
            "published_url": PUBLISHED_URL,
            "published_at": "2026-08-13T22:43:00+09:00",
            "measurement_timezone": "Asia/Tokyo",
            "ga4_property_id": "549810344",
            "gsc_site_url": "https://www.lmdev.org/wpblog/",
            "cta_tracking": {
                "enabled": cta_enabled,
                "event_name": "article_cta_click" if cta_enabled else None,
            },
        },
    }


def response(
    metrics: tuple[str, ...],
    values: tuple[str, ...] | None,
    *,
    data_loss: bool = False,
) -> dict:
    rows = []
    if values is not None:
        rows = [
            {
                "dimensionValues": [{"value": PUBLISHED_URL}],
                "metricValues": [{"value": value} for value in values],
            }
        ]
    return {
        "dimensionHeaders": [{"name": "pageLocation"}],
        "metricHeaders": [{"name": name} for name in metrics],
        "rows": rows,
        "rowCount": len(rows),
        "metadata": {
            "currencyCode": "JPY",
            "timeZone": "Asia/Tokyo",
            "dataLossFromOtherRow": data_loss,
        },
    }


class RegistrySelectionTest(unittest.TestCase):
    def test_only_explicit_registry_file_is_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            correct = root / "publication_registry.jsonl"
            mistake = root / "publication_registry.mistake_20260814.jsonl"
            correct.write_text(
                json.dumps(publication(), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            wrong = publication()
            wrong["publication_id"] = "b" * 24
            mistake.write_text(json.dumps(wrong) + "\n", encoding="utf-8")

            selected = select_publication(correct)

            self.assertEqual(selected["publication_id"], PUBLICATION_ID)

    def test_unknown_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.jsonl"
            item = publication()
            item["schema_version"] = "old-schema"
            path.write_text(json.dumps(item) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Schema"):
                select_publication(path)


class RequestTest(unittest.TestCase):
    def test_main_request_uses_registry_property_and_exact_page_location(self) -> None:
        specs = build_request_specs(publication(), "2026-08-13", "2026-08-14")
        main = specs["main"]
        self.assertEqual(main["property"], "properties/549810344")
        self.assertEqual(
            main["dimension_filter"]["filter"]["field_name"],
            "pageLocation",
        )
        in_list_filter = main["dimension_filter"]["filter"]["in_list_filter"]
        self.assertEqual(
            in_list_filter["values"],
            [PUBLISHED_URL, unquote(PUBLISHED_URL)],
        )
        self.assertFalse(in_list_filter["case_sensitive"])
        self.assertIs(specs["cta"], None)

    def test_page_location_aliases_deduplicate_plain_ascii_url(self) -> None:
        self.assertEqual(
            page_location_aliases("https://example.com/plain/"),
            ("https://example.com/plain/",),
        )

    def test_page_location_aliases_include_unicode_iri(self) -> None:
        aliases = page_location_aliases(PUBLISHED_URL)
        self.assertEqual(aliases[0], PUBLISHED_URL)
        self.assertIn("公式", aliases[1])

    def test_cta_request_is_only_created_when_tracking_is_enabled(self) -> None:
        specs = build_request_specs(
            publication(cta_enabled=True),
            "2026-08-13",
            "2026-08-14",
        )
        cta = specs["cta"]
        expressions = cta["dimension_filter"]["and_group"]["expressions"]
        self.assertEqual(len(expressions), 2)
        self.assertEqual(
            expressions[0]["filter"]["in_list_filter"]["values"],
            [PUBLISHED_URL, unquote(PUBLISHED_URL)],
        )
        self.assertEqual(
            expressions[1]["filter"]["string_filter"]["value"],
            "article_cta_click",
        )

    def test_date_range_defaults_to_publication_day_through_today(self) -> None:
        actual = resolve_date_range(
            publication(),
            None,
            None,
            today=date(2026, 8, 14),
        )
        self.assertEqual(actual, ("2026-08-13", "2026-08-14", True))

    def test_prepublication_and_future_dates_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "公開日以降"):
            resolve_date_range(
                publication(),
                "2026-08-12",
                "2026-08-14",
                today=date(2026, 8, 14),
            )
        with self.assertRaisesRegex(ValueError, "未来日"):
            resolve_date_range(
                publication(),
                "2026-08-13",
                "2026-08-15",
                today=date(2026, 8, 14),
            )


class ObservationTest(unittest.TestCase):
    def test_disabled_cta_is_null_and_average_is_derived(self) -> None:
        responses = {
            "main": response(
                ("screenPageViews", "activeUsers", "userEngagementDuration"),
                ("10", "4", "50.0"),
            ),
            "cta": None,
        }
        observation = build_observation(
            publication(),
            "2026-08-13",
            "2026-08-14",
            responses,
            sha256_json(responses),
            "2026-08-14T20:00:00+09:00",
            includes_today=False,
        )
        metrics = observation["metrics"]
        self.assertEqual(metrics["screen_page_views"], 10)
        self.assertEqual(metrics["active_users"], 4)
        self.assertEqual(metrics["average_engagement_time_sec"], 12.5)
        self.assertIs(metrics["cta_event_count"], None)
        self.assertIs(metrics["cta_rate"], None)
        self.assertIs(metrics["cta_rate_denominator"], None)

    def test_empty_successful_response_is_measured_zero_not_missing(self) -> None:
        responses = {
            "main": response(
                ("screenPageViews", "activeUsers", "userEngagementDuration"),
                None,
            ),
            "cta": None,
        }
        observation = build_observation(
            publication(),
            "2026-08-13",
            "2026-08-14",
            responses,
            sha256_json(responses),
            "2026-08-14T20:00:00+09:00",
            includes_today=False,
        )
        metrics = observation["metrics"]
        self.assertEqual(metrics["screen_page_views"], 0)
        self.assertEqual(metrics["active_users"], 0)
        self.assertIs(metrics["average_engagement_time_sec"], None)

    def test_enabled_cta_rate_uses_page_views(self) -> None:
        responses = {
            "main": response(
                ("screenPageViews", "activeUsers", "userEngagementDuration"),
                ("20", "7", "80"),
            ),
            "cta": response(("eventCount",), ("3",)),
        }
        observation = build_observation(
            publication(cta_enabled=True),
            "2026-08-13",
            "2026-08-14",
            responses,
            sha256_json(responses),
            "2026-08-14T20:00:00+09:00",
            includes_today=False,
        )
        self.assertEqual(observation["metrics"]["cta_event_count"], 3)
        self.assertEqual(observation["metrics"]["cta_rate"], 0.15)
        self.assertEqual(
            observation["metrics"]["cta_rate_denominator"],
            "screen_page_views",
        )

    def test_today_and_data_loss_mark_observation_partial(self) -> None:
        responses = {
            "main": response(
                ("screenPageViews", "activeUsers", "userEngagementDuration"),
                ("1", "1", "2"),
                data_loss=True,
            ),
            "cta": None,
        }
        observation = build_observation(
            publication(),
            "2026-08-13",
            "2026-08-14",
            responses,
            sha256_json(responses),
            "2026-08-14T20:00:00+09:00",
            includes_today=True,
        )
        self.assertTrue(observation["is_partial"])
        self.assertGreaterEqual(len(observation["notes"]), 3)

    def test_persistence_is_idempotent_for_same_request_and_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = publication()
            specs = build_request_specs(item, "2026-08-13", "2026-08-14")
            responses = {
                "main": response(
                    ("screenPageViews", "activeUsers", "userEngagementDuration"),
                    ("2", "1", "4"),
                ),
                "cta": None,
            }
            first = persist_collection(
                publication=item,
                start_date="2026-08-13",
                end_date="2026-08-14",
                specs=specs,
                responses=responses,
                raw_dir=root / "raw",
                observation_path=root / "online_metrics.jsonl",
                collected_at="2026-08-14T20:00:00+09:00",
                includes_today=False,
            )
            second = persist_collection(
                publication=item,
                start_date="2026-08-13",
                end_date="2026-08-14",
                specs=specs,
                responses=responses,
                raw_dir=root / "raw",
                observation_path=root / "online_metrics.jsonl",
                collected_at="2026-08-14T21:00:00+09:00",
                includes_today=False,
            )
            self.assertEqual(first.status, "created")
            self.assertEqual(second.status, "unchanged")
            self.assertEqual(first.observation, second.observation)
            self.assertEqual(len(list((root / "raw").glob("*.json"))), 1)
            self.assertEqual(
                len((root / "online_metrics.jsonl").read_text().splitlines()),
                1,
            )

    def test_partial_to_closed_period_creates_distinct_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = publication()
            specs = build_request_specs(item, "2026-08-13", "2026-08-14")
            responses = {
                "main": response(
                    ("screenPageViews", "activeUsers", "userEngagementDuration"),
                    ("2", "2", "0"),
                ),
                "cta": None,
            }
            partial = persist_collection(
                publication=item,
                start_date="2026-08-13",
                end_date="2026-08-14",
                specs=specs,
                responses=responses,
                raw_dir=root / "raw",
                observation_path=root / "online_metrics.jsonl",
                collected_at="2026-08-14T22:14:47+09:00",
                includes_today=True,
            )
            final = persist_collection(
                publication=item,
                start_date="2026-08-13",
                end_date="2026-08-14",
                specs=specs,
                responses=responses,
                raw_dir=root / "raw",
                observation_path=root / "online_metrics.jsonl",
                collected_at="2026-08-15T00:11:16+09:00",
                includes_today=False,
            )

            self.assertEqual(partial.status, "created")
            self.assertEqual(final.status, "created")
            self.assertNotEqual(
                partial.observation["observation_id"],
                final.observation["observation_id"],
            )
            self.assertTrue(partial.observation["is_partial"])
            self.assertFalse(final.observation["is_partial"])
            self.assertEqual(
                partial.observation["dimensions"]["collection_window_state"],
                "includes_today",
            )
            self.assertEqual(
                final.observation["dimensions"]["collection_window_state"],
                "closed_period",
            )
            self.assertEqual(final.observation["collector_version"], COLLECTOR_VERSION)
            raw_paths = sorted((root / "raw").glob("*.json"))
            self.assertEqual(len(raw_paths), 2)
            raw_payloads = [
                json.loads(path.read_text(encoding="utf-8"))["payload"]
                for path in raw_paths
            ]
            self.assertEqual(
                {
                    payload["collection_context"]["collection_window_state"]
                    for payload in raw_payloads
                },
                {"includes_today", "closed_period"},
            )
            self.assertTrue(
                all(
                    payload["collector_version"] == COLLECTOR_VERSION
                    for payload in raw_payloads
                )
            )
            self.assertEqual(
                len((root / "online_metrics.jsonl").read_text().splitlines()),
                2,
            )


class ClientBoundaryTest(unittest.TestCase):
    def test_request_execution_boundary_can_be_tested_without_network(self) -> None:
        class FakeRequest:
            def __init__(self, **values):
                self.values = values

        class FakeClient:
            def run_report(self, *, request):
                self.request = request
                return response(
                    ("screenPageViews", "activeUsers", "userEngagementDuration"),
                    ("1", "1", "3"),
                )

        specs = build_request_specs(publication(), "2026-08-13", "2026-08-14")
        with patch(
            "tech_blog_mlflow.ga4_collector._import_ga4_client",
            return_value=(object, FakeRequest),
        ):
            actual = execute_reports(specs, client_factory=FakeClient)
        self.assertEqual(actual["main"]["rowCount"], 1)
        self.assertIs(actual["cta"], None)


if __name__ == "__main__":
    unittest.main()
