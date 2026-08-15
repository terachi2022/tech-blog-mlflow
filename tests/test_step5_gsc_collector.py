from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from urllib.parse import unquote

from tech_blog_mlflow.gsc_collector import (
    COLLECTOR_VERSION,
    GSCExecutionResult,
    aggregate_metrics,
    build_detail_records,
    build_observation,
    build_request_plan,
    execute_search_analytics,
    persist_collection,
    resolve_gsc_date_range,
)


PUBLICATION_ID = "65c0b6e43f5dedf39e0011e8"
PUBLISHED_URL = (
    "https://www.lmdev.org/wpblog/"
    "apple-m5-max%e3%81%a7mlflow%e5%85%ac%e5%bc%8f/"
)


def publication() -> dict:
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
            "cta_tracking": {"enabled": False, "event_name": None},
        },
    }


def metric_row(
    clicks: float = 2,
    impressions: float = 10,
    ctr: float = 0.2,
    position: float = 3.5,
    *,
    keys: list[str] | None = None,
) -> dict:
    row = {
        "clicks": clicks,
        "impressions": impressions,
        "ctr": ctr,
        "position": position,
    }
    if keys is not None:
        row["keys"] = keys
    return row


class DateAndRequestTest(unittest.TestCase):
    def test_gsc_dates_use_pacific_time(self) -> None:
        actual = resolve_gsc_date_range(
            publication(),
            None,
            None,
            today=date(2026, 8, 14),
        )
        self.assertEqual(actual, ("2026-08-13", "2026-08-14", True))

    def test_prepublication_and_future_dates_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "PT基準の公開日以降"):
            resolve_gsc_date_range(
                publication(),
                "2026-08-12",
                "2026-08-14",
                today=date(2026, 8, 14),
            )
        with self.assertRaisesRegex(ValueError, "PT基準の未来日"):
            resolve_gsc_date_range(
                publication(),
                "2026-08-13",
                "2026-08-15",
                today=date(2026, 8, 14),
            )

    def test_plan_uses_registry_property_exact_page_aliases_and_pt(self) -> None:
        plan = build_request_plan(
            publication(),
            "2026-08-13",
            "2026-08-14",
            row_limit=100,
        )
        self.assertEqual(plan["site_url"], "https://www.lmdev.org/wpblog/")
        self.assertEqual(
            plan["page_expressions"],
            [PUBLISHED_URL, unquote(PUBLISHED_URL)],
        )
        request = plan["aggregate_requests"][0]
        page_filter = request["dimensionFilterGroups"][0]["filters"][0]
        self.assertEqual(page_filter["dimension"], "page")
        self.assertEqual(page_filter["operator"], "equals")
        self.assertEqual(request["aggregationType"], "auto")
        self.assertEqual(request["dataState"], "all")
        self.assertEqual(plan["date_timezone"], "America/Los_Angeles")
        self.assertEqual(
            plan["detail_request_template"]["dimensions"],
            ["query", "device", "country"],
        )


class ExecutionTest(unittest.TestCase):
    def test_unicode_alias_is_selected_when_canonical_has_no_row(self) -> None:
        plan = build_request_plan(publication(), "2026-08-13", "2026-08-14")
        unicode_url = unquote(PUBLISHED_URL)

        def execute(site_url: str, body: dict) -> dict:
            self.assertEqual(site_url, "https://www.lmdev.org/wpblog/")
            expression = body["dimensionFilterGroups"][0]["filters"][0]["expression"]
            if "dimensions" in body:
                return {"rows": []}
            return {"rows": [metric_row()]} if expression == unicode_url else {}

        result = execute_search_analytics(plan, query_executor=execute)
        self.assertEqual(result.selected_page_expression, unicode_url)
        self.assertEqual(aggregate_metrics(result.aggregate_response)["clicks"], 2)

    def test_different_nonempty_alias_totals_are_rejected(self) -> None:
        plan = build_request_plan(publication(), "2026-08-13", "2026-08-14")
        count = 0

        def execute(_site_url: str, body: dict) -> dict:
            nonlocal count
            if "dimensions" in body:
                return {"rows": []}
            count += 1
            return {"rows": [metric_row(clicks=count)]}

        with self.assertRaisesRegex(ValueError, "二重計上"):
            execute_search_analytics(plan, query_executor=execute)

    def test_detail_rows_are_paginated(self) -> None:
        plan = build_request_plan(
            publication(),
            "2026-08-13",
            "2026-08-14",
            row_limit=2,
        )

        def execute(_site_url: str, body: dict) -> dict:
            if "dimensions" not in body:
                return {"rows": [metric_row()]}
            if body["startRow"] == 0:
                return {
                    "rows": [
                        metric_row(keys=["mlx", "DESKTOP", "jpn"]),
                        metric_row(keys=["mlflow", "MOBILE", "jpn"]),
                    ]
                }
            return {"rows": [metric_row(keys=["optuna", "DESKTOP", "usa"])]}

        result = execute_search_analytics(
            plan,
            query_executor=execute,
            row_limit=2,
        )
        self.assertEqual([item["startRow"] for item in result.detail_requests], [0, 2])
        records = build_detail_records(result.detail_responses)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["query"], "mlx")
        self.assertEqual(records[2]["country"], "usa")


class ObservationTest(unittest.TestCase):
    def execution(self, aggregate: dict, details: list[dict] | None = None) -> GSCExecutionResult:
        return GSCExecutionResult(
            selected_page_expression=PUBLISHED_URL,
            aggregate_attempts=[],
            aggregate_response=aggregate,
            detail_requests=[],
            detail_responses=details or [{"rows": []}],
        )

    def test_empty_success_is_zero_with_nullable_rates(self) -> None:
        metrics = aggregate_metrics({"rows": []})
        self.assertEqual(metrics["clicks"], 0)
        self.assertEqual(metrics["impressions"], 0)
        self.assertIsNone(metrics["ctr"])
        self.assertIsNone(metrics["position"])

    def test_zero_metric_row_is_normalized_to_nullable_rates(self) -> None:
        metrics = aggregate_metrics(
            {
                "rows": [
                    metric_row(
                        clicks=0,
                        impressions=0,
                        ctr=0,
                        position=0,
                    )
                ]
            }
        )
        self.assertEqual(metrics["clicks"], 0)
        self.assertEqual(metrics["impressions"], 0)
        self.assertIsNone(metrics["ctr"])
        self.assertIsNone(metrics["position"])

    def test_click_without_impression_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "impressions=0"):
            aggregate_metrics(
                {
                    "rows": [
                        metric_row(
                            clicks=1,
                            impressions=0,
                            ctr=0,
                            position=0,
                        )
                    ]
                }
            )

    def test_observation_separates_summary_and_detail_identity(self) -> None:
        observation = build_observation(
            publication(),
            "2026-08-13",
            "2026-08-14",
            self.execution({"rows": [metric_row()]}),
            "a" * 64,
            "b" * 64,
            "2026-08-14T06:00:00-07:00",
            includes_today=False,
            data_state="final",
        )
        self.assertEqual(observation["source"], "gsc")
        self.assertEqual(observation["collector_version"], COLLECTOR_VERSION)
        self.assertEqual(observation["metrics"]["impressions"], 10)
        self.assertEqual(
            observation["dimensions"]["detail_artifact_sha256"],
            "b" * 64,
        )
        self.assertFalse(observation["is_partial"])

    def test_today_or_incomplete_metadata_is_partial(self) -> None:
        response = {
            "rows": [metric_row()],
            "metadata": {"firstIncompleteDate": "2026-08-14"},
        }
        observation = build_observation(
            publication(),
            "2026-08-13",
            "2026-08-14",
            self.execution(response),
            "a" * 64,
            "b" * 64,
            "2026-08-14T06:00:00-07:00",
            includes_today=True,
            data_state="all",
        )
        self.assertTrue(observation["is_partial"])

    def test_zero_metric_row_note_does_not_claim_zero_rows(self) -> None:
        response = {
            "rows": [
                metric_row(
                    clicks=0,
                    impressions=0,
                    ctr=0,
                    position=0,
                )
            ]
        }
        observation = build_observation(
            publication(),
            "2026-08-13",
            "2026-08-14",
            self.execution(response),
            "a" * 64,
            "b" * 64,
            "2026-08-14T06:00:00-07:00",
            includes_today=True,
            data_state="all",
        )
        notes = "\n".join(observation["notes"])
        self.assertIn("ゼロ指標Row", notes)
        self.assertIn("CTRとPositionはnull", notes)
        self.assertNotIn("Rowは0件", notes)


class PersistenceTest(unittest.TestCase):
    def test_same_collection_is_idempotent(self) -> None:
        plan = build_request_plan(publication(), "2026-08-13", "2026-08-14")
        execution = GSCExecutionResult(
            selected_page_expression=PUBLISHED_URL,
            aggregate_attempts=[
                {
                    "page_expression": PUBLISHED_URL,
                    "request": plan["aggregate_requests"][0],
                    "response": {"rows": [metric_row()]},
                }
            ],
            aggregate_response={"rows": [metric_row()]},
            detail_requests=[plan["detail_request_template"]],
            detail_responses=[
                {"rows": [metric_row(keys=["mlflow", "DESKTOP", "jpn"])]}
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kwargs = {
                "publication": publication(),
                "start_date": "2026-08-13",
                "end_date": "2026-08-14",
                "plan": plan,
                "execution": execution,
                "raw_dir": root / "raw",
                "detail_dir": root / "details",
                "observation_path": root / "online_metrics.jsonl",
                "includes_today": True,
                "data_state": "all",
            }
            first = persist_collection(
                **kwargs,
                collected_at="2026-08-14T06:00:00-07:00",
            )
            second = persist_collection(
                **kwargs,
                collected_at="2026-08-14T06:05:00-07:00",
            )
            self.assertEqual(first.status, "created")
            self.assertEqual(second.status, "unchanged")
            lines = (root / "online_metrics.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertTrue(first.raw_path.exists())
            self.assertTrue(first.detail_path.exists())
            detail = json.loads(first.detail_path.read_text())
            self.assertEqual(detail["row_count"], 1)


if __name__ == "__main__":
    unittest.main()
