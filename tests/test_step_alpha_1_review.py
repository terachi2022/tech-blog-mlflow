from __future__ import annotations

import hashlib
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace

from tech_blog_mlflow.review_workflow import (
    HUMAN_SCORE_NAMES,
    JUDGE_AXES,
    NOTES_NAME,
    PUBLISHABLE_NAME,
    build_validation_report,
    extract_article_payload,
    expected_schema_contract,
    ensure_review_presentation,
    find_review_presentation_trace,
    latest_assessment_values,
    markdown_preview_valid,
    question_specs,
    raw_markdown_preview_shape_valid,
    resolve_single_trace,
    schema_contract,
)


@dataclass
class FakeInputCategorical:
    options: list[str]
    multi_select: bool = False


FakeInputCategorical.__name__ = "InputCategorical"


@dataclass
class FakeAssessment:
    name: str
    value: object
    last_update_time_ms: int = 0
    valid: bool = True
    source: object = field(
        default_factory=lambda: SimpleNamespace(source_type="HUMAN")
    )


@dataclass
class FakeInfo:
    trace_id: str
    tags: dict[str, str] = field(default_factory=dict)
    assessments: list[FakeAssessment] = field(default_factory=list)
    response_preview: str = "# Article\n\n## Section\n\nBody"


@dataclass
class FakeTrace:
    info: FakeInfo
    data: object | None = None


@dataclass
class FakeQueue:
    queue_id: str = "queue-1"
    name: str = "article-quality-human-review-v2"
    schema_ids: list[str] = field(
        default_factory=lambda: [f"schema-{index}" for index in range(8)]
    )


@dataclass
class FakeItem:
    item_id: str
    status: str
    queue_id: str = "queue-1"
    item_type: str = "trace"
    completed_by: str | None = None
    completed_time_ms: int | None = None


class QuestionContractTest(unittest.TestCase):
    def test_eight_unique_questions_are_defined(self):
        specs = question_specs()
        self.assertEqual(len(specs), 8)
        self.assertEqual(len({spec.name for spec in specs}), 8)
        self.assertEqual(
            {spec.name for spec in specs},
            set(HUMAN_SCORE_NAMES.values()) | {PUBLISHABLE_NAME, NOTES_NAME},
        )

    def test_scores_use_oss_compatible_categorical_widget(self):
        for spec in question_specs()[:6]:
            contract = expected_schema_contract(spec)
            self.assertEqual(contract["widget"], "categorical")
            self.assertEqual(contract["options"], ["1", "2", "3", "4", "5"])
            self.assertFalse(contract["multi_select"])

    def test_existing_schema_contract_can_be_compared(self):
        spec = question_specs()[0]
        schema = SimpleNamespace(
            name=spec.name,
            type=SimpleNamespace(value="feedback"),
            input=FakeInputCategorical(["1", "2", "3", "4", "5"]),
            instruction=spec.instruction,
            enable_comment=True,
        )
        self.assertEqual(schema_contract(schema), expected_schema_contract(spec))


class TraceResolutionTest(unittest.TestCase):
    def test_single_trace_is_resolved(self):
        calls = []

        def search_fn(**kwargs):
            calls.append(kwargs)
            return [FakeTrace(FakeInfo("tr-1", {"article_variant": "baseline"}))]

        target = resolve_single_trace(
            search_fn,
            experiment_id="1",
            evaluation_run_id="run-1",
            label="baseline",
        )
        self.assertEqual(target.trace_id, "tr-1")
        self.assertEqual(target.article_variant, "baseline")
        self.assertEqual(calls[0]["run_id"], "run-1")
        self.assertEqual(calls[0]["locations"], ["1"])
        self.assertEqual(calls[0]["return_type"], "list")

    def test_missing_trace_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Traceがありません"):
            resolve_single_trace(
                lambda **_: [],
                experiment_id="1",
                evaluation_run_id="run-1",
                label="baseline",
            )

    def test_multiple_traces_are_rejected(self):
        traces = [
            FakeTrace(FakeInfo("tr-1")),
            FakeTrace(FakeInfo("tr-2")),
        ]
        with self.assertRaisesRegex(ValueError, "1 Trace"):
            resolve_single_trace(
                lambda **_: traces,
                experiment_id="1",
                evaluation_run_id="run-1",
                label="baseline",
            )


class MarkdownPresentationTest(unittest.TestCase):
    def make_source_trace(self, article: str | None = None):
        markdown = article or "# Title\n\n## Steps\n\n```bash\necho ok\n```"
        sha256 = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
        root = SimpleNamespace(
            parent_id=None,
            inputs={
                "article_path": "articles/example.md",
                "article_sha256": sha256,
            },
            outputs=markdown,
        )
        return FakeTrace(
            FakeInfo("tr-source"),
            data=SimpleNamespace(spans=[root]),
        )

    def test_article_markdown_and_sha_are_extracted(self):
        payload = extract_article_payload(self.make_source_trace())
        self.assertEqual(payload.article_path, "articles/example.md")
        self.assertTrue(payload.article_markdown.startswith("# Title"))
        self.assertEqual(
            payload.article_sha256,
            hashlib.sha256(payload.article_markdown.encode()).hexdigest(),
        )

    def test_json_encoded_markdown_preview_is_rejected(self):
        article = "# Title\n\n## Steps\n\nBody"
        encoded = '"# Title\\n\\n## Steps\\n\\nBody"'
        self.assertFalse(raw_markdown_preview_shape_valid(encoded))
        self.assertFalse(
            markdown_preview_valid(
                response_preview=encoded,
                article_markdown=article,
            )
        )

    def test_raw_markdown_preview_is_accepted_only_on_full_match(self):
        article = "# Title\n\n## Steps\n\nBody"
        self.assertTrue(
            markdown_preview_valid(
                response_preview=article,
                article_markdown=article,
            )
        )
        self.assertFalse(
            markdown_preview_valid(
                response_preview=article[:-1],
                article_markdown=article,
            )
        )

    def test_sha_mismatch_is_rejected(self):
        trace = self.make_source_trace()
        trace.data.spans[0].inputs["article_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            extract_article_payload(trace)

    def test_review_trace_search_is_scoped_by_location_and_tags(self):
        calls = []

        def search_fn(**kwargs):
            calls.append(kwargs)
            return [FakeTrace(FakeInfo("tr-review"))]

        result = find_review_presentation_trace(
            search_fn,
            experiment_id="1",
            source_trace_id="tr-source",
        )
        self.assertEqual(result.info.trace_id, "tr-review")
        self.assertEqual(calls[0]["locations"], ["1"])
        self.assertIn("review_source_trace_id", calls[0]["filter_string"])
        self.assertIn(
            "review_presentation_version",
            calls[0]["filter_string"],
        )

    def test_new_review_trace_is_created_in_queue_experiment(self):
        source_trace = self.make_source_trace()
        source_trace.info.assessments = [
            FakeAssessment(axis, 4.0) for axis in JUDGE_AXES
        ]
        events = []

        class FakeSpan:
            trace_id = "tr-review"

            def __enter__(self):
                events.append("start_span")
                return self

            def __exit__(self, *_):
                return False

            def set_inputs(self, _):
                pass

            def set_outputs(self, _):
                pass

        review_trace = FakeTrace(FakeInfo("tr-review"))

        class FakeMlflow:
            @staticmethod
            def search_traces(**_):
                return []

            @staticmethod
            def set_experiment(*, experiment_id):
                events.append(("set_experiment", experiment_id))

            @staticmethod
            def start_span(*, name):
                self.assertEqual(name, "article-human-review")
                return FakeSpan()

            @staticmethod
            def update_current_trace(**kwargs):
                review_trace.info.response_preview = kwargs["response_preview"]

            @staticmethod
            def get_trace(trace_id, *, flush):
                self.assertEqual(trace_id, "tr-review")
                self.assertTrue(flush)
                return review_trace

            @staticmethod
            def log_feedback(*, name, value, **_):
                review_trace.info.assessments.append(
                    FakeAssessment(name, value)
                )

        target = SimpleNamespace(
            label="baseline",
            article_variant="baseline",
            evaluation_run_id="run-1",
            trace_id="tr-source",
        )
        result = ensure_review_presentation(
            mlflow_module=FakeMlflow,
            experiment_id="1",
            target=target,
            source_trace=source_trace,
        )

        self.assertEqual(result.review_trace_id, "tr-review")
        self.assertEqual(events[0], ("set_experiment", "1"))
        self.assertEqual(events[1], "start_span")


class AssessmentTest(unittest.TestCase):
    def test_latest_valid_assessment_is_selected(self):
        values = latest_assessment_values(
            [
                FakeAssessment("score", "2", last_update_time_ms=1),
                FakeAssessment("score", "3", last_update_time_ms=2),
                FakeAssessment("score", "5", last_update_time_ms=3, valid=False),
            ]
        )
        self.assertEqual(values["score"], "3")

    @staticmethod
    def complete_assessments(human_shift: float = 0.0):
        assessments = []
        for axis in JUDGE_AXES:
            assessments.append(FakeAssessment(axis, 4.0))
            assessments.append(
                FakeAssessment(HUMAN_SCORE_NAMES[axis], str(4.0 + human_shift))
            )
        assessments.extend(
            [
                FakeAssessment(PUBLISHABLE_NAME, True),
                FakeAssessment(NOTES_NAME, "確認済み"),
            ]
        )
        return assessments

    def test_pending_queue_reports_setup_only(self):
        trace = FakeTrace(FakeInfo("tr-1"))
        report = build_validation_report(
            queue=FakeQueue(),
            items=[FakeItem("tr-1", "pending")],
            traces_by_id={"tr-1": trace},
            expected_trace_ids={"tr-1"},
        )
        self.assertEqual(report["validation_status"], "ready_for_human_review")
        self.assertTrue(report["effectiveness"]["setup_effective"])
        self.assertFalse(
            report["effectiveness"]["workflow_completion_effective"]
        )
        self.assertIsNone(report["judge_alignment"]["mean_absolute_error"])

    def test_complete_reviews_calculate_alignment(self):
        trace_a = FakeTrace(
            FakeInfo("tr-a", assessments=self.complete_assessments(0.0))
        )
        trace_b = FakeTrace(
            FakeInfo("tr-b", assessments=self.complete_assessments(1.0))
        )
        items = [
            FakeItem("tr-a", "complete", completed_by="default"),
            FakeItem("tr-b", "complete", completed_by="default"),
        ]
        report = build_validation_report(
            queue=FakeQueue(),
            items=items,
            traces_by_id={"tr-a": trace_a, "tr-b": trace_b},
            expected_trace_ids={"tr-a", "tr-b"},
        )
        self.assertEqual(report["validation_status"], "validated")
        self.assertTrue(
            report["effectiveness"]["workflow_completion_effective"]
        )
        self.assertEqual(report["judge_alignment"]["comparison_count"], 12)
        self.assertEqual(report["judge_alignment"]["mean_absolute_error"], 0.5)
        self.assertEqual(report["judge_alignment"]["exact_agreement_rate"], 0.5)
        self.assertEqual(report["judge_alignment"]["within_one_agreement_rate"], 1.0)

    def test_unrelated_complete_item_does_not_complete_expected_item(self):
        expected_trace = FakeTrace(
            FakeInfo("tr-expected", assessments=self.complete_assessments())
        )
        extra_trace = FakeTrace(
            FakeInfo("tr-extra", assessments=self.complete_assessments())
        )
        report = build_validation_report(
            queue=FakeQueue(),
            items=[
                FakeItem("tr-expected", "pending"),
                FakeItem("tr-extra", "complete"),
            ],
            traces_by_id={
                "tr-expected": expected_trace,
                "tr-extra": extra_trace,
            },
            expected_trace_ids={"tr-expected"},
        )
        self.assertFalse(report["effectiveness"]["expected_items_complete"])
        self.assertFalse(
            report["effectiveness"]["workflow_completion_effective"]
        )

    def test_schema_contract_mismatch_invalidates_setup(self):
        trace = FakeTrace(FakeInfo("tr-1"))
        report = build_validation_report(
            queue=FakeQueue(),
            items=[FakeItem("tr-1", "pending")],
            traces_by_id={"tr-1": trace},
            expected_trace_ids={"tr-1"},
            schema_contracts_valid=False,
        )
        self.assertEqual(report["validation_status"], "invalid_setup")
        self.assertFalse(report["effectiveness"]["setup_effective"])

    def test_json_encoded_preview_invalidates_setup(self):
        trace = FakeTrace(
            FakeInfo(
                "tr-1",
                response_preview='"# Article\\n\\n## Section\\n\\nBody"',
            )
        )
        report = build_validation_report(
            queue=FakeQueue(),
            items=[FakeItem("tr-1", "pending")],
            traces_by_id={"tr-1": trace},
            expected_trace_ids={"tr-1"},
        )
        self.assertEqual(report["validation_status"], "invalid_setup")
        self.assertFalse(
            report["effectiveness"]["markdown_previews_valid"]
        )

    def test_invalid_human_score_is_rejected(self):
        assessments = self.complete_assessments(0.0)
        assessments[1].value = "6"
        trace = FakeTrace(FakeInfo("tr-1", assessments=assessments))
        with self.assertRaisesRegex(ValueError, "1〜5"):
            build_validation_report(
                queue=FakeQueue(),
                items=[FakeItem("tr-1", "complete")],
                traces_by_id={"tr-1": trace},
                expected_trace_ids={"tr-1"},
            )

    def test_code_generated_human_label_is_not_accepted(self):
        assessments = self.complete_assessments(0.0)
        assessments[1].source = SimpleNamespace(source_type="CODE")
        trace = FakeTrace(FakeInfo("tr-1", assessments=assessments))
        report = build_validation_report(
            queue=FakeQueue(),
            items=[FakeItem("tr-1", "complete")],
            traces_by_id={"tr-1": trace},
            expected_trace_ids={"tr-1"},
        )
        self.assertFalse(report["trace_reviews"][0]["human_sources_valid"])
        self.assertFalse(
            report["effectiveness"]["human_assessment_capture_effective"]
        )


if __name__ == "__main__":
    unittest.main()
