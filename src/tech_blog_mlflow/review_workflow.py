"""MLflow Review Queueを使った人手評価Workflow。

STEP-α-1では、Baselineと採用Candidateの記事をMarkdown表示専用Traceへ
複製して同じQueueへ登録し、Local LLM Judgeと同じ6軸を人が独立採点
できるようにする。元の評価Traceは変更しない。
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Sequence


TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "tech-blog-generation"
QUEUE_NAME = "article-quality-human-review-v2"
WORKFLOW_VERSION = "step-alpha-1-review-v2.0.0"
REVIEW_PRESENTATION_VERSION = "markdown-review-trace-v1"
REVIEW_TRACE_NAME = "article-human-review"
MAX_MARKDOWN_PREVIEW_CHARS = 9000

SOURCE_TRACE_TAG = "review_source_trace_id"
SOURCE_RUN_TAG = "review_source_evaluation_run_id"
PRESENTATION_VERSION_TAG = "review_presentation_version"
TARGET_LABEL_TAG = "review_target_label"

BASELINE_EVALUATION_RUN_ID = (
    "4f56c781fdfb4e95805c6b957302373f"
)
ADOPTED_EVALUATION_RUN_ID = (
    "fdf0c239445f44a0999a6b1fe7a419b6"
)

JUDGE_AXES = (
    "technical_accuracy",
    "helpfulness",
    "reproducibility",
    "citation_quality",
    "readability_ja",
    "original_value",
)

HUMAN_SCORE_NAMES = {
    axis: f"human_{axis}_v1"
    for axis in JUDGE_AXES
}

PUBLISHABLE_NAME = "human_publishable_v1"
NOTES_NAME = "human_review_notes_v1"


@dataclass(frozen=True)
class QuestionSpec:
    """Review画面へ登録する質問のVersion固定仕様。"""

    name: str
    widget: str
    instruction: str
    enable_comment: bool = True
    options: tuple[str, ...] = ()
    positive_label: str | None = None
    negative_label: str | None = None
    max_length: int | None = None
    schema_type: str = "feedback"


@dataclass(frozen=True)
class ReviewTarget:
    """評価Runと、そのRunが生成したTraceの対応。"""

    label: str
    evaluation_run_id: str
    trace_id: str
    article_variant: str | None


@dataclass(frozen=True)
class ArticlePayload:
    """評価Traceから取得したReview表示用の記事と不変Metadata。"""

    article_markdown: str
    article_path: str
    article_sha256: str


@dataclass(frozen=True)
class ReviewPresentation:
    """元評価TraceとMarkdown表示専用Traceの対応。"""

    label: str
    evaluation_run_id: str
    source_trace_id: str
    review_trace_id: str
    article_variant: str | None
    article_path: str
    article_sha256: str
    article_chars: int
    reused: bool
    markdown_preview_valid: bool


def question_specs() -> tuple[QuestionSpec, ...]:
    """6軸採点、公開可否、総評の質問を返す。"""

    descriptions = {
        "technical_accuracy": (
            "技術概念、API、Command、記事内の整合性を1（重大な誤り）〜"
            "5（正確）で採点してください。自動評価値を根拠にせず、"
            "記事本文を確認してください。"
        ),
        "helpfulness": (
            "目的の明確さ、実行可能性、対象読者への適合、切り分けへの"
            "有用性を1〜5で採点してください。"
        ),
        "reproducibility": (
            "環境、依存関係、Code、実行順、成功確認を含む再現性を"
            "1〜5で採点してください。"
        ),
        "citation_quality": (
            "出典の権威性、主張との対応、Coverage、Link Contextを"
            "1〜5で採点してください。"
        ),
        "readability_ja": (
            "構成、文の明瞭さ、用語説明、情報密度を含む日本語の"
            "読みやすさを1〜5で採点してください。"
        ),
        "original_value": (
            "実測根拠、失敗分析、比較から得た知見、環境固有の知見を"
            "1〜5で採点してください。"
        ),
    }

    scores = tuple(
        QuestionSpec(
            name=HUMAN_SCORE_NAMES[axis],
            widget="categorical",
            options=("1", "2", "3", "4", "5"),
            instruction=descriptions[axis],
        )
        for axis in JUDGE_AXES
    )

    return scores + (
        QuestionSpec(
            name=PUBLISHABLE_NAME,
            widget="pass_fail",
            positive_label="Publishable",
            negative_label="Needs revision",
            instruction=(
                "現在の内容を公開可能と判断する場合はPublishable、"
                "修正が必要な場合はNeeds revisionを選択してください。"
            ),
        ),
        QuestionSpec(
            name=NOTES_NAME,
            widget="text",
            max_length=4000,
            instruction=(
                "採点根拠、重要な問題、採用判断に必要な補足を記録して"
                "ください。特記事項がなければ空欄でも構いません。"
            ),
            enable_comment=False,
        ),
    )


def expected_schema_contract(spec: QuestionSpec) -> dict[str, Any]:
    """QuestionSpecを比較可能なContractへ変換する。"""

    contract: dict[str, Any] = {
        "name": spec.name,
        "type": spec.schema_type,
        "widget": spec.widget,
        "instruction": spec.instruction,
        "enable_comment": spec.enable_comment,
    }
    if spec.widget == "categorical":
        contract["options"] = list(spec.options)
        contract["multi_select"] = False
    elif spec.widget == "pass_fail":
        contract["positive_label"] = spec.positive_label
        contract["negative_label"] = spec.negative_label
    elif spec.widget == "text":
        contract["max_length"] = spec.max_length
    else:
        raise ValueError(
            f"未対応のReview widgetです: {spec.widget}"
        )
    return contract


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def schema_contract(schema: Any) -> dict[str, Any]:
    """MLflow LabelSchemaを比較可能なContractへ変換する。"""

    input_spec = schema.input
    class_name = type(input_spec).__name__
    contract: dict[str, Any] = {
        "name": schema.name,
        "type": _enum_value(schema.type),
        "instruction": schema.instruction,
        "enable_comment": bool(schema.enable_comment),
    }

    if class_name == "InputCategorical":
        contract.update(
            {
                "widget": "categorical",
                "options": list(input_spec.options),
                "multi_select": bool(input_spec.multi_select),
            }
        )
    elif class_name == "InputPassFail":
        contract.update(
            {
                "widget": "pass_fail",
                "positive_label": input_spec.positive_label,
                "negative_label": input_spec.negative_label,
            }
        )
    elif class_name == "InputText":
        contract.update(
            {
                "widget": "text",
                "max_length": input_spec.max_length,
            }
        )
    else:
        contract["widget"] = class_name

    return contract


def question_input(spec: QuestionSpec) -> Any:
    """QuestionSpecからMLflowのInput objectを生成する。"""

    from mlflow.genai.label_schemas import (
        InputCategorical,
        InputPassFail,
        InputText,
    )

    if spec.widget == "categorical":
        return InputCategorical(
            options=list(spec.options),
            multi_select=False,
        )
    if spec.widget == "pass_fail":
        return InputPassFail(
            positive_label=str(spec.positive_label),
            negative_label=str(spec.negative_label),
        )
    if spec.widget == "text":
        return InputText(max_length=spec.max_length)
    raise ValueError(f"未対応のReview widgetです: {spec.widget}")


def resolve_single_trace(
    search_fn: Callable[..., Sequence[Any]],
    *,
    experiment_id: str,
    evaluation_run_id: str,
    label: str,
) -> ReviewTarget:
    """評価Runに紐づく唯一のTraceを取得する。"""

    traces = list(
        search_fn(
            run_id=evaluation_run_id,
            locations=[experiment_id],
            return_type="list",
        )
    )
    if not traces:
        raise ValueError(
            "評価RunにTraceがありません: "
            f"label={label}, run_id={evaluation_run_id}"
        )
    if len(traces) != 1:
        raise ValueError(
            "STEP-α-1は評価Runあたり1 Traceを前提とします: "
            f"label={label}, run_id={evaluation_run_id}, "
            f"trace_count={len(traces)}"
        )

    trace = traces[0]
    tags = getattr(trace.info, "tags", {}) or {}
    return ReviewTarget(
        label=label,
        evaluation_run_id=evaluation_run_id,
        trace_id=trace.info.trace_id,
        article_variant=tags.get("article_variant"),
    )


def _root_span(trace: Any) -> Any:
    """Traceから唯一のRoot Spanを取得する。"""

    spans = list(getattr(getattr(trace, "data", None), "spans", []) or [])
    roots = [span for span in spans if getattr(span, "parent_id", None) is None]
    if len(roots) != 1:
        raise ValueError(
            "Review対象TraceはRoot Spanが1件である必要があります: "
            f"trace_id={trace.info.trace_id}, root_count={len(roots)}"
        )
    return roots[0]


def extract_article_payload(trace: Any) -> ArticlePayload:
    """評価TraceのRoot Spanから元MarkdownとMetadataを検証して取得する。"""

    root = _root_span(trace)
    article = getattr(root, "outputs", None)
    inputs = getattr(root, "inputs", None) or {}
    if not isinstance(article, str) or not article.strip():
        raise ValueError(
            "評価TraceのoutputsがMarkdown文字列ではありません: "
            f"trace_id={trace.info.trace_id}"
        )
    if not article.startswith("# ") or "\n## " not in article:
        raise ValueError(
            "記事がH1/H2を持つMarkdownではありません: "
            f"trace_id={trace.info.trace_id}"
        )
    if len(article) > MAX_MARKDOWN_PREVIEW_CHARS:
        raise ValueError(
            "Review画面へ全文表示できる上限を超えています。"
            "記事を切断せず、Review表示方式を再設計してください: "
            f"trace_id={trace.info.trace_id}, chars={len(article)}, "
            f"limit={MAX_MARKDOWN_PREVIEW_CHARS}"
        )
    if not isinstance(inputs, dict):
        raise ValueError(
            "評価Traceのinputsがdictではありません: "
            f"trace_id={trace.info.trace_id}"
        )

    article_path = str(inputs.get("article_path") or "")
    recorded_sha = str(inputs.get("article_sha256") or "")
    actual_sha = hashlib.sha256(article.encode("utf-8")).hexdigest()
    if not article_path:
        raise ValueError(
            "評価Traceにarticle_pathがありません: "
            f"trace_id={trace.info.trace_id}"
        )
    if not recorded_sha:
        raise ValueError(
            "評価Traceにarticle_sha256がありません: "
            f"trace_id={trace.info.trace_id}"
        )
    if recorded_sha != actual_sha:
        raise ValueError(
            "評価Traceの記事SHA-256がoutputsと一致しません: "
            f"trace_id={trace.info.trace_id}, "
            f"recorded={recorded_sha}, actual={actual_sha}"
        )
    return ArticlePayload(
        article_markdown=article,
        article_path=article_path,
        article_sha256=actual_sha,
    )


def markdown_preview_valid(
    *,
    response_preview: Any,
    article_markdown: str,
) -> bool:
    """Review UIへ渡すPreviewがJSON化されず全文一致することを確認する。"""

    return (
        raw_markdown_preview_shape_valid(response_preview)
        and response_preview == article_markdown
    )


def raw_markdown_preview_shape_valid(response_preview: Any) -> bool:
    """PreviewがJSON文字列ではなくMarkdown本文として見える形か判定する。"""

    return (
        isinstance(response_preview, str)
        and response_preview.startswith("# ")
        and "\n## " in response_preview
        and not response_preview.startswith('"')
        and len(response_preview) <= MAX_MARKDOWN_PREVIEW_CHARS
    )


def _quoted_filter_value(value: str) -> str:
    """Trace検索DSLの単引用符をEscapeする。"""

    return value.replace("\\", "\\\\").replace("'", "\\'")


def find_review_presentation_trace(
    search_fn: Callable[..., Sequence[Any]],
    *,
    experiment_id: str,
    source_trace_id: str,
) -> Any | None:
    """元Traceと表示Versionが一致するReview専用Traceを冪等検索する。"""

    source = _quoted_filter_value(source_trace_id)
    version = _quoted_filter_value(REVIEW_PRESENTATION_VERSION)
    traces = list(
        search_fn(
            locations=[experiment_id],
            filter_string=(
                f"tag.{SOURCE_TRACE_TAG} = '{source}' AND "
                f"tag.{PRESENTATION_VERSION_TAG} = '{version}'"
            ),
            max_results=3,
            return_type="list",
        )
    )
    if len(traces) > 1:
        raise ValueError(
            "同じ元TraceにReview表示専用Traceが複数あります: "
            f"source_trace_id={source_trace_id}, count={len(traces)}"
        )
    return traces[0] if traces else None


def _judge_feedback_payloads(trace: Any) -> dict[str, Any]:
    """元評価Traceの有効なJudge 6軸Feedbackを取得する。"""

    latest = latest_assessments(trace.info.assessments)
    missing = [axis for axis in JUDGE_AXES if axis not in latest]
    if missing:
        raise ValueError(
            "元評価TraceにJudge評価が不足しています: "
            f"trace_id={trace.info.trace_id}, missing={missing}"
        )
    return {axis: latest[axis] for axis in JUDGE_AXES}


def ensure_copied_judge_feedback(
    *,
    mlflow_module: Any,
    source_trace: Any,
    review_trace: Any,
) -> list[str]:
    """Judge 6軸をReview Traceへ出典付きで冪等複製する。"""

    from mlflow.entities import AssessmentSource, AssessmentSourceType

    source_feedback = _judge_feedback_payloads(source_trace)
    current = latest_assessments(review_trace.info.assessments)
    copied: list[str] = []
    for axis, assessment in source_feedback.items():
        if axis in current:
            if current[axis].value != assessment.value:
                raise ValueError(
                    "Review TraceのJudge値が元Traceと一致しません: "
                    f"axis={axis}, review_trace_id={review_trace.info.trace_id}"
                )
            continue
        mlflow_module.log_feedback(
            trace_id=review_trace.info.trace_id,
            name=axis,
            value=assessment.value,
            rationale=getattr(assessment, "rationale", None),
            source=AssessmentSource(
                source_type=AssessmentSourceType.CODE,
                source_id="review-workflow-copy",
            ),
            metadata={
                "copied_from_trace_id": source_trace.info.trace_id,
                "copied_by": WORKFLOW_VERSION,
            },
        )
        copied.append(axis)
    return copied


def ensure_review_presentation(
    *,
    mlflow_module: Any,
    experiment_id: str,
    target: ReviewTarget,
    source_trace: Any,
) -> ReviewPresentation:
    """Markdown全文をRaw Previewに持つReview表示専用Traceを冪等作成する。"""

    payload = extract_article_payload(source_trace)
    review_trace = find_review_presentation_trace(
        mlflow_module.search_traces,
        experiment_id=experiment_id,
        source_trace_id=target.trace_id,
    )
    reused = review_trace is not None

    if review_trace is None:
        # start_span() records a new trace in the active experiment.  Merely
        # scoping search_traces() and the Review Queue to experiment_id does
        # not change that destination, so select it explicitly before creating
        # the presentation trace.
        mlflow_module.set_experiment(experiment_id=experiment_id)
        request_preview = (
            f"## Review target\n\n"
            f"- Label: `{target.label}`\n"
            f"- Variant: `{target.article_variant or 'unknown'}`\n"
            f"- Article: `{payload.article_path}`\n"
            f"- Source evaluation run: `{target.evaluation_run_id}`"
        )
        with mlflow_module.start_span(name=REVIEW_TRACE_NAME) as span:
            span.set_inputs(
                {
                    "label": target.label,
                    "article_variant": target.article_variant,
                    "article_path": payload.article_path,
                    "article_sha256": payload.article_sha256,
                    "source_evaluation_run_id": target.evaluation_run_id,
                    "source_evaluation_trace_id": target.trace_id,
                }
            )
            mlflow_module.update_current_trace(
                request_preview=request_preview,
                response_preview=payload.article_markdown,
                tags={
                    SOURCE_TRACE_TAG: target.trace_id,
                    SOURCE_RUN_TAG: target.evaluation_run_id,
                    PRESENTATION_VERSION_TAG: REVIEW_PRESENTATION_VERSION,
                    TARGET_LABEL_TAG: target.label,
                    "article_variant": target.article_variant or "unknown",
                },
                metadata={
                    SOURCE_TRACE_TAG: target.trace_id,
                    SOURCE_RUN_TAG: target.evaluation_run_id,
                    "article_path": payload.article_path,
                    "article_sha256": payload.article_sha256,
                    PRESENTATION_VERSION_TAG: REVIEW_PRESENTATION_VERSION,
                },
            )
            span.set_outputs(payload.article_markdown)
            review_trace_id = span.trace_id
        review_trace = mlflow_module.get_trace(review_trace_id, flush=True)
        if review_trace is None:
            raise ValueError(
                "作成したReview表示専用Traceを取得できません: "
                f"trace_id={review_trace_id}"
            )

    if not markdown_preview_valid(
        response_preview=review_trace.info.response_preview,
        article_markdown=payload.article_markdown,
    ):
        raise ValueError(
            "Review Traceのresponse_previewがMarkdown全文ではありません: "
            f"review_trace_id={review_trace.info.trace_id}"
        )

    ensure_copied_judge_feedback(
        mlflow_module=mlflow_module,
        source_trace=source_trace,
        review_trace=review_trace,
    )
    refreshed = mlflow_module.get_trace(review_trace.info.trace_id, flush=True)
    if refreshed is None:
        raise ValueError(
            "Judge複製後のReview Traceを取得できません: "
            f"trace_id={review_trace.info.trace_id}"
        )
    _judge_feedback_payloads(refreshed)

    return ReviewPresentation(
        label=target.label,
        evaluation_run_id=target.evaluation_run_id,
        source_trace_id=target.trace_id,
        review_trace_id=refreshed.info.trace_id,
        article_variant=target.article_variant,
        article_path=payload.article_path,
        article_sha256=payload.article_sha256,
        article_chars=len(payload.article_markdown),
        reused=reused,
        markdown_preview_valid=True,
    )


def ensure_label_schemas(
    *,
    experiment_id: str,
) -> tuple[list[Any], list[str]]:
    """Label Schemaを冪等に作成し、既存の仕様差分は停止する。"""

    from mlflow.genai.label_schemas import (
        create_label_schema,
        list_label_schemas,
    )

    existing = {
        schema.name: schema
        for schema in list_label_schemas(
            experiment_id=experiment_id,
            max_results=100,
        )
    }
    schemas: list[Any] = []
    created_names: list[str] = []

    for spec in question_specs():
        current = existing.get(spec.name)
        if current is not None:
            actual = schema_contract(current)
            expected = expected_schema_contract(spec)
            if actual != expected:
                raise ValueError(
                    "同名Label Schemaの仕様が異なります。"
                    "履歴を壊さないため自動更新しません: "
                    f"name={spec.name}, expected={expected}, actual={actual}"
                )
            schemas.append(current)
            continue

        created = create_label_schema(
            name=spec.name,
            type=spec.schema_type,
            input=question_input(spec),
            instruction=spec.instruction,
            enable_comment=spec.enable_comment,
            experiment_id=experiment_id,
        )
        schemas.append(created)
        created_names.append(spec.name)

    return schemas, created_names


def ensure_review_queue(
    *,
    experiment_id: str,
    queue_name: str,
    schema_ids: Sequence[str],
) -> tuple[Any, bool]:
    """Custom Review Queueを冪等に作成する。"""

    from mlflow.genai.review_queues import (
        create_review_queue,
        list_review_queues,
    )

    queues = list(
        list_review_queues(
            experiment_id=experiment_id,
            max_results=100,
        )
    )
    matches = [queue for queue in queues if queue.name == queue_name]
    if len(matches) > 1:
        raise ValueError(
            f"同名Review Queueが複数あります: {queue_name}"
        )

    expected_ids = set(schema_ids)
    if matches:
        queue = matches[0]
        actual_type = _enum_value(queue.queue_type)
        if actual_type != "custom":
            raise ValueError(
                "同名Queueのtypeがcustomではありません: "
                f"name={queue_name}, type={actual_type}"
            )
        if set(queue.schema_ids) != expected_ids:
            raise ValueError(
                "同名QueueのLabel Schema構成が異なります。"
                "Item投入後のSchema変更は行いません: "
                f"name={queue_name}"
            )
        if list(queue.users):
            raise ValueError(
                "無認証Local Server用Queueに想定外のuserがあります: "
                f"name={queue_name}, users={queue.users}"
            )
        return queue, False

    queue = create_review_queue(
        name=queue_name,
        queue_type="custom",
        users=[],
        schema_ids=list(schema_ids),
        experiment_id=experiment_id,
    )
    return queue, True


def setup_review_workflow(
    *,
    tracking_uri: str = TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
    queue_name: str = QUEUE_NAME,
    baseline_evaluation_run_id: str = BASELINE_EVALUATION_RUN_ID,
    candidate_evaluation_run_id: str = ADOPTED_EVALUATION_RUN_ID,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Review Queue、質問、対象Traceを構成する。"""

    import mlflow
    from mlflow.genai.review_queues import add_items_to_review_queue
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    experiment = MlflowClient().get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow Experimentがありません: {experiment_name}")

    targets = [
        resolve_single_trace(
            mlflow.search_traces,
            experiment_id=experiment.experiment_id,
            evaluation_run_id=baseline_evaluation_run_id,
            label="baseline",
        ),
        resolve_single_trace(
            mlflow.search_traces,
            experiment_id=experiment.experiment_id,
            evaluation_run_id=candidate_evaluation_run_id,
            label="adopted-candidate",
        ),
    ]
    source_trace_ids = [target.trace_id for target in targets]
    if len(set(source_trace_ids)) != len(source_trace_ids):
        raise ValueError("BaselineとCandidateが同じTraceを参照しています。")

    source_traces: list[Any] = []
    source_payloads: list[ArticlePayload] = []
    for target in targets:
        trace = mlflow.get_trace(target.trace_id, flush=True)
        if trace is None:
            raise ValueError(
                "評価Traceを取得できません: "
                f"label={target.label}, trace_id={target.trace_id}"
            )
        source_traces.append(trace)
        source_payloads.append(extract_article_payload(trace))

    presentation_plan = [
        {
            "label": target.label,
            "evaluation_run_id": target.evaluation_run_id,
            "source_trace_id": target.trace_id,
            "article_variant": target.article_variant,
            "article_path": payload.article_path,
            "article_sha256": payload.article_sha256,
            "article_chars": len(payload.article_markdown),
            "response_preview_mode": "raw-markdown-full-text",
        }
        for target, payload in zip(targets, source_payloads, strict=True)
    ]

    base = {
        "workflow_version": WORKFLOW_VERSION,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "experiment_id": experiment.experiment_id,
        "queue_name": queue_name,
        "targets": [asdict(target) for target in targets],
        "presentation_plan": presentation_plan,
        "question_contracts": [
            expected_schema_contract(spec)
            for spec in question_specs()
        ],
    }
    if dry_run:
        return {
            **base,
            "mode": "dry-run",
            "writes": [
                "create_or_reuse_markdown_review_traces",
                "copy_judge_feedback_with_provenance",
                "create_or_reuse_review_queue_v2",
                "add_review_trace_items",
            ],
        }

    schemas, created_schema_names = ensure_label_schemas(
        experiment_id=experiment.experiment_id,
    )
    schema_ids = [str(schema.schema_id) for schema in schemas]
    if any(schema_id == "None" for schema_id in schema_ids):
        raise ValueError("作成済みLabel Schemaにschema_idがありません。")

    presentations = [
        ensure_review_presentation(
            mlflow_module=mlflow,
            experiment_id=experiment.experiment_id,
            target=target,
            source_trace=source_trace,
        )
        for target, source_trace in zip(targets, source_traces, strict=True)
    ]
    review_trace_ids = [item.review_trace_id for item in presentations]
    if len(set(review_trace_ids)) != len(review_trace_ids):
        raise ValueError(
            "BaselineとCandidateのReview表示Traceが同一です。"
        )

    queue, queue_created = ensure_review_queue(
        experiment_id=experiment.experiment_id,
        queue_name=queue_name,
        schema_ids=schema_ids,
    )
    items = add_items_to_review_queue(
        queue.queue_id,
        item_ids=review_trace_ids,
    )

    return {
        **base,
        "mode": "apply",
        "queue": {
            "queue_id": queue.queue_id,
            "queue_name": queue.name,
            "queue_type": _enum_value(queue.queue_type),
            "created": queue_created,
            "users": list(queue.users),
            "schema_ids": list(queue.schema_ids),
        },
        "schemas": [
            {
                "schema_id": schema.schema_id,
                **schema_contract(schema),
                "created": schema.name in created_schema_names,
            }
            for schema in schemas
        ],
        "presentations": [asdict(item) for item in presentations],
        "items": [review_item_dict(item) for item in items],
        "review_url": (
            f"{tracking_uri.rstrip('/')}/#/experiments/"
            f"{experiment.experiment_id}/review-queue"
        ),
    }


def review_item_dict(item: Any) -> dict[str, Any]:
    return {
        "queue_id": item.queue_id,
        "item_id": item.item_id,
        "item_type": _enum_value(item.item_type),
        "status": _enum_value(item.status),
        "completed_by": item.completed_by,
        "completed_time_ms": item.completed_time_ms,
    }


def latest_assessments(assessments: Iterable[Any]) -> dict[str, Any]:
    """Override後に有効な最新Assessmentをname単位で取得する。"""

    latest: dict[str, Any] = {}
    timestamps: dict[str, int] = {}
    for assessment in assessments:
        if getattr(assessment, "valid", True) is False:
            continue
        if not hasattr(assessment, "value"):
            continue
        name = str(assessment.name)
        timestamp = int(
            getattr(assessment, "last_update_time_ms", 0) or 0
        )
        if name not in latest or timestamp >= timestamps[name]:
            latest[name] = assessment
            timestamps[name] = timestamp
    return latest


def latest_assessment_values(assessments: Iterable[Any]) -> dict[str, Any]:
    """最新Assessmentの値だけをname単位で取得する。"""

    return {
        name: assessment.value
        for name, assessment in latest_assessments(assessments).items()
    }


def _score(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}が数値ではありません: {value!r}") from exc
    if not 1.0 <= result <= 5.0:
        raise ValueError(f"{name}は1〜5である必要があります: {result}")
    return result


def build_validation_report(
    *,
    queue: Any,
    items: Sequence[Any],
    traces_by_id: dict[str, Any],
    expected_trace_ids: set[str] | None = None,
    schema_contracts_valid: bool = True,
) -> dict[str, Any]:
    """Queue状態と人手・Judge採点の対応を検証する。"""

    expected = expected_trace_ids or set(traces_by_id)
    item_ids = {item.item_id for item in items}
    queue_has_expected_items = expected.issubset(item_ids)
    rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []

    for item in items:
        trace = traces_by_id.get(item.item_id)
        if trace is None:
            rows.append(
                {
                    "trace_id": item.item_id,
                    "status": _enum_value(item.status),
                    "trace_retrievable": False,
                    "markdown_preview_valid": False,
                    "human_scores_complete": False,
                    "judge_scores_complete": False,
                    "publishable_answered": False,
                    "human_sources_valid": False,
                    "missing_assessments": [],
                }
            )
            continue

        latest = latest_assessments(trace.info.assessments)
        preview_valid = raw_markdown_preview_shape_valid(
            getattr(trace.info, "response_preview", None)
        )
        values = {
            name: assessment.value
            for name, assessment in latest.items()
        }
        missing_human = [
            HUMAN_SCORE_NAMES[axis]
            for axis in JUDGE_AXES
            if HUMAN_SCORE_NAMES[axis] not in values
        ]
        missing_judge = [
            axis for axis in JUDGE_AXES if axis not in values
        ]
        publishable_answered = PUBLISHABLE_NAME in values
        human_assessment_names = [
            *HUMAN_SCORE_NAMES.values(),
            PUBLISHABLE_NAME,
        ]
        human_sources_valid = all(
            getattr(
                getattr(latest[name], "source", None),
                "source_type",
                None,
            ) == "HUMAN"
            for name in human_assessment_names
            if name in latest
        ) and all(name in latest for name in human_assessment_names)

        row = {
            "trace_id": item.item_id,
            "status": _enum_value(item.status),
            "completed_by": item.completed_by,
            "trace_retrievable": True,
            "markdown_preview_valid": preview_valid,
            "human_scores_complete": not missing_human,
            "judge_scores_complete": not missing_judge,
            "publishable_answered": publishable_answered,
            "human_sources_valid": human_sources_valid,
            "publishable": values.get(PUBLISHABLE_NAME),
            "review_notes": values.get(NOTES_NAME),
            "missing_assessments": missing_human + missing_judge,
            "scores": {},
        }

        if not missing_human and not missing_judge:
            for axis in JUDGE_AXES:
                human = _score(
                    values[HUMAN_SCORE_NAMES[axis]],
                    name=HUMAN_SCORE_NAMES[axis],
                )
                judge = _score(values[axis], name=axis)
                difference = human - judge
                comparison = {
                    "trace_id": item.item_id,
                    "axis": axis,
                    "human": human,
                    "judge": judge,
                    "difference": round(difference, 6),
                    "absolute_difference": round(abs(difference), 6),
                }
                comparisons.append(comparison)
                row["scores"][axis] = comparison

        rows.append(row)

    status_counts = {"pending": 0, "complete": 0, "declined": 0}
    for item in items:
        status = _enum_value(item.status)
        status_counts[status] = status_counts.get(status, 0) + 1

    expected_rows = [
        row for row in rows if row["trace_id"] in expected
    ]
    capture_complete = (
        {row["trace_id"] for row in expected_rows} == expected
        and all(
            row["trace_retrievable"]
            and row["markdown_preview_valid"]
            and row["human_scores_complete"]
            and row["judge_scores_complete"]
            and row["publishable_answered"]
            and row["human_sources_valid"]
            for row in expected_rows
        )
    )
    expected_items_complete = all(
        _enum_value(item.status) == "complete"
        for item in items
        if item.item_id in expected
    ) and len(
        {item.item_id for item in items if item.item_id in expected}
    ) == len(expected)
    workflow_complete = (
        queue_has_expected_items
        and expected_items_complete
        and capture_complete
    )

    if comparisons:
        abs_differences = [
            item["absolute_difference"] for item in comparisons
        ]
        exact_count = sum(value == 0 for value in abs_differences)
        within_one_count = sum(value <= 1 for value in abs_differences)
        alignment = {
            "comparison_count": len(comparisons),
            "mean_absolute_error": round(
                sum(abs_differences) / len(abs_differences), 6
            ),
            "exact_agreement_rate": round(
                exact_count / len(abs_differences), 6
            ),
            "within_one_agreement_rate": round(
                within_one_count / len(abs_differences), 6
            ),
            "interpretation": (
                "Judge校正の参考値です。Review機能の成否判定には使いません。"
            ),
        }
    else:
        alignment = {
            "comparison_count": 0,
            "mean_absolute_error": None,
            "exact_agreement_rate": None,
            "within_one_agreement_rate": None,
            "interpretation": "人手採点完了後に算出されます。",
        }

    setup_effective = (
        queue_has_expected_items
        and len(getattr(queue, "schema_ids", [])) == len(question_specs())
        and schema_contracts_valid
        and all(
            row["trace_retrievable"] and row["markdown_preview_valid"]
            for row in rows
        )
    )
    if workflow_complete:
        validation_status = "validated"
    elif setup_effective:
        validation_status = "ready_for_human_review"
    else:
        validation_status = "invalid_setup"

    return {
        "workflow_version": WORKFLOW_VERSION,
        "queue": {
            "queue_id": queue.queue_id,
            "queue_name": queue.name,
            "schema_count": len(queue.schema_ids),
            "item_count": len(items),
            "status_counts": status_counts,
        },
        "validation_status": validation_status,
        "effectiveness": {
            "setup_effective": setup_effective,
            "human_assessment_capture_effective": capture_complete,
            "workflow_completion_effective": workflow_complete,
            "queue_has_expected_items": queue_has_expected_items,
            "schema_contracts_valid": schema_contracts_valid,
            "markdown_previews_valid": all(
                row["markdown_preview_valid"] for row in rows
            ),
            "expected_items_complete": expected_items_complete,
        },
        "trace_reviews": rows,
        "judge_alignment": alignment,
    }


def validate_review_workflow(
    *,
    tracking_uri: str = TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
    queue_name: str = QUEUE_NAME,
    expected_evaluation_run_ids: Sequence[str] = (
        BASELINE_EVALUATION_RUN_ID,
        ADOPTED_EVALUATION_RUN_ID,
    ),
) -> dict[str, Any]:
    """MLflowからQueue、Trace、Assessmentを読み、検証結果を返す。"""

    import mlflow
    from mlflow.genai.label_schemas import list_label_schemas
    from mlflow.genai.review_queues import (
        get_review_queue,
        list_review_queue_items,
    )
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    experiment = MlflowClient().get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow Experimentがありません: {experiment_name}")

    queue = get_review_queue(
        name=queue_name,
        experiment_id=experiment.experiment_id,
    )
    items = list(
        list_review_queue_items(
            queue.queue_id,
            max_results=100,
        )
    )
    all_schemas = list(
        list_label_schemas(
            experiment_id=experiment.experiment_id,
            max_results=100,
        )
    )
    selected_schemas = {
        str(schema.schema_id): schema
        for schema in all_schemas
        if str(schema.schema_id) in set(queue.schema_ids)
    }
    actual_contracts = {
        schema.name: schema_contract(schema)
        for schema in selected_schemas.values()
    }
    expected_contracts = {
        spec.name: expected_schema_contract(spec)
        for spec in question_specs()
    }
    schema_contracts_valid = (
        len(selected_schemas) == len(queue.schema_ids)
        and actual_contracts == expected_contracts
    )
    traces_by_id = {
        item.item_id: mlflow.get_trace(item.item_id)
        for item in items
    }
    source_targets = [
        resolve_single_trace(
            mlflow.search_traces,
            experiment_id=experiment.experiment_id,
            evaluation_run_id=run_id,
            label=run_id,
        )
        for run_id in expected_evaluation_run_ids
    ]
    expected_review_traces = [
        find_review_presentation_trace(
            mlflow.search_traces,
            experiment_id=experiment.experiment_id,
            source_trace_id=target.trace_id,
        )
        for target in source_targets
    ]
    missing_presentations = [
        target.trace_id
        for target, review_trace in zip(
            source_targets,
            expected_review_traces,
            strict=True,
        )
        if review_trace is None
    ]
    if missing_presentations:
        raise ValueError(
            "Review表示専用Traceがありません。先にsetupを実行してください: "
            f"source_trace_ids={missing_presentations}"
        )
    expected_trace_ids = {
        trace.info.trace_id
        for trace in expected_review_traces
        if trace is not None
    }

    report = build_validation_report(
        queue=queue,
        items=items,
        traces_by_id=traces_by_id,
        expected_trace_ids=expected_trace_ids,
        schema_contracts_valid=schema_contracts_valid,
    )
    report.update(
        {
            "tracking_uri": tracking_uri,
            "experiment_name": experiment_name,
            "experiment_id": experiment.experiment_id,
            "expected_evaluation_run_ids": list(
                expected_evaluation_run_ids
            ),
            "source_evaluation_trace_ids": sorted(
                target.trace_id for target in source_targets
            ),
            "expected_trace_ids": sorted(expected_trace_ids),
            "review_url": (
                f"{tracking_uri.rstrip('/')}/#/experiments/"
                f"{experiment.experiment_id}/review-queue"
            ),
        }
    )
    return report
