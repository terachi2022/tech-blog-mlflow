"""Candidate記事をQwen3.6でレビュー・校正し、別Runへ保存する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import mlflow

from tech_blog_mlflow.article_reviewer import ArticleReviewer
from tech_blog_mlflow.article_v3_checks import ARTICLE_MAX_CHARS, ARTICLE_MIN_CHARS, article_checks
from tech_blog_mlflow.candidate_models import PIPELINE_VERSION, REVIEWER


PROMPT = Path("prompts/article_review_qwen3_6_v1.md")
IMMUTABLE_EVIDENCE = (
    "Apple M5 Max",
    "macOS 26.5.1",
    "Python | 3.14.6",
    "MLflow | 3.15.1",
    "MLX-LM | 0.31.3",
    "e251b8dae8f04d2fb22e68f1ae6fa41e",
    "5e2866776b564b4aa28b933f77fe5b51",
    "bded3f7711c04701b50ec83d59b52b3e",
    "20b1a60a129f4e77a136d844f799af5c",
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--article", type=Path, required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--model", default=REVIEWER.model_id)
    parser.add_argument("--prompt", type=Path, default=PROMPT)
    parser.add_argument("--max-tokens", type=int, default=REVIEWER.max_tokens)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    article = args.article.read_text(encoding="utf-8")
    plan = {
        "pipeline_version": PIPELINE_VERSION, "article": str(args.article),
        "article_sha256": digest(article), "source_run_id": args.source_run_id,
        "model": args.model, "prompt": str(args.prompt), "max_tokens": args.max_tokens,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("tech-blog-generation")
    reviewer = ArticleReviewer(model_id=args.model, prompt_path=args.prompt, max_tokens=args.max_tokens)
    result = reviewer.review(article)
    revised = result.revised_article.rstrip() + "\n"
    checks = article_checks(revised, "")
    checks["article_length_in_range"] = ARTICLE_MIN_CHARS <= len(revised) <= ARTICLE_MAX_CHARS
    checks["immutable_evidence_preserved"] = all(
        value in revised for value in IMMUTABLE_EVIDENCE
    )
    checks["substantive_content_preserved"] = len(revised) >= int(len(article) * 0.9)
    failed_checks = [name for name, passed in checks.items() if not passed]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("review_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    article_path = Path("articles") / f"candidate_reviewed_{timestamp}.md"
    article_path.write_text(revised, encoding="utf-8")

    with mlflow.start_run(run_name="candidate-qwen3.6-review-v1") as run:
        payload = {
            **plan, "review_run_id": run.info.run_id,
            "revised_article_path": str(article_path),
            "revised_article_sha256": digest(revised),
            "review": result.model_dump(exclude={"revised_article"}),
            "prechecks": checks,
            "all_prechecks_passed": not failed_checks,
            "failed_prechecks": failed_checks,
            "model_load_time_sec": round(reviewer.load_elapsed_sec, 3),
            "generation_time_sec": round(reviewer.generation_elapsed_sec, 3),
            "raw_response": reviewer.raw_response,
        }
        result_path = output_dir / f"candidate_review_{timestamp}_{run.info.run_id}.json"
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        mlflow.log_params({
            "pipeline_version": PIPELINE_VERSION, "reviewer_model": args.model,
            "review_prompt_version": "article-review-qwen3.6-v1", "source_run_id": args.source_run_id,
            "source_article_sha256": digest(article), "article_sha256": digest(revised),
            "prompt_version": "article-v3.5.2",
        })
        mlflow.set_tags({"stage": "candidate-review", "article_variant": "candidate-reviewed"})
        mlflow.log_metrics({"review_model_load_time_sec": reviewer.load_elapsed_sec, "review_generation_time_sec": reviewer.generation_elapsed_sec})
        mlflow.log_metric("review_precheck_all_passed", int(not failed_checks))
        mlflow.log_artifact(str(article_path), artifact_path="reviewed_article")
        mlflow.log_artifact(str(result_path), artifact_path="review_metadata")
        print("Review Run :", run.info.run_id)
        print("Article    :", article_path)
        print("Metadata   :", result_path)
        print("Prechecks  :", checks)


if __name__ == "__main__":
    main()
