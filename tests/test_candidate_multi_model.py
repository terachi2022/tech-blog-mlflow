import json
import tempfile
import unittest
from pathlib import Path

from evaluation.evaluate_candidate_dual import evaluation_commands
from tech_blog_mlflow.article_reviewer import ArticleReviewResult, ArticleReviewer
from tech_blog_mlflow.candidate_models import GENERATOR, INDEPENDENT_JUDGE, PRIMARY_JUDGE, REVIEWER, model_manifest
from tech_blog_mlflow.generate_candidate import extract_final_channel, render_prompt


class CandidateModelContractTest(unittest.TestCase):
    def test_roles_and_models_are_separated(self) -> None:
        self.assertIn("GPT-OSS-Swallow-120B", GENERATOR.model_id)
        self.assertEqual(GENERATOR.max_tokens, 10000)
        self.assertIn("Qwen3.6-27B", REVIEWER.model_id)
        self.assertEqual(PRIMARY_JUDGE.model_id, REVIEWER.model_id)
        self.assertNotEqual(PRIMARY_JUDGE.model_id, INDEPENDENT_JUDGE.model_id)
        self.assertIn("gemma-3", INDEPENDENT_JUDGE.model_id)

    def test_manifest_marks_independent_evaluation(self) -> None:
        self.assertTrue(model_manifest()["independent_evaluation"])

    def test_dual_commands_have_distinct_roles(self) -> None:
        commands = evaluation_commands(article=Path("article.md"), source_run_id="run-1", variant="candidate")
        self.assertEqual(len(commands), 2)
        self.assertIn("primary", commands[0])
        self.assertIn("independent", commands[1])
        self.assertIn(PRIMARY_JUDGE.model_id, commands[0])
        self.assertIn(INDEPENDENT_JUDGE.model_id, commands[1])


class CandidatePromptAndReviewTest(unittest.TestCase):
    def test_generation_prompt_is_fully_rendered(self) -> None:
        rendered = render_prompt(Path("prompts/article_generation_v3_5_2.md"), "テストテーマ")
        self.assertIn("テストテーマ", rendered)
        self.assertNotIn("{{", rendered)

    def test_harmony_analysis_is_removed(self) -> None:
        raw = "<|channel|>analysis<|message|>secret<|end|><|channel|>final<|message|># 記事<|return|>"
        self.assertEqual(extract_final_channel(raw), "# 記事")

    def test_missing_harmony_final_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            extract_final_channel("<|channel|>analysis<|message|>unfinished")

    def test_review_json_is_strictly_validated(self) -> None:
        payload = {
            "technical_errors": [], "unsupported_claims": [], "citation_issues": [],
            "reproducibility_issues": [], "readability_issues": [], "required_changes": [],
            "summary": "問題なし", "revised_article": "# 記事\n本文",
        }
        raw = "説明\n```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"
        result = ArticleReviewResult.model_validate(ArticleReviewer.extract_json(raw))
        self.assertEqual(result.summary, "問題なし")

    def test_review_json_rejects_unknown_fields(self) -> None:
        with self.assertRaises(Exception):
            ArticleReviewResult.model_validate({
                "technical_errors": [], "unsupported_claims": [], "citation_issues": [],
                "reproducibility_issues": [], "readability_issues": [], "required_changes": [],
                "summary": "x", "revised_article": "x", "unknown": True,
            })


if __name__ == "__main__":
    unittest.main()
