"""Qwen3.6を使う記事レビュー・校正処理。"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ArticleReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    technical_errors: list[str]
    unsupported_claims: list[str]
    citation_issues: list[str]
    reproducibility_issues: list[str]
    readability_issues: list[str]
    required_changes: list[str]
    summary: str = Field(min_length=1)
    revised_article: str = Field(min_length=1)


class ArticleReviewer:
    def __init__(self, *, model_id: str, prompt_path: Path, max_tokens: int) -> None:
        self.model_id = model_id
        self.prompt_path = prompt_path
        self.max_tokens = max_tokens
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._lock = threading.Lock()
        self.load_elapsed_sec = 0.0
        self.generation_elapsed_sec = 0.0
        self.raw_response = ""

    @staticmethod
    def extract_json(raw: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        for position, character in enumerate(raw):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(raw[position:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise ValueError("Reviewer出力からJSON objectを抽出できません。")

    def load_model(self) -> None:
        if self._model is not None:
            return
        from mlx_lm import load

        started = time.perf_counter()
        self._model, self._tokenizer = load(self.model_id)
        self.load_elapsed_sec = time.perf_counter() - started

    def review(self, article: str) -> ArticleReviewResult:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        if not article.strip():
            raise ValueError("記事が空です。")
        template = self.prompt_path.read_text(encoding="utf-8")
        if "{{ARTICLE}}" not in template:
            raise ValueError("Review Promptに{{ARTICLE}}がありません。")
        prompt = template.replace("{{ARTICLE}}", article)

        with self._lock:
            self.load_model()
            assert self._model is not None and self._tokenizer is not None
            messages = [{"role": "user", "content": prompt}]
            formatted = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            started = time.perf_counter()
            self.raw_response = generate(
                self._model,
                self._tokenizer,
                prompt=formatted,
                max_tokens=self.max_tokens,
                sampler=make_sampler(temp=0.0),
                verbose=False,
            )
            self.generation_elapsed_sec = time.perf_counter() - started

        return ArticleReviewResult.model_validate(self.extract_json(self.raw_response))
