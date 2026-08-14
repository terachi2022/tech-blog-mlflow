import json
import threading
import time
from pathlib import Path
from typing import Any

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from pydantic import ValidationError

from evaluation.judge_schema import (
    ArticleJudgeResult,
)


DEFAULT_JUDGE_MODEL = (
    "mlx-community/"
    "gemma-3-text-27b-it-4bit"
)

DEFAULT_PROMPT_PATH = Path(
    "prompts/article_judge_v1.md"
)


class JudgeOutputError(RuntimeError):
    """
    Judgeの出力を有効なJSONとして処理できない場合のエラー。
    """


class LocalArticleJudge:
    """
    MLX-LM上のGemmaを使って、
    日本語技術記事を6項目で評価する。
    """

    def __init__(
        self,
        model_id: str = DEFAULT_JUDGE_MODEL,
        prompt_path: Path = DEFAULT_PROMPT_PATH,
        max_tokens: int = 1600,
    ) -> None:
        self.model_id = model_id
        self.prompt_path = prompt_path
        self.max_tokens = max_tokens

        self._model: Any | None = None
        self._tokenizer: Any | None = None

        # temp=0.0でgreedy generationにする
        self._sampler = make_sampler(
            temp=0.0
        )

        # 将来複数記事を評価する場合でも、
        # 同じモデルへ同時アクセスしない
        self._generation_lock = (
            threading.Lock()
        )

        self.load_elapsed_sec = 0.0
        self.records: list[dict[str, Any]] = []

    def load_model(self) -> None:
        """
        Judgeモデルをメモリへロードする。
        2回目以降はロードし直さない。
        """
        if (
            self._model is not None
            and self._tokenizer is not None
        ):
            return

        print("=" * 60)
        print("Loading Local LLM Judge")
        print("=" * 60)
        print("Model:", self.model_id)
        print()

        started_at = time.perf_counter()

        self._model, self._tokenizer = load(
            self.model_id
        )

        self.load_elapsed_sec = (
            time.perf_counter()
            - started_at
        )

        print()
        print(
            "Model loaded:",
            f"{self.load_elapsed_sec:.2f} sec",
        )

    def _read_prompt_template(self) -> str:
        """
        外出ししたJudgeプロンプトを読み込む。
        """
        if not self.prompt_path.exists():
            raise FileNotFoundError(
                "Judge promptがありません: "
                f"{self.prompt_path}"
            )

        return self.prompt_path.read_text(
            encoding="utf-8"
        )

    def _render_prompt(
        self,
        article: str,
    ) -> str:
        """
        プロンプト内のARTICLEを評価記事で置き換える。
        """
        template = (
            self._read_prompt_template()
        )

        marker = "{{ARTICLE}}"

        if marker not in template:
            raise ValueError(
                "Judge promptに"
                "{{ARTICLE}}がありません。"
            )

        return template.replace(
            marker,
            article,
        )

    def _apply_chat_template(
        self,
        user_prompt: str,
    ) -> str:
        """
        Gemmaのchat templateを適用する。

        Gemmaはsystem roleではなく、
        userメッセージ1件として評価指示を渡す。
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizerがロードされていません。"
            )

        chat_template = getattr(
            self._tokenizer,
            "chat_template",
            None,
        )

        if chat_template is None:
            return user_prompt

        messages = [
            {
                "role": "user",
                "content": user_prompt,
            }
        ]

        return (
            self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    def _generate(
        self,
        user_prompt: str,
    ) -> tuple[str, float]:
        """
        Gemmaで評価結果を生成する。
        """
        self.load_model()

        if (
            self._model is None
            or self._tokenizer is None
        ):
            raise RuntimeError(
                "Judgeモデルのロードに失敗しました。"
            )

        formatted_prompt = (
            self._apply_chat_template(
                user_prompt
            )
        )

        started_at = time.perf_counter()

        raw_response = generate(
            self._model,
            self._tokenizer,
            prompt=formatted_prompt,
            max_tokens=self.max_tokens,
            sampler=self._sampler,
            verbose=False,
        )

        elapsed_sec = (
            time.perf_counter()
            - started_at
        )

        return raw_response, elapsed_sec

    @staticmethod
    def _extract_json_object(
        raw_response: str,
    ) -> dict[str, Any]:
        """
        モデル出力から最初の有効なJSON objectを抽出する。

        Markdownコードフェンスや前後の説明が
        混入した場合にも対応する。
        """
        text = raw_response.strip()

        decoder = json.JSONDecoder()

        for position, character in enumerate(
            text
        ):
            if character != "{":
                continue

            candidate = text[position:]

            try:
                payload, _ = decoder.raw_decode(
                    candidate
                )
            except json.JSONDecodeError:
                continue

            if isinstance(payload, dict):
                return payload

        raise JudgeOutputError(
            "Judge出力からJSON objectを"
            "抽出できませんでした。"
        )

    @classmethod
    def _validate_response(
        cls,
        raw_response: str,
    ) -> ArticleJudgeResult:
        """
        JSON抽出後、Pydanticで厳密に検証する。
        """
        payload = (
            cls._extract_json_object(
                raw_response
            )
        )

        return (
            ArticleJudgeResult.model_validate(
                payload
            )
        )

    def evaluate(
        self,
        article: str,
    ) -> ArticleJudgeResult:
        """
        記事を評価する。

        1回目の出力がJSONとして不正な場合は、
        修正指示を追加して1回だけ再試行する。
        """
        if not article.strip():
            raise ValueError(
                "評価対象の記事が空です。"
            )

        judge_prompt = self._render_prompt(
            article
        )

        with self._generation_lock:
            first_response, first_elapsed = (
                self._generate(judge_prompt)
            )

            try:
                result = self._validate_response(
                    first_response
                )

                self.records.append(
                    {
                        "attempts": 1,
                        "elapsed_sec": round(
                            first_elapsed,
                            3,
                        ),
                        "raw_response": (
                            first_response
                        ),
                        "validated_result": (
                            result.model_dump()
                        ),
                    }
                )

                return result

            except (
                JudgeOutputError,
                ValidationError,
            ) as first_error:
                repair_prompt = (
                    judge_prompt
                    + "\n\n"
                    + "# JSON修正指示\n\n"
                    + "前回の回答はJSON検証に"
                    + "失敗しました。\n"
                    + "説明やMarkdownを一切付けず、"
                    + "指定された6項目を含む"
                    + "有効なJSONだけを再出力"
                    + "してください。\n\n"
                    + "<invalid_response>\n"
                    + first_response
                    + "\n</invalid_response>"
                )

                second_response, second_elapsed = (
                    self._generate(
                        repair_prompt
                    )
                )

                try:
                    result = (
                        self._validate_response(
                            second_response
                        )
                    )

                except (
                    JudgeOutputError,
                    ValidationError,
                ) as second_error:
                    raise JudgeOutputError(
                        "Judge出力のJSON検証に"
                        "2回連続で失敗しました。\n"
                        f"1回目: {first_error}\n"
                        f"2回目: {second_error}"
                    ) from second_error

                self.records.append(
                    {
                        "attempts": 2,
                        "elapsed_sec": round(
                            first_elapsed
                            + second_elapsed,
                            3,
                        ),
                        "first_validation_error": (
                            str(first_error)
                        ),
                        "first_raw_response": (
                            first_response
                        ),
                        "raw_response": (
                            second_response
                        ),
                        "validated_result": (
                            result.model_dump()
                        ),
                    }
                )

                return result

    @property
    def total_generation_time_sec(
        self,
    ) -> float:
        """
        全評価に使った生成時間の合計。
        """
        return round(
            sum(
                float(
                    record.get(
                        "elapsed_sec",
                        0.0,
                    )
                )
                for record in self.records
            ),
            3,
        )

    def save_records(
        self,
        output_path: Path,
    ) -> None:
        """
        Judgeの生出力と検証済み結果をJSON保存する。
        """
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "judge_model": self.model_id,
            "prompt_path": str(
                self.prompt_path
            ),
            "max_tokens": self.max_tokens,
            "model_load_time_sec": round(
                self.load_elapsed_sec,
                3,
            ),
            "records": self.records,
        }

        output_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
