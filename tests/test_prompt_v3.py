"""Prompt-v3生成フローの回帰テスト。"""

from __future__ import annotations

import json
import sys
import unittest
import re
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR),
)
sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from evaluation.evaluate_prompt_v3 import (  # noqa: E402
    EXPECTED_CONFIG_VERSION,
    EXPECTED_PROMPT_VERSION,
    build_evaluation_command,
    load_metadata,
)
from tech_blog_mlflow.article_v3_checks import (  # noqa: E402
    article_checks,
)


PROMPT_PATH = (
    PROJECT_ROOT
    / "prompts"
    / "article_generation_v3_3.md"
)

GENERATOR_PATH = (
    PROJECT_ROOT
    / "src"
    / "tech_blog_mlflow"
    / "generate_prompt_v3.py"
)


class PromptV3Test(unittest.TestCase):
    def test_prompt_has_expected_placeholders(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(
            encoding="utf-8"
        )

        expected = {
            "{{THEME}}",
            "{{MACOS_VERSION}}",
            "{{PYTHON_VERSION}}",
            "{{MLFLOW_VERSION}}",
            "{{MLX_LM_VERSION}}",
        }

        for marker in expected:
            self.assertIn(marker, prompt)

        actual = set(
            re.findall(
                r"\{\{[A-Z0-9_]+\}\}",
                prompt,
            )
        )

        self.assertEqual(
            actual,
            expected,
        )

    def test_prompt_contains_observed_run_ids(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "e251b8dae8f04d2fb22e68f1ae6fa41e",
            prompt,
        )
        self.assertIn(
            "5e2866776b564b4aa28b933f77fe5b51",
            prompt,
        )
        self.assertIn(
            "888648d67bbdd6aa5f1e1a6ca34ced8cf0cc1f7b858af7a97df2f4762d1448f3",
            prompt,
        )

    def test_prompt_is_compact_and_has_no_polite_commands(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(
            encoding="utf-8"
        )

        self.assertLess(len(prompt), 12000)
        self.assertNotIn(
            "してください",
            prompt,
        )

    def test_prompt_targets_v3_2_judge_gaps(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(
            encoding="utf-8"
        )

        for fragment in (
            "MlflowException",
            "Connection refused",
            "Address already in use",
            "ModuleNotFoundError",
            "mlflow.set_tracking_uri API",
            "mlflow.sklearn.log_model API",
            "MLX-LM公式Repository",
            "Technical Accuracy | 5.00 | 4.75",
            "Original Value | 1.00 | 2.50",
        ):
            self.assertIn(fragment, prompt)

    def test_generation_settings_are_fixed(
        self,
    ) -> None:
        source = GENERATOR_PATH.read_text(
            encoding="utf-8"
        )

        expected_fragments = (
            '"Qwen/Qwen3-8B-MLX-4bit"',
            "MAX_TOKENS: Final = 4096",
            "TEMPERATURE: Final = 0.0",
            "SEED: Final = 42",
            "ENABLE_THINKING: Final = False",
            'PROMPT_VERSION: Final = "article-v3.3"',
            '"generation-v3.4"',
            "MAX_TOKENS - 64",
            '"max-tokens-only"',
            "PREVIOUS_MAX_TOKENS: Final = 3072",
            '"max_tokens_changed": True',
        )

        for fragment in expected_fragments:
            self.assertIn(fragment, source)


class ArticleChecksTest(unittest.TestCase):
    def valid_article(self) -> str:
        return """# MLflowを使って機械学習の実験を管理する方法

## 結論

MLflowでRunを管理します。

## この記事で実施すること

実験を記録します。

## 前提条件

```bash
uv add "mlflow==3.15.1" scikit-learn
uv run python train.py
```

## MLflowの用語と保存先

実験管理Toolです。

## 環境構築

環境を構築します。

## 実行可能なtrain.py

```python
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
# Python内のCommentはMarkdown H1ではありません
X, y = load_iris(return_X_y=True)
train_test_split(X, y)
LogisticRegression(max_iter=200)
mlflow.sklearn.log_model(sk_model=model, name="iris_model")
print("Accuracy:", accuracy)
```

## train.pyを実行する

実行します。

## MLflow UIで確認・比較する

UIを確認します。

## 実測した生成Runの制御比較

`e251b8dae8f04d2fb22e68f1ae6fa41e`
`5e2866776b564b4aa28b933f77fe5b51`
`888648d67bbdd6aa5f1e1a6ca34ced8cf0cc1f7b858af7a97df2f4762d1448f3`

2,048から3,072へ変更し、2,586 Tokenで完了しました。
上限まで486 Tokenの余裕がありました。
生成時間は20.317秒から25.656秒になりました。

## 実測した評価Runの比較

評価結果を比較します。

## 実際に検出した失敗と根本原因

### 出力上限による記事切断

Token上限が原因でした。

### 依存追加Commandの重複

重複を確認しました。

### 執筆指示の混入

過去記事では`説明してください`が残りました。

```bash
rg -n '説明してください' article.md
```

## Apple Silicon環境で得られた知見

M5 Maxで計測しました。

## エラー別の切り分け

`Connection refused`
`[Errno 48] Address already in use`
`ModuleNotFoundError: No module named 'mlflow'`

## 制約と注意点

SQLiteはTutorial向けです。

## まとめ

完了です。

## 参考資料

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [Backend](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/)
- [Tracking URI](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html#mlflow.set_tracking_uri)
- [log_model](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html#mlflow.sklearn.log_model)
- [Registry](https://mlflow.org/docs/latest/ml/model-registry/)
- [Iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
- [Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)
- [MLX](https://github.com/ml-explore/mlx)
"""

    def test_valid_article_passes(
        self,
    ) -> None:
        checks = article_checks(
            self.valid_article()
        )

        self.assertTrue(all(checks.values()))

    def test_instruction_leak_in_prose_fails(
        self,
    ) -> None:
        article = self.valid_article()
        article += (
            "\nこの流れを説明してください。\n"
        )

        checks = article_checks(article)

        self.assertFalse(
            checks["no_instruction_leakage"]
        )

    def test_duplicate_uv_add_fails(
        self,
    ) -> None:
        article = self.valid_article()
        article += "\nuv add mlflow\n"

        checks = article_checks(article)

        self.assertFalse(
            checks["single_uv_add_command"]
        )

    def test_unbalanced_fence_fails(
        self,
    ) -> None:
        article = self.valid_article()
        article += "\n```python\n"

        checks = article_checks(article)

        self.assertFalse(
            checks["balanced_code_fences"]
        )

    def test_python_comment_is_not_counted_as_h1(
        self,
    ) -> None:
        checks = article_checks(
            self.valid_article()
        )

        self.assertTrue(checks["single_h1"])


class EvaluatePromptV3Test(unittest.TestCase):
    def test_build_evaluation_command(
        self,
    ) -> None:
        metadata = {
            "article_path": (
                "articles/prompt_v3_test.md"
            ),
            "run_id": "generation-run-id",
            "generation_config_version": (
                EXPECTED_CONFIG_VERSION
            ),
            "all_prechecks_passed": True,
            "failed_prechecks": [],
        }

        command = build_evaluation_command(
            metadata=metadata,
            run_name="test-run",
        )

        self.assertIn(
            "evaluation.evaluate_combined_v2_2",
            command,
        )
        self.assertIn("prompt-v3.4", command)
        self.assertIn(
            EXPECTED_PROMPT_VERSION,
            command,
        )
        self.assertIn(
            "generation-run-id",
            command,
        )

    def test_failed_prechecks_stop_evaluation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            article_path = root / "article.md"
            metadata_path = root / "metadata.json"

            article_path.write_text(
                "# test\n",
                encoding="utf-8",
            )
            metadata_path.write_text(
                json.dumps(
                    {
                        "run_id": "run-id",
                        "article_path": str(
                            article_path
                        ),
                        "prompt_version": (
                            EXPECTED_PROMPT_VERSION
                        ),
                        "generation_config_version": (
                            EXPECTED_CONFIG_VERSION
                        ),
                        "all_prechecks_passed": (
                            False
                        ),
                        "failed_prechecks": [
                            "balanced_code_fences"
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_metadata(metadata_path)


if __name__ == "__main__":
    unittest.main()
