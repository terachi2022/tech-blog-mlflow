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
    text_sha256,
)
from tech_blog_mlflow.article_v3_checks import (  # noqa: E402
    ARTICLE_MAX_CHARS,
    ARTICLE_MIN_CHARS,
    article_checks,
)


PROMPT_PATH = (
    PROJECT_ROOT
    / "prompts"
    / "article_generation_v3_5_2.md"
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
        self.assertIn(
            "bded3f7711c04701b50ec83d59b52b3e",
            prompt,
        )
        self.assertIn(
            "20b1a60a129f4e77a136d844f799af5c",
            prompt,
        )
        self.assertIn(
            "7a8494145b33964db7c6cfa8c1f8567d58db1174ea345e467f8ab9adad6f9042",
            prompt,
        )

    def test_prompt_is_compact_and_has_no_polite_commands(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(
            encoding="utf-8"
        )

        self.assertLess(len(prompt), 11000)
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
            "argparse.ArgumentParser",
            "Connection refused",
            "Address already in use",
            "ModuleNotFoundError",
            "lsof -nP -iTCP:5000 -sTCP:LISTEN",
            "System Metrics Loggingを有効化していない",
            "mlflow.set_tracking_uri API",
            "mlflow.sklearn.log_model API",
            "MLX-LM公式Repository",
            "Technical Accuracy 4.75 / 4.50",
            "Original Value 2.50 / 2.50",
            "with mlflow.start_run() as run:",
            "run.info.run_id",
            "`pass`、省略記号、未定義変数は禁止",
            "同じURLは記事中で1回だけ使用",
            "Iris分類結果ではありません",
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
            'PROMPT_VERSION: Final = "article-v3.5.2"',
            '"generation-v3.5.2"',
            "MAX_TOKENS - 64",
            '"generator-prompt-only"',
            "PREVIOUS_MAX_TOKENS: Final = 4096",
            '"max_tokens_changed": False',
        )

        for fragment in expected_fragments:
            self.assertIn(fragment, source)


class ArticleChecksTest(unittest.TestCase):
    def valid_article(self) -> str:
        article = """# MLflowを使って機械学習の実験を管理する方法

## 結論

MLflowでRunを管理します。

## この記事で実施すること

実験を記録します。

## 前提条件

```bash
mkdir -p "$HOME/dev"
cd "$HOME/dev"
uv init my_mlflow_project
cd my_mlflow_project
uv add "mlflow==3.15.1" scikit-learn
uv run mlflow server --backend-store-uri sqlite:///mlflow.db
curl http://127.0.0.1:5000/health
uv run python train.py --max-iter 100
uv run python train.py --max-iter 200
```

## MLflowの用語と保存先

実験管理Toolです。
Irisはscikit-learn同梱のDatasetです。
train/test分割は学習用と評価用へDataを分けます。
Logistic Regressionは今回使う分類Algorithmです。
accuracyは正解率を表すMetricです。

## 環境構築

環境を構築します。

## 実行可能なtrain.py

```python
import argparse
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=200)
    return parser.parse_args()

def main():
    args = parse_args()
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Iris Classification Experiment")
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = LogisticRegression(max_iter=args.max_iter)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    with mlflow.start_run() as run:
        mlflow.log_params({"max_iter": args.max_iter})
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(sk_model=model, name="iris_model")
        print("Run ID:", run.info.run_id)
        print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
```

## train.pyを実行する

実行します。

## MLflow UIで確認・比較する

1. Iris Classification Experimentを開く
2. Parametersを確認する
3. Metricsを確認する
4. Artifactsを確認する
5. 2 Runを選択してCompareでRun Durationを確認する

## 実測した生成Runの制御比較

`e251b8dae8f04d2fb22e68f1ae6fa41e`
`5e2866776b564b4aa28b933f77fe5b51`
`888648d67bbdd6aa5f1e1a6ca34ced8cf0cc1f7b858af7a97df2f4762d1448f3`
`bded3f7711c04701b50ec83d59b52b3e`
`20b1a60a129f4e77a136d844f799af5c`
`7a8494145b33964db7c6cfa8c1f8567d58db1174ea345e467f8ab9adad6f9042`

2,048から3,072へ変更し、2,586 Tokenで完了しました。
上限まで486 Tokenの余裕がありました。
生成時間は20.317秒から25.656秒になりました。
別のRunは3,594 Token、36.993秒でした。

## 実測した評価Runの比較

評価結果は3.742から3.392へ変化しました。

## 実際に検出した失敗と根本原因

### 出力上限による記事切断

- 現象: 末尾が切断
- 確認: Token数を比較
- 根本原因: 当該Runでは上限不足
- 修正: 上限を増加
- 再検証: 事前検査PASS

### 依存追加Commandの重複

- 現象: 重複
- 確認: rgで確認
- 根本原因: 同じ操作を重複出力
- 修正: 1行へ統合
- 再検証: 1行だけ存在

### 執筆指示の混入

- 現象: 執筆指示が残存
- 確認: rgで確認
- 根本原因: Prompt文が転記
- 修正: 事前検査を追加
- 再検証: leakage検査PASS

```bash
rg -n '説明してください' article.md
```

## Apple Silicon環境で得られた知見

M5 Maxで計測しました。

## エラー別の切り分け

### Connection refused

- 表示例: `Connection refused`
- 確認: `curl -i http://127.0.0.1:5000/health`
- 原因候補: Server未起動
- 対処: Serverを起動
- 再確認: Status 200を確認

### Address already in use

- 表示例: `[Errno 48] Address already in use`
- 確認: `lsof -nP -iTCP:5000 -sTCP:LISTEN`
- 原因候補: Port使用中
- 対処: Processを確認
- 再確認: Serverを再起動

### ModuleNotFoundError

- 表示例: `ModuleNotFoundError: No module named 'mlflow'`
- 確認: `uv run python -c 'import mlflow'`
- 原因候補: 依存未同期
- 対処: `uv sync`
- 再確認: import mlflowが成功

### UIにRunが表示されない

- 表示例: UIにRunがない
- 確認: `mlflow.get_tracking_uri()`
- 原因候補: URI不一致
- 対処: URIを統一
- 再確認: Iris Classification Experimentを開く

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

        if len(article) < ARTICLE_MIN_CHARS:
            padding = "再現条件と確認結果を区別して記録します。"
            required = ARTICLE_MIN_CHARS - len(article)
            filler = (padding * (required // len(padding) + 1))[:required]
            article = article.replace(
                "## 参考資料",
                f"{filler}\n\n## 参考資料",
                1,
            )

        self.assertLessEqual(len(article), ARTICLE_MAX_CHARS)
        return article

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

    def test_article_meta_sentence_fails(
        self,
    ) -> None:
        article = self.valid_article()
        article += "\n次の環境表を掲載します。\n"

        checks = article_checks(article)

        self.assertFalse(
            checks["no_instruction_leakage"]
        )

    def test_incomplete_train_code_fails(
        self,
    ) -> None:
        article = self.valid_article().replace(
            "model.fit(X_train, y_train)",
            "model.predict(X_train)",
        )

        checks = article_checks(article)

        self.assertFalse(
            checks["has_complete_train_code"]
        )

    def test_missing_troubleshooting_detail_fails(
        self,
    ) -> None:
        article = self.valid_article().replace(
            "- 対処: Serverを起動",
            "- 補足: Serverを確認",
            1,
        )

        checks = article_checks(article)

        self.assertFalse(
            checks["has_complete_troubleshooting"]
        )

    def test_unlogged_resource_claim_fails(
        self,
    ) -> None:
        article = self.valid_article().replace(
            "Run Durationを確認する",
            "Run DurationとResource使用量を確認する",
            1,
        )

        checks = article_checks(article)

        self.assertFalse(
            checks["does_not_claim_unlogged_system_metrics"]
        )

    def test_unlogged_resource_exclusion_passes(
        self,
    ) -> None:
        article = self.valid_article().replace(
            "Run Durationを確認する",
            (
                "Run Durationを確認する\n\n"
                "System Metrics Loggingを有効化していないため、"
                "CPUやMemory使用量を比較対象に含めない。"
            ),
            1,
        )

        checks = article_checks(article)

        self.assertTrue(
            checks["does_not_claim_unlogged_system_metrics"]
        )

    def test_out_of_range_article_fails(
        self,
    ) -> None:
        article = self.valid_article() + (
            "追加説明" * ARTICLE_MAX_CHARS
        )

        checks = article_checks(article)

        self.assertFalse(
            checks["article_length_in_range"]
        )

    def test_duplicate_uv_add_fails(
        self,
    ) -> None:
        article = self.valid_article().replace(
            'uv add "mlflow==3.15.1" scikit-learn',
            'uv add "mlflow==3.15.1" scikit-learn\nuv add mlflow',
            1,
        )

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
            "evaluation.evaluate_combined_v2_4",
            command,
        )
        self.assertIn("prompt-v3.5.2", command)
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
                        "article_sha256": "unused",
                        "rendered_prompt_path": str(
                            root / "rendered.md"
                        ),
                        "prompt_sha256": "unused",
                        "formatted_prompt_sha256": "unused",
                        "system_prompt_sha256": "unused",
                        "prompt_version": (
                            EXPECTED_PROMPT_VERSION
                        ),
                        "generation_config_version": (
                            EXPECTED_CONFIG_VERSION
                        ),
                        "generation_parameters": {
                            "max_tokens": 4096
                        },
                        "metrics": {
                            "article_length": 7,
                            "output_tokens": 1,
                        },
                        "prechecks": {},
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

    def test_tampered_article_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            article_path = root / "article.md"
            prompt_path = root / "rendered.md"
            metadata_path = root / "metadata.json"

            article_path.write_text(
                "# original\n",
                encoding="utf-8",
            )
            prompt_path.write_text(
                "prompt\n",
                encoding="utf-8",
            )
            metadata_path.write_text(
                json.dumps(
                    {
                        "run_id": "run-id",
                        "article_path": str(article_path),
                        "article_sha256": text_sha256("# before-tamper\n"),
                        "rendered_prompt_path": str(prompt_path),
                        "prompt_sha256": text_sha256("prompt\n"),
                        "formatted_prompt_sha256": "a" * 64,
                        "system_prompt_sha256": "b" * 64,
                        "prompt_version": EXPECTED_PROMPT_VERSION,
                        "generation_config_version": EXPECTED_CONFIG_VERSION,
                        "generation_parameters": {"max_tokens": 4096},
                        "metrics": {
                            "article_length": len("# original\n"),
                            "output_tokens": 10,
                        },
                        "prechecks": {},
                        "all_prechecks_passed": True,
                        "failed_prechecks": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "記事SHA-256"):
                load_metadata(metadata_path)

    def test_valid_metadata_is_recomputed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            article_path = root / "article.md"
            prompt_path = root / "rendered.md"
            metadata_path = root / "metadata.json"

            article = ArticleChecksTest().valid_article()
            prompt = "structured source facts\n"
            article_path.write_text(article, encoding="utf-8")
            prompt_path.write_text(prompt, encoding="utf-8")

            prechecks = article_checks(article, prompt)
            prechecks["output_tokens_below_safety_limit"] = True
            prechecks["article_length_in_range"] = True
            self.assertTrue(all(prechecks.values()))

            metadata = {
                "run_id": "run-id",
                "article_path": str(article_path),
                "article_sha256": text_sha256(article),
                "rendered_prompt_path": str(prompt_path),
                "prompt_sha256": text_sha256(prompt),
                "formatted_prompt_sha256": "a" * 64,
                "system_prompt_sha256": "b" * 64,
                "prompt_version": EXPECTED_PROMPT_VERSION,
                "generation_config_version": EXPECTED_CONFIG_VERSION,
                "generation_parameters": {"max_tokens": 4096},
                "metrics": {
                    "article_length": len(article),
                    "output_tokens": 1000,
                },
                "prechecks": prechecks,
                "all_prechecks_passed": True,
                "failed_prechecks": [],
            }
            metadata_path.write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            self.assertEqual(
                load_metadata(metadata_path)["run_id"],
                "run-id",
            )


if __name__ == "__main__":
    unittest.main()
