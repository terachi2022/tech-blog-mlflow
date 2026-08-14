"""Prompt-v3生成記事の決定論的な事前検査。"""

from __future__ import annotations

import re


BASELINE_RUN_ID = (
    "e251b8dae8f04d2fb22e68f1ae6fa41e"
)

PROMPT_V2_RUN_ID = (
    "5e2866776b564b4aa28b933f77fe5b51"
)

CONTROLLED_PROMPT_SHA256 = (
    "888648d67bbdd6aa5f1e1a6ca34ced8cf0cc1f7b858af7a97df2f4762d1448f3"
)

INSTRUCTION_LEAK_PATTERN = re.compile(
    r"(?:説明|提示|掲載|明記|出力)"
    r"してください"
)

REQUIRED_H2 = (
    "結論",
    "この記事で実施すること",
    "前提条件",
    "MLflowの用語と保存先",
    "環境構築",
    "実行可能なtrain.py",
    "train.pyを実行する",
    "MLflow UIで確認・比較する",
    "実測した生成Runの制御比較",
    "実測した評価Runの比較",
    "実際に検出した失敗と根本原因",
    "Apple Silicon環境で得られた知見",
    "エラー別の切り分け",
    "制約と注意点",
    "まとめ",
    "参考資料",
)

REQUIRED_REFERENCE_URLS = (
    "https://docs.astral.sh/uv/getting-started/installation/",
    "https://mlflow.org/docs/latest/ml/tracking/",
    "https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/",
    "https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html#mlflow.set_tracking_uri",
    "https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html#mlflow.sklearn.log_model",
    "https://mlflow.org/docs/latest/ml/model-registry/",
    "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html",
    "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html",
    "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
    "https://github.com/ml-explore/mlx-lm",
    "https://github.com/ml-explore/mlx",
)

UV_ADD_PATTERN = re.compile(
    r"^\s*uv\s+add\b.*$",
    flags=re.MULTILINE,
)


def article_checks(
    article: str,
) -> dict[str, bool]:
    """v3記事が必須条件を満たすか返す。"""
    markdown_structure = re.sub(
        r"```.*?```",
        "",
        article,
        flags=re.DOTALL,
    )
    prose_for_leak_check = markdown_structure
    prose_for_leak_check = re.sub(
        r"`[^`]*`",
        "",
        prose_for_leak_check,
    )

    instruction_leak_lines = [
        line
        for line in prose_for_leak_check.splitlines()
        if INSTRUCTION_LEAK_PATTERN.search(line)
        and not re.match(
            r"^\s*-\s*(?:現象|確認(?:Command)?):",
            line,
        )
    ]

    uv_add_lines = (
        UV_ADD_PATTERN.findall(article)
    )

    code_fence_count = article.count(
        "```"
    )

    return {
        "starts_with_h1": (
            article.startswith(
                "# MLflowを使って機械学習の実験を管理する方法"
            )
        ),
        "single_h1": (
            len(
                re.findall(
                    r"^#\s+.+$",
                    markdown_structure,
                    flags=re.MULTILINE,
                )
            )
            == 1
        ),
        "has_prerequisites": (
            "## 前提条件" in article
        ),
        "has_all_required_h2": all(
            f"## {heading}" in article
            for heading in REQUIRED_H2
        ),
        "has_failure_analysis": (
            "## 実際に検出した失敗と根本原因"
            in article
        ),
        "has_observed_results": (
            "## 実測した生成Runの制御比較"
            in article
            and "## 実測した評価Runの比較"
            in article
        ),
        "has_apple_silicon_insight": (
            "## Apple Silicon環境で得られた知見"
            in article
        ),
        "has_references": (
            "## 参考資料" in article
        ),
        "has_summary": (
            "## まとめ" in article
        ),
        "has_train_command": (
            "uv run python train.py"
            in article
        ),
        "has_accuracy_print": (
            re.search(
                r"print\s*\(\s*f?[\"']accuracy",
                article,
                flags=re.IGNORECASE,
            )
            is not None
        ),
        "has_complete_train_code": all(
            fragment in article
            for fragment in (
                "import mlflow",
                "import mlflow.sklearn",
                "load_iris(",
                "train_test_split(",
                "LogisticRegression(",
                "mlflow.sklearn.log_model(",
                "MlflowException",
            )
        ),
        "has_failure_cases": (
            "### 出力上限による記事切断"
            in article
            and "### 依存追加Commandの重複"
            in article
            and "### 執筆指示の混入"
            in article
        ),
        "has_comparison_run_ids": (
            BASELINE_RUN_ID in article
            and PROMPT_V2_RUN_ID in article
            and CONTROLLED_PROMPT_SHA256
            in article
        ),
        "single_uv_add_command": (
            len(uv_add_lines) == 1
        ),
        "no_instruction_leakage": (
            not instruction_leak_lines
        ),
        "balanced_code_fences": (
            code_fence_count % 2 == 0
        ),
        "has_no_thinking_output": (
            "<think>" not in article
            and "</think>" not in article
        ),
        "has_all_reference_urls": all(
            url in article
            for url in REQUIRED_REFERENCE_URLS
        ),
        "has_specific_error_messages": all(
            message in article
            for message in (
                "Connection refused",
                "Address already in use",
                "ModuleNotFoundError: No module named 'mlflow'",
            )
        ),
        "has_controlled_comparison_evidence": all(
            value in article
            for value in (
                "2,048",
                "3,072",
                "2,586",
                "486",
                "20.317",
                "25.656",
            )
        ),
    }


def failed_article_checks(
    article: str,
) -> list[str]:
    """失敗した事前検査名を返す。"""
    return [
        name
        for name, passed in (
            article_checks(article).items()
        )
        if not passed
    ]
