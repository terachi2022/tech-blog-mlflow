"""Prompt-v3生成記事の決定論的な事前検査。"""

from __future__ import annotations

import ast
import re


BASELINE_RUN_ID = "e251b8dae8f04d2fb22e68f1ae6fa41e"
SECOND_CONTROL_FAILURE_RUN_ID = "bded3f7711c04701b50ec83d59b52b3e"
SECOND_CONTROL_SUCCESS_RUN_ID = "20b1a60a129f4e77a136d844f799af5c"
SECOND_CONTROL_PROMPT_SHA256 = (
    "7a8494145b33964db7c6cfa8c1f8567d58db1174ea345e467f8ab9adad6f9042"
)
PROMPT_V2_RUN_ID = "5e2866776b564b4aa28b933f77fe5b51"
CONTROLLED_PROMPT_SHA256 = (
    "888648d67bbdd6aa5f1e1a6ca34ced8cf0cc1f7b858af7a97df2f4762d1448f3"
)

ARTICLE_MIN_CHARS = 5800
ARTICLE_MAX_CHARS = 6800

PUBLIC_URL_PATTERN = re.compile(r"https?://[^\s)>\"']+")
UV_ADD_PATTERN = re.compile(r"^\s*uv\s+add\b.*$", flags=re.MULTILINE)

# 完成記事に残ってはいけない、執筆作業そのものを説明する表現。
# 「この記事では〜を紹介します」のような通常の導入文は対象外にする。
ARTICLE_META_PATTERN = re.compile(
    r"(?:"
    r"(?:説明|提示|掲載|明記|出力)してください|"
    r"記事内では.*(?:説明|明示|記載|掲載)します|"
    r"完成記事(?:では|に).*(?:説明|明示|記載|掲載|出力)します|"
    r"次の(?:環境表|Pythonコード|各Error|各エラー).*(?:説明|掲載)します|"
    r"(?:手順|説明|判断|知見)も含めます|"
    r"(?:判断|根拠|対応関係)を示します|"
    r"(?:現象|確認|根本原因|修正|再検証).*順で.*説明します|"
    r"(?:Prompt|プロンプト)(?:-[A-Za-z0-9.]+)?記事.*説明します"
    r")"
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

REQUIRED_SHELL_COMMANDS = (
    "uv init",
    "cd ",
    "uv add",
    "uv run mlflow server",
    "--backend-store-uri sqlite:///mlflow.db",
    "curl http://127.0.0.1:5000/health",
    "uv run python train.py",
)

FAILURE_HEADINGS = (
    "出力上限による記事切断",
    "依存追加Commandの重複",
    "執筆指示の混入",
)

TROUBLESHOOTING_REQUIREMENTS = {
    "Connection refused": (
        "表示例",
        "確認",
        "原因候補",
        "対処",
        "再確認",
        "curl -i http://127.0.0.1:5000/health",
        "200",
    ),
    "Address already in use": (
        "表示例",
        "確認",
        "原因候補",
        "対処",
        "再確認",
        "lsof -nP -iTCP:5000 -sTCP:LISTEN",
    ),
    "ModuleNotFoundError": (
        "表示例",
        "確認",
        "原因候補",
        "対処",
        "再確認",
        "uv sync",
        "import mlflow",
    ),
    "UIにRunが表示されない": (
        "表示例",
        "確認",
        "原因候補",
        "対処",
        "再確認",
        "mlflow.get_tracking_uri()",
        "Iris Classification Experiment",
    ),
}


def _strip_fenced_code(article: str) -> str:
    return re.sub(r"```.*?```", "", article, flags=re.DOTALL)


def _fenced_blocks(article: str, language: str) -> list[str]:
    return re.findall(
        rf"```{language}\s*\n(.*?)```",
        article,
        flags=re.DOTALL | re.IGNORECASE,
    )


def _section(article: str, heading: str, level: int = 2) -> str:
    marker = "#" * level
    pattern = re.compile(
        rf"^{re.escape(marker)}\s+{re.escape(heading)}\s*$\n"
        rf"(.*?)(?=^{'#' * level}\s+|\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(article)
    return match.group(1) if match else ""


def _train_code(article: str) -> str:
    return next(
        (
            block
            for block in _fenced_blocks(article, "python")
            if "def main(" in block and "mlflow.start_run(" in block
        ),
        "",
    )


def _call_name(node: ast.Call) -> str:
    parts: list[str] = []
    current: ast.expr = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _train_code_checks(code: str) -> tuple[bool, bool]:
    """train.pyの構文・必須処理と例外処理の妥当性を確認する。"""
    if not code:
        return False, False

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, False

    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    calls = {
        _call_name(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    required_calls = {
        "argparse.ArgumentParser",
        "parser.add_argument",
        "load_iris",
        "train_test_split",
        "LogisticRegression",
        "model.fit",
        "model.score",
        "mlflow.set_tracking_uri",
        "mlflow.set_experiment",
        "mlflow.start_run",
        "mlflow.log_params",
        "mlflow.log_metric",
        "mlflow.sklearn.log_model",
        "print",
    }
    main = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        ),
        None,
    )
    complete = bool(
        "argparse" in imports
        and "mlflow" in imports
        and "mlflow.sklearn" in imports
        and {"load_iris", "LogisticRegression", "train_test_split"}
        <= imported_names
        and required_calls <= calls
        and main is not None
        and any(isinstance(node, ast.If) for node in tree.body)
    )

    if main is None:
        return complete, False

    try_node = next(
        (node for node in main.body if isinstance(node, ast.Try)),
        None,
    )
    if try_node is None:
        # Tutorialで例外処理は必須ではない。存在しない場合は妥当とする。
        return complete, True

    try_calls = {
        _call_name(node)
        for statement in try_node.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
    }
    catches_mlflow = any(
        isinstance(handler.type, ast.Name)
        and handler.type.id == "MlflowException"
        and any(isinstance(node, ast.Raise) for node in ast.walk(handler))
        for handler in try_node.handlers
    )
    tracked_calls = {
        "mlflow.set_tracking_uri",
        "mlflow.set_experiment",
        "mlflow.start_run",
        "mlflow.log_params",
        "mlflow.log_metric",
        "mlflow.sklearn.log_model",
    }
    exception_valid = bool(
        "MlflowException" in imported_names
        and tracked_calls <= try_calls
        and catches_mlflow
    )
    return complete, exception_valid


def _prompt_instruction_echoes(
    article: str,
    rendered_prompt: str | None,
) -> list[str]:
    if not rendered_prompt:
        return []

    def prose_lines(text: str) -> set[str]:
        return {
            re.sub(r"`[^`]*`", "", line).strip()
            for line in _strip_fenced_code(text).splitlines()
            if re.sub(r"`[^`]*`", "", line).strip()
        }

    article_lines = prose_lines(article)
    instruction_verbs = re.compile(
        r"(?:"
        r"(?:説明|掲載|明示|記載|案内|整理|出力|含め|示|述べ)します|"
        r"(?:配置|記述)(?:します|する|。|$)"
        r")"
    )
    return [
        line
        for line in prose_lines(rendered_prompt)
        if len(line) >= 24
        and instruction_verbs.search(line)
        and line in article_lines
    ]


def _failure_analysis_complete(article: str) -> bool:
    required_labels = ("現象", "確認", "根本原因", "修正", "再検証")
    return all(
        all(label in _section(article, heading, level=3) for label in required_labels)
        for heading in FAILURE_HEADINGS
    )


def _claims_unlogged_system_metrics(ui_section: str) -> bool:
    """未記録のSystem Metricsを確認・比較できるとする記述を検出する。"""
    metric_pattern = re.compile(
        r"(?:Resource使用量|CPU使用率|Memory使用量|メモリ使用量)",
        flags=re.IGNORECASE,
    )
    exclusion_pattern = re.compile(
        r"(?:"
        r"比較対象に含めない|比較対象外|対象外|"
        r"記録していない|有効化していない|"
        r"確認できない|比較できない"
        r")"
    )
    return any(
        metric_pattern.search(line) and not exclusion_pattern.search(line)
        for line in ui_section.splitlines()
    )


def _troubleshooting_complete(article: str) -> bool:
    """各Error節に切り分けの5要素と必須Commandがあるか確認する。"""
    return all(
        all(
            value in _section(article, heading, level=3)
            for value in requirements
        )
        for heading, requirements in TROUBLESHOOTING_REQUIREMENTS.items()
    )


def _required_h2_in_order(article: str) -> bool:
    """必須H2が重複せず、指定順で存在するか確認する。"""
    headings = re.findall(
        r"^##\s+(.+?)\s*$",
        _strip_fenced_code(article),
        flags=re.MULTILINE,
    )
    required = list(REQUIRED_H2)
    positions = [
        headings.index(heading)
        for heading in required
        if headings.count(heading) == 1
    ]
    return len(positions) == len(required) and positions == sorted(positions)


def article_checks(
    article: str,
    rendered_prompt: str | None = None,
) -> dict[str, bool]:
    """v3記事が必須条件を満たすか返す。"""
    markdown_structure = _strip_fenced_code(article)
    prose = re.sub(r"`[^`]*`", "", markdown_structure)
    train_code = _train_code(article)
    train_complete, exception_valid = _train_code_checks(train_code)
    shell_code = "\n".join(
        _fenced_blocks(article, "bash") + _fenced_blocks(article, "sh")
    )
    public_urls = [
        url.rstrip(".,;:!?。、）]}>")
        for url in PUBLIC_URL_PATTERN.findall(article)
        if "127.0.0.1" not in url and "localhost" not in url.lower()
    ]
    ui_section = _section(article, "MLflow UIで確認・比較する")
    instruction_echoes = _prompt_instruction_echoes(article, rendered_prompt)

    return {
        "starts_with_h1": article.startswith(
            "# MLflowを使って機械学習の実験を管理する方法"
        ),
        "single_h1": len(
            re.findall(r"^#\s+.+$", markdown_structure, flags=re.MULTILINE)
        ) == 1,
        "has_prerequisites": "## 前提条件" in article,
        "has_all_required_h2": all(
            f"## {heading}" in article for heading in REQUIRED_H2
        ),
        "required_h2_in_order": _required_h2_in_order(article),
        "has_failure_analysis": (
            "## 実際に検出した失敗と根本原因" in article
        ),
        "has_detailed_failure_analysis": _failure_analysis_complete(article),
        "has_observed_results": (
            "## 実測した生成Runの制御比較" in article
            and "## 実測した評価Runの比較" in article
        ),
        "has_apple_silicon_insight": (
            "## Apple Silicon環境で得られた知見" in article
        ),
        "has_references": "## 参考資料" in article,
        "has_summary": "## まとめ" in article,
        "has_required_shell_commands": all(
            command in shell_code for command in REQUIRED_SHELL_COMMANDS
        ),
        "has_train_command": "uv run python train.py" in shell_code,
        "has_accuracy_print": bool(
            re.search(
                r"print\s*\(\s*f?[\"']Accuracy",
                train_code,
                flags=re.IGNORECASE,
            )
        ),
        "has_complete_train_code": train_complete,
        "exception_handling_is_valid": exception_valid,
        "has_ui_comparison_steps": all(
            value in ui_section
            for value in (
                "Iris Classification Experiment",
                "Parameters",
                "Metrics",
                "Artifacts",
                "Compare",
                "Run Duration",
            )
        )
        and len(
            re.findall(r"^\s*\d+[.)]\s+", ui_section, flags=re.MULTILINE)
        )
        >= 5,
        "has_beginner_ml_explanations": all(
            value in _section(article, "MLflowの用語と保存先")
            for value in (
                "Iris",
                "train/test",
                "Logistic Regression",
                "accuracy",
            )
        ),
        "has_complete_troubleshooting": _troubleshooting_complete(article),
        "does_not_claim_unlogged_system_metrics": not (
            _claims_unlogged_system_metrics(ui_section)
        ),
        "has_failure_cases": all(
            f"### {heading}" in article for heading in FAILURE_HEADINGS
        ),
        "has_comparison_run_ids": all(
            value in article
            for value in (
                BASELINE_RUN_ID,
                PROMPT_V2_RUN_ID,
                CONTROLLED_PROMPT_SHA256,
                SECOND_CONTROL_FAILURE_RUN_ID,
                SECOND_CONTROL_SUCCESS_RUN_ID,
                SECOND_CONTROL_PROMPT_SHA256,
            )
        ),
        "single_uv_add_command": len(UV_ADD_PATTERN.findall(shell_code)) == 1,
        "no_instruction_leakage": (
            not ARTICLE_META_PATTERN.search(prose) and not instruction_echoes
        ),
        "balanced_code_fences": article.count("```") % 2 == 0,
        "has_no_thinking_output": "<think>" not in article
        and "</think>" not in article,
        "has_all_reference_urls": all(
            url in article for url in REQUIRED_REFERENCE_URLS
        ),
        "no_duplicate_public_urls": len(public_urls) == len(set(public_urls)),
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
                "3,594",
                "36.993",
                "3.742",
                "3.392",
            )
        ),
        "article_length_in_range": (
            ARTICLE_MIN_CHARS <= len(article) <= ARTICLE_MAX_CHARS
        ),
    }


def failed_article_checks(
    article: str,
    rendered_prompt: str | None = None,
) -> list[str]:
    """失敗した事前検査名を返す。"""
    return [
        name
        for name, passed in article_checks(article, rendered_prompt).items()
        if not passed
    ]
