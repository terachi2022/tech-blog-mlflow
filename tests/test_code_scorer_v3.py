"""Code Scorer-v3の決定論的な回帰テスト。"""

from __future__ import annotations

import sys
import types
import unittest

try:
    from mlflow.entities import Feedback  # noqa: F401
    from mlflow.genai import scorer  # noqa: F401
except ModuleNotFoundError:
    mlflow_stub = sys.modules.get(
        "mlflow",
        types.ModuleType("mlflow"),
    )
    entities_stub = types.ModuleType(
        "mlflow.entities"
    )
    genai_stub = types.ModuleType(
        "mlflow.genai"
    )

    class Feedback:  # type: ignore[no-redef]
        def __init__(
            self,
            value=None,
            rationale=None,
            name=None,
        ) -> None:
            self.value = value
            self.rationale = rationale
            self.name = name

    def scorer(function=None, **_kwargs):
        if callable(function):
            return function

        def decorate(inner):
            return inner

        return decorate

    entities_stub.Feedback = Feedback
    genai_stub.scorer = scorer
    sys.modules["mlflow"] = mlflow_stub
    sys.modules[
        "mlflow.entities"
    ] = entities_stub
    sys.modules[
        "mlflow.genai"
    ] = genai_stub

from evaluation.scorers import (  # noqa: E402
    _numbered_step_count,
    _public_external_urls,
)


class NumberedStepCountTest(
    unittest.TestCase
):
    def test_markdown_numbered_list_counts(
        self,
    ) -> None:
        article = """
1. Serverを起動
2. 記事を生成
3. UIを確認
"""

        self.assertEqual(
            _numbered_step_count(article),
            3,
        )

    def test_fenced_code_is_excluded(
        self,
    ) -> None:
        article = """
```text
1. code内
2. code内
3. code内
```
"""

        self.assertEqual(
            _numbered_step_count(article),
            0,
        )


class PublicExternalURLTest(
    unittest.TestCase
):
    def test_duplicate_urls_count_once(
        self,
    ) -> None:
        url = (
            "https://mlflow.org/docs/latest/"
            "ml/tracking/"
        )
        article = f"{url}\n{url}\n"

        self.assertEqual(
            _public_external_urls(article),
            [url],
        )

    def test_local_url_is_excluded(
        self,
    ) -> None:
        self.assertEqual(
            _public_external_urls(
                "http://127.0.0.1:5000"
            ),
            [],
        )

    def test_single_label_internal_host_is_excluded(
        self,
    ) -> None:
        self.assertEqual(
            _public_external_urls(
                "http://mlflow-server:5000"
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
