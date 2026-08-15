# STEP-α-2: MLflow Prompt Registry

## 結論

採用中のGenerator/Judge PromptをMLflowへVersion登録し、`production` Aliasと既存Runへの対応を保存します。Source Promptファイルは変更しません。

## 登録対象

| Role | Source | MLflow Prompt | Run |
|---|---|---|---|
| Generator | `prompts/article_generation_v3_5_2.md` | `tech-blog-article-generator` | `b5c925c2322b4e30b04f07e24d160a04` |
| Judge | `prompts/article_judge_v2_4.md` | `tech-blog-article-judge` | `fdf0c239445f44a0999a6b1fe7a419b6` |

## テスト

```bash
uv run python -m py_compile \
  src/tech_blog_mlflow/prompt_registry.py \
  evaluation/setup_prompt_registry.py \
  evaluation/validate_prompt_registry.py \
  tests/test_step_alpha_2_prompts.py

uv run python -m unittest \
  tests.test_step_alpha_2_prompts \
  -v
```

## Dry Run

```bash
uv run python -m evaluation.setup_prompt_registry --dry-run
```

Promptファイル、変数Contract、Source SHA-256、Model Config、Judge JSON Schemaの登録計画だけを検査します。

## 登録

```bash
uv run python -m evaluation.setup_prompt_registry
```

同一Source SHA-256とTemplateのVersionは再利用するため、再実行してもVersionを増やしません。`production` Aliasを登録Versionへ設定し、GeneratorはGeneration Run、JudgeはEvaluation Runへ接続します。

## 検証

```bash
uv run python -m evaluation.validate_prompt_registry
```

成功条件:

```text
Status     : validated
Generator : prompts:/tech-blog-article-generator/1 valid=True
Judge     : prompts:/tech-blog-article-judge/1 valid=True
```

検証項目:

1. Registry TemplateとSource FileがByte単位で一致する
2. Generatorの5変数とJudgeの`ARTICLE`変数が一致する
3. Model名、temperature、max_tokensが一致する
4. Judgeの25サブ項目JSON Schemaが一致する
5. `production` Aliasが登録Versionを指す
6. 採用Runの`mlflow.linkedPrompts`が登録Versionを指す

## MLflow 3.15.1 OSSのRun Link検証

`link_prompt_version_to_run()`はRun側の`mlflow.linkedPrompts`へ接続を保存します。一方、同Versionの`list_logged_prompts()`はModel Version側の別Tagを検索するため、OSS 3.15.1では空Listになる場合があります。本実装は、実際に書き込まれるRun側Tagを検証元とします。

## 実施結果（2026-08-15）

2件ともVersion 1として登録され、`production` Alias、Model Config、Judge JSON Schema、Run Linkを含めて`validated`になりました。再実行では`created: false`かつ`run_link_created: false`を確認しています。

証跡:

- `evaluation_results/prompt_registry_alpha_2_applied_20260815_142649.json`
- `evaluation_results/prompt_registry_alpha_2_validation_20260815_142813.json`
- `evaluation_results/prompt_registry_alpha_2_applied_20260815_142822.json`
