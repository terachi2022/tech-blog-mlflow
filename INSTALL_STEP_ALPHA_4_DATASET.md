# STEP-α-4: MLflow Evaluation Dataset

## 結論

Baselineと採用Candidateの記事、およびSTEP-α-1の人手評価を校正用Evaluation Datasetとして固定します。

## Dataset Contract

```text
Name    : tech-blog-article-quality-calibration-v1
Records : baseline, prompt-v3.5.2
Mode    : immutable
```

Record構造:

- `inputs`: 記事Markdown、Theme
- `expectations`: 人手6軸、公開可否、Review notes
- `tags`: Variant、記事Path/SHA-256、Generation/Evaluation Run、Trace、Prompt Version

## テスト

```bash
uv run python -m py_compile \
  src/tech_blog_mlflow/evaluation_dataset_registry.py \
  evaluation/setup_evaluation_dataset.py \
  evaluation/validate_evaluation_dataset.py \
  tests/test_step_alpha_4_dataset.py

uv run python -m unittest \
  tests.test_step_alpha_4_dataset \
  tests.test_step_alpha_3_models \
  tests.test_step_alpha_2_prompts \
  tests.test_step_alpha_1_review \
  -v
```

期待値は40件成功です。

## Dry Runと登録

```bash
uv run python -m evaluation.setup_evaluation_dataset --dry-run
uv run python -m evaluation.setup_evaluation_dataset
```

## 検証

```bash
uv run python -m evaluation.validate_evaluation_dataset
```

期待値:

```text
Status     : validated
Dataset    : tech-blog-article-quality-calibration-v1
Records    : 2
Variants   : ['baseline', 'prompt-v3.5.2']
```

検証項目:

1. DatasetがExperiment `tech-blog-generation`へ接続されている
2. 固定TagとManifest SHA-256が一致する
3. Recordが2件でVariantが重複しない
4. 記事Markdownと記事SHA-256がSource Fileに一致する
5. 人手6軸と公開可否が両Recordに存在する
6. 再実行で既存Datasetを再利用する

## 更新規則

このDatasetは不変です。記事、由来、人手期待値のいずれかを変更する場合、既存Recordを更新せず、新しいDataset名を`-v2`として作成します。

## 実施結果（2026-08-15）

```text
Dataset ID : d-f21d22043d7749a387cf34bc06fcffd5
Digest     : 0064e950
Status     : validated
```

再実行は`Created: False`となり、同じDataset IDが再利用されました。

証跡:

- `evaluation_results/evaluation_dataset_alpha_4_dry_run_20260815_151544.json`
- `evaluation_results/evaluation_dataset_alpha_4_applied_20260815_151552.json`
- `evaluation_results/evaluation_dataset_alpha_4_validation_20260815_151637.json`
- `evaluation_results/evaluation_dataset_alpha_4_applied_20260815_151647.json`
