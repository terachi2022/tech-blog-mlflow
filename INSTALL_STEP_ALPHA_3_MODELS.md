# STEP-α-3: MLflow External Models

## 結論

GeneratorとJudgeをExternal ModelとしてModels画面へ登録します。Hugging Faceのモデル重みはMLflowへコピーせず、識別情報と実行設定だけを追跡します。

## テスト

```bash
uv run python -m py_compile \
  src/tech_blog_mlflow/external_model_registry.py \
  evaluation/setup_external_models.py \
  evaluation/validate_external_models.py \
  tests/test_step_alpha_3_models.py

uv run python -m unittest \
  tests.test_step_alpha_3_models \
  tests.test_step_alpha_2_prompts \
  tests.test_step_alpha_1_review \
  -v
```

期待値は34件成功です。

## Dry Run

MLflow Serverを起動した状態で実行します。

```bash
uv run python -m evaluation.setup_external_models --dry-run
```

## 登録

```bash
uv run python -m evaluation.setup_external_models
```

登録対象:

```text
tech-blog-generator-qwen3-8b-mlx-4bit
tech-blog-judge-gemma3-27b-mlx-4bit
```

同じ名前と仕様SHA-256のModelは再利用します。同名で仕様が異なる場合は既存Modelを上書きせず停止します。

## 検証

```bash
uv run python -m evaluation.validate_external_models
```

期待値:

```text
Status     : validated
Generator : models:/<model-id> valid=True
Judge     : models:/<model-id> valid=True
```

検証項目:

1. Statusが`READY`
2. Model TypeとSource Runが仕様通り
3. Hugging Face ID、用途、仕様SHA-256が一致
4. Runtime、量子化、max_tokensなどのParameterが一致
5. STEP-α-2のPrompt VersionがModelへ接続されている
6. Artifactがメタデータ用`MLmodel`だけで、重みが保存されていない

## GUI確認

MLflowのExperiment `tech-blog-generation`でModelsを開き、2件が表示されることを確認します。各ModelでSource Run、Parameter、Tag、Linked Promptを確認してください。

画面確認とValidationが成功するまではSTEP-α-3を完了扱いにしません。

## 実施結果（2026-08-15）

```text
Status     : validated
Generator : models:/m-a280e8ca3d5e48f386e5397bae653606 valid=True
Judge     : models:/m-57625c5d614f4b9382aa9a243abb340c valid=True
```

両Modelは`READY`で、Artifactは`MLmodel`だけです。再実行では同じModel IDを再利用し、ModelとPrompt Linkの重複作成がないことを確認しました。

証跡:

- `evaluation_results/external_models_alpha_3_dry_run_20260815_150709.json`
- `evaluation_results/external_models_alpha_3_applied_20260815_150717.json`
- `evaluation_results/external_models_alpha_3_validation_20260815_150723.json`
- `evaluation_results/external_models_alpha_3_applied_20260815_150732.json`
