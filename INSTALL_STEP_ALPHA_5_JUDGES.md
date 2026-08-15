# STEP-α-5: MLflow Judges Integration

## 結論

OSS MLflow 3.15.1ではBuilt-in ScorerをJudgesへ登録できますが、`@scorer` Decoratorで実装したLocal MLX JudgeはSecurity上登録できません。

Local Judgeを別形式に偽装せず、実行可能な決定論的Guardだけを登録し、Local JudgeはModel、Prompt、Run、Assessment、Datasetで追跡します。

## 登録対象

```text
Name       : article_length_guard_v1
Class      : ResponseLength
Min chars  : 1800
Max chars  : 7000
Auto run   : disabled
```

## Local Judgeの境界

```text
Name   : local_article_judge_v2_4
Kind   : ScorerKind.DECORATOR
OSS    : registration unsupported
Reason : arbitrary code execution during deserialization
```

この判定はMLflow APIへ実際のLocal Judge Scorerを渡し、想定された登録拒否を確認して記録します。モデルはロードせず、推論も実行しません。

## テスト

```bash
uv run python -m py_compile \
  src/tech_blog_mlflow/judge_integration.py \
  evaluation/setup_judges.py \
  evaluation/validate_judges.py \
  tests/test_step_alpha_5_judges.py

uv run python -m unittest \
  tests.test_step_alpha_5_judges \
  tests.test_step_alpha_4_dataset \
  tests.test_step_alpha_3_models \
  tests.test_step_alpha_2_prompts \
  tests.test_step_alpha_1_review \
  -v
```

期待値は46件成功です。

## Dry Run、登録、検証

```bash
uv run python -m evaluation.setup_judges --dry-run
uv run python -m evaluation.setup_judges
uv run python -m evaluation.validate_judges
```

期待値:

```text
Status       : validated
Registered   : ['article_length_guard_v1']
Local MLX GUI: unsupported (tracked as offline scorer evidence)
```

## GUI確認

Experiment `tech-blog-generation`のJudgesを開き、次を確認します。

1. `article_length_guard_v1`が1件だけ存在する
2. Built-in `ResponseLength`である
3. 1,800〜7,000 charsの設定である
4. 自動Samplingが停止している
5. `local_article_judge_v2_4`という偽の登録が存在しない

Local MLX Judgeの結果はEvaluation RunとReview Assessmentで確認します。

## 実施結果（2026-08-15）

Validationは`validated`です。Built-in Scorerは`STOPPED`で登録され、Local decorator Scorerの想定内の拒否、External Model、Prompt、Evaluation Run、Datasetの取得をすべて確認しました。

再実行は`Created: False`となり、重複Versionは作成されていません。

証跡:

- `evaluation_results/judges_alpha_5_dry_run_20260815_152423.json`
- `evaluation_results/judges_alpha_5_applied_20260815_152432.json`
- `evaluation_results/judges_alpha_5_validation_20260815_152439.json`
- `evaluation_results/judges_alpha_5_applied_20260815_152447.json`
