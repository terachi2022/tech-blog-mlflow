# STEP-α-1 v2: Markdown表示対応のMLflow Review

## 結論

記事ファイルは修正しません。既存の評価Traceを直接Reviewへ入れると、MLflow UIが記事本文をJSON文字列として扱い、先頭と末尾の引用符および`\n`を表示します。

この版では次の順に処理します。

1. 既存のBaseline/Candidate評価Traceから記事Markdownを取得する
2. SHA-256、H1/H2、全文表示上限を検証する
3. `response_preview`へRaw Markdown全文を明示したReview表示専用Traceを作る
4. Judge 6軸を元Trace ID付きで表示専用Traceへ複製する
5. 新Queue `article-quality-human-review-v2`へ表示専用Traceだけを登録する
6. CLIでRaw Markdown表示とQueue構成を検証してから人手採点する

旧Queue `article-quality-human-review-v1`は監査履歴として残します。採点には使用しません。

## 1. 完全差し替え

Project RootでArchiveを展開します。

```bash
cd ~/dev/tech-blog-mlflow

tar -xvf step_alpha_1_review_v2_complete.tar.gz
```

差し替わるファイル:

```text
src/tech_blog_mlflow/review_workflow.py
evaluation/setup_review_queue.py
evaluation/validate_review_queue.py
tests/test_step_alpha_1_review.py
INSTALL_STEP_ALPHA_1_REVIEW.md
README.md
```

## 2. 構文検査とテスト

```bash
uv run python -m py_compile \
  src/tech_blog_mlflow/review_workflow.py \
  evaluation/setup_review_queue.py \
  evaluation/validate_review_queue.py \
  tests/test_step_alpha_1_review.py

uv run python -m unittest \
  tests.test_step_alpha_1_review \
  -v
```

期待値:

```text
Ran 19 tests

OK
```

全回帰テストも実行します。

```bash
uv run python -m unittest discover \
  -s tests \
  -p 'test_*.py' \
  -v
```

## 3. MLflow Server

Terminal Aで起動し、そのまま開いておきます。

```bash
cd ~/dev/tech-blog-mlflow

uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

Terminal Bで確認します。

```bash
curl -f http://127.0.0.1:5000/health
```

## 4. Dry Run

```bash
uv run python -m evaluation.setup_review_queue \
  --dry-run
```

この段階ではMLflowへ書き込みません。次を検証します。

- Baseline/Candidate Evaluation RunがExperiment `tech-blog-generation`に存在する
- 各RunにTraceが1件だけある
- 記事がH1/H2を持つMarkdownである
- Traceの`article_sha256`と記事本文が一致する
- 記事がRaw Markdown全文表示の安全上限9,000文字以内である
- 新Queue v2と8質問の作成計画が生成できる

## 5. 新Queue v2を作る

```bash
uv run python -m evaluation.setup_review_queue
```

成功条件:

```text
Mode           : apply
Queue          : article-quality-human-review-v2
Target Traces  : 2
Questions      : 8
Review Traces  : 2
Markdown valid : True
```

このCommandは冪等です。再実行時は同じ表示専用TraceとQueueを再利用します。

## 6. 採点前の機械検証

```bash
uv run python -m evaluation.validate_review_queue
```

採点前の期待値:

```text
Status          : ready_for_human_review
Items           : 2
Pending         : 2
Markdown valid  : True
Setup effective : True
Review effective: False
```

`Review effective: False`は未採点を意味します。`Markdown valid: True`と`Setup effective: True`になるまでは採点しません。

## 7. MLflow UIで表示を目視確認する

1. `http://127.0.0.1:5000`を開く
2. GenAI → Experiment `tech-blog-generation` → Reviewを開く
3. Queueで必ず`article-quality-human-review-v2`を選ぶ
4. 2件とも次を確認する

目視合格条件:

- 記事タイトルが見出しとして表示される
- H2見出し、表、リスト、Code BlockがMarkdownとして読める
- 本文全体が先頭と末尾の`"`で囲まれていない
- 改行が文字列`\n`として並んでいない
- BaselineとCandidateを切り替えて読める

1つでも満たさない場合は採点せず、次を保存して停止します。

```bash
LATEST_REVIEW_SETUP=$(
  find evaluation_results \
    -maxdepth 1 \
    -name 'review_setup_alpha_1_applied_*.json' \
    -print \
  | sort \
  | tail -1
)

jq '{
  queue_name,
  presentations,
  review_url
}' "$LATEST_REVIEW_SETUP"
```

## 8. 人手採点

表示確認後、2件それぞれについて実施します。

1. 既存Judge値を見ずに6軸を1〜5で採点する
2. `Publishable`または`Needs revision`を選ぶ
3. 必要なら総評を入力する
4. ItemをCompleteにする

## 9. 完了検証

```bash
uv run python -m evaluation.validate_review_queue \
  --require-complete
```

期待値:

```text
Status          : validated
Items           : 2
Pending         : 0
Complete        : 2
Markdown valid  : True
Setup effective : True
Review effective: True
```

最新Report:

```bash
REVIEW_VALIDATION_JSON=$(
  find evaluation_results \
    -maxdepth 1 \
    -name 'review_validation_alpha_1_*.json' \
    -print \
  | sort \
  | tail -1
)

jq '{
  validation_status,
  queue,
  effectiveness,
  trace_reviews,
  judge_alignment
}' "$REVIEW_VALIDATION_JSON"
```

## 10. 有効性の判定

| 判定 | 必須条件 |
|---|---|
| `setup_effective` | v2 Queueに2表示専用Trace、8 Schema、Raw Markdown Previewがある |
| `human_assessment_capture_effective` | 2件とも人手6軸と公開可否がTraceへ保存された |
| `workflow_completion_effective` | 上記に加えて2 ItemがCompleteになった |

3項目がすべて`true`ならSTEP-α-1は完了です。JudgeとのMAEと一致率は校正の参考値であり、Review機能自体のPASS/FAILには使いません。

### 実施済み結果（2026-08-15）

2件の人手採点と完了検証を実施し、次の結果を確認しました。

```text
Status          : validated
Items           : 2
Pending         : 0
Complete        : 2
Markdown valid  : True
Setup effective : True
Review effective: True
Judge MAE       : 1.183333
Within ±1       : 0.5
```

証跡は`evaluation_results/review_validation_alpha_1_20260815_140944.json`です。両記事の公開可否は`Needs revision`でした。標本が2件だけなのでJudgeの即時変更は行わず、複数記事のDatasetを用意した後の校正候補として扱います。

## 11. 停止時の確認

### v1 Queueを開いている

今回の正解は`article-quality-human-review-v2`です。v1は読みにくい旧表示を再現する監査履歴です。

### `Markdown valid: False`

Review表示専用TraceのPreviewが全文Raw Markdownではありません。採点せず、Setup JSONとValidation JSONを確認します。

### 記事が9,000文字を超える

暗黙の切断はしません。全文を一度にReviewする設計をやめ、Section分割などを別Versionとして設計します。

### `--require-complete`がExit 1

v2 Queueの2件について、6軸、公開可否、Complete操作が済んでいるか確認します。総評だけは任意です。
