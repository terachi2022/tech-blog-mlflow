# MLflowで技術ブログを生成・評価・分析するチュートリアル

## 1. ゴール

ローカルLLMで記事を生成し、MLflowで次を一貫して管理します。

1. Prompt、生成設定、記事、速度、メモリ
2. Code ScorerとLocal LLM Judgeの評価
3. BaselineとCandidateの比較
4. 人手ReviewとJudge校正
5. Prompts、Models、Datasets、Judges
6. 公開後のGA4・Search Console分析（任意）

採用構成:

| 項目 | 値 |
|---|---|
| Generator | `Qwen/Qwen3-8B-MLX-4bit` |
| Generator Prompt | `article-v3.5.2` |
| Judge | `mlx-community/gemma-3-text-27b-it-4bit` |
| Judge Prompt | `article-judge-v2.4` |
| Evaluator | `combined-v2.4.0` |
| Tracking URI | `http://127.0.0.1:5000` |

```text
Prompt → 記事生成 → 自動評価 → Run比較 → 人手Review
       → Prompt/Model/Dataset/Judges管理 → 公開後分析
```

## 2. 準備

前提はApple Silicon Mac、Python 3.14.6、uv、MLflow 3.15.1、MLX-LM 0.31.3です。Judgeの初回取得は約16GBです。

```bash
cd ~/dev/tech-blog-mlflow
uv sync
uv run python -c 'import mlflow; print(mlflow.__version__)'
```

Terminal AでMLflowを起動します。

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

Terminal Bで確認します。

```bash
curl -i http://127.0.0.1:5000/health
```

GUI: `http://127.0.0.1:5000`

## 3. 回帰テスト

```bash
uv run python -m unittest \
  tests.test_step_alpha_5_judges \
  tests.test_step_alpha_4_dataset \
  tests.test_step_alpha_3_models \
  tests.test_step_alpha_2_prompts \
  tests.test_step_alpha_1_review \
  -v
```

期待値は46件成功です。MLXを使うテストはMetalが必要なため、Headless環境ではGPU非検出で停止する場合があります。

## 4. 記事を生成する

```bash
uv run python -m tech_blog_mlflow.generate_prompt_v3
```

生成処理はPrompt展開、MLX推論、事前検査、SHA-256計算、記事保存、Generation RunへのParameter・Metric・Artifact・Trace保存を行います。

```bash
GENERATION_JSON=$(
  find generation_results -maxdepth 1 \
    -name 'prompt_v3_5_2_generation_*.json' -print \
  | sort | tail -1
)

jq '{run_id, article_path, article_sha256, prompt_version,
     generation_config_version, generation_parameters, metrics,
     all_prechecks_passed, failed_prechecks}' "$GENERATION_JSON"
```

`all_prechecks_passed: true`を確認します。確定済み成果物だけを確認する場合、再生成は不要です。

```text
Article        : articles/prompt_v3_5_2_20260814_164216.md
Generation Run : b5c925c2322b4e30b04f07e24d160a04
```

## 5. 自動評価する

```bash
uv run python -m evaluation.evaluate_prompt_v3 \
  --generation-json "$GENERATION_JSON" \
  --run-name calibrated-prompt-v3.5.2-evaluation-v2.4
```

Code Scorerは構造、Version、前提条件、失敗事例、Link、文字数などを検査します。Local Judgeは技術的正確性、有用性、再現性、引用品質、日本語可読性、独自価値を25サブ項目から評価します。

確定済みEvaluation Run:

```text
fdf0c239445f44a0999a6b1fe7a419b6
```

最新結果と校正監査を確認します。

```bash
EVALUATION_JSON=$(
  find evaluation_results -maxdepth 1 \
    -name 'combined_v2_4_0_prompt-v3.5.2_*.json' -print \
  | sort | tail -1
)

jq '.judge.records[0] | {
  raw_aggregate_scores,
  content_calibration,
  citation_calibration,
  adjusted_aggregate_scores
}' "$EVALUATION_JSON"
```

最終Scoreだけでなく、Raw値と補正理由も確認します。

## 6. Baselineと比較する

```bash
uv run python -m evaluation.compare_runs \
  --baseline-run-id 4f56c781fdfb4e95805c6b957302373f \
  --candidate-run-id fdf0c239445f44a0999a6b1fe7a419b6 \
  --changed-variable-name generator_prompt \
  --baseline-label baseline-v1 \
  --candidate-label article-v3.5.2-max4096 \
  --output-prefix comparison_baseline_vs_prompt_v3_5_2_v2_4
```

比較ではEvaluator Versionを揃え、変更変数を1つに限定します。平均だけでなく各軸、Guardrail、記事本文を確認します。

## 7. MLflow UIでRunを確認する

Generation Run:

- Overview: Model、Prompt、生成設定、時間・メモリ
- Artifacts: 記事、Rendered Prompt、Metadata
- Traces: Themeと生成結果

Evaluation Run:

- Overview: Code MetricsとJudge Metrics
- Traces: 記事と評価Span
- Assessments: 6軸のScoreとRationale
- Artifacts: 記事、Judge Prompt、JSON、CSV

## 8. 人手Reviewする

```bash
uv run python -m evaluation.setup_review_queue --dry-run
uv run python -m evaluation.setup_review_queue
uv run python -m evaluation.validate_review_queue
```

GUIで`article-quality-human-review-v2`を開き、BaselineとCandidateをMarkdown表示で読みます。自動Judgeへ合わせず、6軸、公開可否、Review notesを入力し、2件をCompleteにします。

```bash
uv run python -m evaluation.validate_review_queue --require-complete
```

実施結果は`validated`、Judge MAEは`1.183333`、±1一致率は`0.5`でした。一致度はJudge校正用で、Review機能の合否には使いません。

## 9. Promptを登録する

```bash
uv run python -m evaluation.setup_prompt_registry --dry-run
uv run python -m evaluation.setup_prompt_registry
uv run python -m evaluation.validate_prompt_registry
```

```text
Generator: prompts:/tech-blog-article-generator/1
Judge    : prompts:/tech-blog-article-judge/1
Alias    : production
```

Source SHA-256とTemplateが同じ場合は既存Versionを再利用します。

## 10. Modelを登録する

```bash
uv run python -m evaluation.setup_external_models --dry-run
uv run python -m evaluation.setup_external_models
uv run python -m evaluation.validate_external_models
```

```text
Generator: models:/m-a280e8ca3d5e48f386e5397bae653606
Judge    : models:/m-57625c5d614f4b9382aa9a243abb340c
```

External Model方式のため、MLflowには小さな`MLmodel`だけを保存し、Hugging Face Cacheの重みを複製しません。

## 11. 校正Datasetを登録する

```bash
uv run python -m evaluation.setup_evaluation_dataset --dry-run
uv run python -m evaluation.setup_evaluation_dataset
uv run python -m evaluation.validate_evaluation_dataset
```

```text
Name       : tech-blog-article-quality-calibration-v1
Dataset ID : d-f21d22043d7749a387cf34bc06fcffd5
Records    : baseline, prompt-v3.5.2
```

記事、SHA-256、Run、Trace、Prompt、人手6軸、公開可否、Review notesを固定します。内容変更時は上書きせず`-v2`を作ります。

## 12. Judgesを構成する

```bash
uv run python -m evaluation.setup_judges --dry-run
uv run python -m evaluation.setup_judges
uv run python -m evaluation.validate_judges
```

Judgesには`article_length_guard_v1`が登録されます。`ResponseLength`による1,800〜7,000文字の決定論的検査で、`Evaluating traces: OFF`が正常です。

Local MLX Judgeは`@scorer`形式です。OSS MLflow 3.15.1はSecurity上この形式のサーバー登録を禁止するため、Model、Prompt、Evaluation Assessment、Review、Datasetへ分離して追跡します。JudgesのModel欄が空でも問題ありません。

## 13. 公開後分析（任意）

Online値でOffline Scoreを上書きしません。

### 公開記事を登録

```bash
uv run python -m evaluation.register_publication \
  --published-url '<CANONICAL_URL>' \
  --published-at '<ISO8601_WITH_TIMEZONE>' \
  --ga4-property-id '<GA4_PROPERTY_ID>' \
  --gsc-site-url '<GSC_PROPERTY_URL>' \
  --cta-not-implemented \
  --dry-run
```

確認後に`--dry-run`を外します。

### GA4とSearch Consoleを収集

```bash
uv run python -m evaluation.collect_ga4 \
  --publication-id '<PUBLICATION_ID>' \
  --start-date '<YYYY-MM-DD>' --end-date '<YYYY-MM-DD>' --dry-run

uv run python -m evaluation.collect_gsc \
  --publication-id '<PUBLICATION_ID>' \
  --start-date '<YYYY-MM-DD>' --end-date '<YYYY-MM-DD>' --dry-run
```

### OfflineとOnlineを結合

```bash
uv run python -m evaluation.evaluate_online \
  --publication-id '<PUBLICATION_ID>' \
  --ga4-observation-id '<GA4_OBSERVATION_ID>' \
  --gsc-observation-id '<GSC_OBSERVATION_ID>' \
  --dry-run
```

GA4/GSCが非Partial、GSCが`dataState=final`、日付Label範囲が一致する場合だけFinal扱いにします。確認後に`--dry-run`を外します。

## 14. 日常運用の最短手順

```bash
# Terminal A
uv run mlflow server --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 --port 5000

# Terminal B
uv run python -m tech_blog_mlflow.generate_prompt_v3

GENERATION_JSON=$(
  find generation_results -maxdepth 1 \
    -name 'prompt_v3_5_2_generation_*.json' -print \
  | sort | tail -1
)

uv run python -m evaluation.evaluate_prompt_v3 \
  --generation-json "$GENERATION_JSON"
```

その後、GUIでGeneration/Evaluation Run、Trace、Assessment、Artifactを確認します。正式採用するCandidateだけ、同一Evaluator比較、人手Review、Prompt/Model/Datasetの新Version登録へ進めます。

## 15. トラブルシューティング

| 症状 | 確認・対処 |
|---|---|
| MLflowへ接続できない | `curl -i http://127.0.0.1:5000/health` |
| Judge初回が遅い | 約16GBの取得とModel Load完了を待つ |
| Metal deviceがない | Apple SiliconのGUI Sessionで実行する |
| ReviewがJSON文字列表示 | v2 Queueを選ぶ |
| JudgesのModel欄が空 | ResponseLengthはLLM不要なので正常 |
| 同じVersionが増える | 同一仕様ならSetup CLIが既存項目を再利用する |

## 16. 完了チェック

- Generation Runに記事、Prompt、設定、Metric、Artifact、Traceがある
- Evaluation RunにCode/Judge Metrics、Assessment、校正監査がある
- 同一EvaluatorでBaseline/Candidateを比較できる
- ReviewでMarkdown表示と人手評価保存ができる
- PromptsにGenerator/Judge Versionがある
- Modelsに2つのExternal Modelがある
- Datasetに記事と人手期待値がある
- Judgesに`article_length_guard_v1`が1件あり、自動評価がOFF
- Local Judge結果がEvaluation RunとAssessmentに残る
- Online分析がOffline評価と別Namespaceで保存される

## 17. Multi-model Candidate（任意）

既存構成をBaselineとして残したまま、Swallow 120Bによる生成、Qwen3.6による校正と主評価、Gemmaによる独立評価を追加できます。

```bash
uv run python -m tech_blog_mlflow.generate_candidate --dry-run
uv run python -m unittest tests.test_candidate_multi_model -v
```

実モデルを使う手順、校正Run、二重評価、採用条件は`INSTALL_CANDIDATE_MULTI_MODEL.md`を参照してください。公式MXFP4を直接使わず、Apple Silicon向け第三者MLX変換をCandidateとして検証するため、既存の`production` aliasは自動変更しません。

詳細は次の資料を参照してください。

- `README.md`
- `INSTALL_STEP_ALPHA_1_REVIEW.md`
- `INSTALL_STEP_ALPHA_2_PROMPTS.md`
- `INSTALL_STEP_ALPHA_3_MODELS.md`
- `INSTALL_STEP_ALPHA_4_DATASET.md`
- `INSTALL_STEP_ALPHA_5_JUDGES.md`
- `INSTALL_CANDIDATE_MULTI_MODEL.md`
