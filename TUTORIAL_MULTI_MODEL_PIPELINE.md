# MLflowで構築するマルチモデル技術記事パイプライン

## 1. このチュートリアルで行うこと

このチュートリアルでは、Apple Silicon上のローカルLLMを役割ごとに分け、技術記事を次の順序で生成・校正・評価します。

```text
テーマ
  → Swallow 120Bで記事生成
  → Qwen3.6 27Bでレビュー・校正
  → Qwen3.6 27Bで一次評価
  → Gemma 3 27Bで独立評価
  → MLflowでRun、Artifact、Scoreを確認
  → ModelsとHuman Review Queueへ公開
  → 人が採否を判断
```

生成モデル自身に採点させず、ReviewerとJudgeも分離します。さらに、同じ校正済み記事を異なる2モデルで評価することで、単一Judgeの傾向をそのまま採用判断にしない構成です。

実装済みPipeline Versionは`candidate-multi-model-v1.0.0`です。

## 2. モデル構成

| 役割 | モデル | 最大生成Token | 用途 |
|---|---|---:|---|
| Generator | `tocchitocchi/GPT-OSS-Swallow-120B-RL-v0.1-6bit-mlx` | 10,000 | 記事本文の生成 |
| Reviewer | `mlx-community/Qwen3.6-27B-5bit` | 8,192 | 構造・内容のレビューと校正 |
| Primary Judge | `mlx-community/Qwen3.6-27B-5bit` | 3,600 | 一次評価 |
| Independent Judge | `mlx-community/gemma-3-text-27b-it-4bit` | 3,600 | 独立評価 |

Swallowの公式MXFP4 checkpointはvLLM向けです。このPipelineではApple Siliconで動かすため、第三者がMLX向けに変換した6bitモデルをCandidateとして使用します。既存のProduction Modelを置き換える前に、複数テーマで品質とリソース消費を検証してください。

## 3. 前提条件

検証済み環境は次のとおりです。

| 項目 | Version・設定 |
|---|---|
| Hardware | Apple M5 Max |
| OS | macOS 26.5.1 |
| Python | 3.14.6 |
| MLflow | 3.15.1 |
| MLX-LM | 0.31.3 |
| Package manager | `uv` |
| MLflow Tracking URI | `http://127.0.0.1:5000` |
| Backend Store | SQLite `mlflow.db` |
| Experiment | `tech-blog-generation` |

Generatorは実行例で約96 GBのPeak Memoryを使用しました。必要なUnified Memoryに余裕があるApple Silicon Macで実行してください。また、モデルを初めて使うときはHugging Faceから重みを取得するため、十分なDisk容量とNetwork接続が必要です。

Project Rootへ移動し、依存関係を同期します。

```bash
cd ~/dev/tech-blog-mlflow
uv sync

uv run python --version
uv run python -c 'import mlflow; print(mlflow.__version__)'
uv run python -c 'import mlx_lm; print(mlx_lm.__version__)'
```

## 4. MLflow Tracking Serverを起動する

Terminal AでTracking Serverを起動します。このTerminalはPipeline完了まで開いたままにします。

```bash
cd ~/dev/tech-blog-mlflow

uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

Terminal BでHealth Checkを実行します。

```bash
curl -i http://127.0.0.1:5000/health
```

HTTP Status `200`を確認したら、Browserで`http://127.0.0.1:5000`を開きます。

## 5. 軽量検証を行う

モデルを読み込む前に、設定とCommandの組み立てを確認します。

```bash
uv run python -m tech_blog_mlflow.generate_candidate --dry-run
uv run python -m unittest tests.test_candidate_multi_model -v
```

Dry Runでは、次の値を確認します。

- `pipeline_version`が`candidate-multi-model-v1.0.0`
- GeneratorがSwallow 120BのMLX変換版
- `max_tokens`が`10000`
- `temperature`が`0.0`

## 6. Swallow 120Bで記事を生成する

記事テーマを指定して生成します。

```bash
uv run python -m tech_blog_mlflow.generate_candidate \
  --theme 'MLflowを使って機械学習の実験を管理する方法'
```

処理が完了すると、Generation Run ID、記事Path、Metadata JSON、事前検査結果が表示されます。出力を再利用しやすいよう、最新Metadataから値を取得します。

```bash
GENERATION_JSON=$(
  find generation_results -maxdepth 1 \
    -name 'candidate_swallow_generation_*.json' -print \
  | sort | tail -1
)

GENERATION_RUN_ID=$(jq -r '.run_id' "$GENERATION_JSON")
GENERATED_ARTICLE=$(jq -r '.article_path' "$GENERATION_JSON")

jq '{
  run_id,
  article_path,
  article_sha256,
  generation_parameters,
  metrics,
  all_prechecks_passed,
  failed_prechecks
}' "$GENERATION_JSON"
```

Generatorでは、公開用本文だけでなく内部Reasoningも10,000 Tokenの生成枠を共有します。次を確認してください。

- `generated_tokens_including_reasoning`が上限付近に達していない
- 記事が途中で切れていない
- `article_length`が1,800〜7,000文字
- 架空の実行結果や存在しない引用がない
- `failed_prechecks`の内容がReviewerで安全に修正できる構造上の問題か

事前検査に失敗したからといって、無条件に次へ進めてはいけません。本文の切断、根拠のない主張、重要情報の欠落がある場合は再生成します。見出しの不足や順序など、校正対象として設計された問題だけをReviewerへ渡します。

## 7. Qwen3.6でレビュー・校正する

最初にDry Runで入力PathとRun IDを確認します。

```bash
uv run python -m evaluation.review_candidate \
  --article "$GENERATED_ARTICLE" \
  --source-run-id "$GENERATION_RUN_ID" \
  --dry-run
```

問題がなければ校正を実行します。

```bash
uv run python -m evaluation.review_candidate \
  --article "$GENERATED_ARTICLE" \
  --source-run-id "$GENERATION_RUN_ID"
```

Reviewerは元記事を上書きしません。校正済み記事を`articles/candidate_reviewed_*.md`へ、指摘と監査情報を`review_results/candidate_review_*.json`へ保存し、Generation Runとは別のReview Runを作成します。

```bash
REVIEW_JSON=$(
  find review_results -maxdepth 1 \
    -name 'candidate_review_*.json' -print \
  | sort | tail -1
)

REVIEW_RUN_ID=$(jq -r '.review_run_id' "$REVIEW_JSON")
REVIEWED_ARTICLE=$(jq -r '.revised_article_path' "$REVIEW_JSON")

jq '{
  review_run_id,
  source_run_id,
  revised_article_path,
  revised_article_sha256,
  review: {
    summary: .review.summary,
    issues: .review.issues
  },
  all_prechecks_passed,
  failed_prechecks
}' "$REVIEW_JSON"
```

評価へ進む条件は`all_prechecks_passed: true`です。Reviewerでは通常の構造検査に加えて、次も検査します。

- 検証済みVersionやRun IDなどの不変Evidenceが保持されている
- 校正後の長さが元記事の90%以上で、実質的内容が失われていない
- 記事長が1,800〜7,000文字に収まっている

`failed_prechecks`が1件でも残った場合は評価を止め、元記事、Review JSON、校正済み記事の差分を確認します。

```bash
diff -u "$GENERATED_ARTICLE" "$REVIEWED_ARTICLE"
```

## 8. 2つのJudgeで順番に評価する

同じ校正済み記事をPrimary JudgeとIndependent Judgeで評価します。2モデルを同時にMemoryへ載せないよう、評価CommandはQwen3.6、Gemmaの順に逐次実行されます。

```bash
uv run python -m evaluation.evaluate_candidate_dual \
  --article "$REVIEWED_ARTICLE" \
  --source-run-id "$REVIEW_RUN_ID" \
  --dry-run
```

Dry Runには`evaluation.evaluate_combined_v2_4`を呼び出す2つのCommandが表示されます。記事Path、Review Run ID、Judge Model、`judge-role`を確認してから実行します。

```bash
uv run python -m evaluation.evaluate_candidate_dual \
  --article "$REVIEWED_ARTICLE" \
  --source-run-id "$REVIEW_RUN_ID"
```

各Judgeは同じCode Scorerと`article-judge-v2.4`で次の6軸を採点します。

| 評価軸 | 主な確認内容 |
|---|---|
| `technical_accuracy` | 概念、API、Command、記事内整合性 |
| `helpfulness` | 目的、実行可能性、対象読者、問題解決力 |
| `reproducibility` | 環境、依存関係、Code、実行順、確認方法 |
| `citation_quality` | 情報源の権威性、主張との対応、Coverage |
| `readability_ja` | 構成、日本語、用語説明、情報密度 |
| `original_value` | 実測Evidence、失敗分析、比較、環境固有知見 |

## 9. 評価結果を確認する

最新2件の評価結果を一覧にします。

```bash
find evaluation_results -maxdepth 1 \
  -name 'combined_v2_4_0_candidate-reviewed_*.json' -print \
| sort | tail -2 \
| while read -r result; do
    jq '{
      run_id,
      model: .judge.model,
      article_sha256: .article.sha256,
      source_run_id: .article.source_run_id,
      metrics
    }' "$result"
  done
```

2つの結果で、次が一致していることを確認します。

- `article_sha256`
- `source_run_id`
- `combined_version`
- `judge.prompt_version`
- `citation_calibration_version`
- `content_calibration_version`

記事やEvaluator Versionが異なる結果は、Judge間比較に使用できません。MLflow UIではRun Tagの`judge_role=primary`と`judge_role=independent`も確認します。

### 9.1 Models、Human Review、GUI比較を有効にする

GenerationからEvaluationまではRunを作成しますが、それだけではMLflowの
Models画面やHuman Review Queueには登録されません。既存Runを再実行せず、
最新の4 RunをGUI用Entityへ公開します。

```bash
uv run python -m evaluation.publish_candidate_gui --dry-run
uv run python -m evaluation.publish_candidate_gui
```

誤って別テーマのRunを混ぜないため、Dry Runに表示された4つのRun IDを確認します。
必要なら明示的に固定できます。

```bash
uv run python -m evaluation.publish_candidate_gui \
  --generation-run-id "$GENERATION_RUN_ID" \
  --review-run-id "$REVIEW_RUN_ID" \
  --primary-run-id '<Primary Judge Run ID>' \
  --independent-run-id '<Independent Judge Run ID>'
```

公開後はMLflow GUIで次を確認できます。

- **Models**: Generator、Reviewer、Primary Judge、Independent Judgeの4件
- **Review**: `candidate-multi-model-human-review-v1` Queue内の校正済み記事
- **Experiments**: PrimaryとIndependentの2 Runを選択して **Compare** を押すと、6軸ScoreをGUI上で比較可能

第7章のReview RunはLLMによる校正履歴です。Human Reviewの承認は別Entityであり、
この公開ステップで初めてReview Queueに現れます。

## 10. 実行例

2026年8月15日の実行では、Swallow生成直後に3つの構造検査が失敗しましたが、Qwen3.6による校正後は全件通過しました。

| Stage | Run ID | 結果 |
|---|---|---|
| Generation | `2c6b8629ad8245b2815cdb99b24a3e62` | 6,396文字、生成67.866秒、Peak Memory 95.904 GB |
| Review | `9237f6f9f54445bfb3eb20d4a63f2f8a` | 全事前検査PASS |
| Primary Judge | `2902d2e7831b4ca6abb519a249cb8389` | Qwen3.6による一次評価 |
| Independent Judge | `641f9270c0f3493db7cfc9c217b65df6` | Gemma 3による独立評価 |

校正済み記事は6,403文字で、Code Scorerは全必須条件を満たしました。Judge Scoreは次のとおりです。

| 評価軸 | Qwen3.6 Primary | Gemma Independent | 差の絶対値 |
|---|---:|---:|---:|
| Technical accuracy | 4.50 | 5.00 | 0.50 |
| Helpfulness | 4.75 | 4.50 | 0.25 |
| Reproducibility | 4.80 | 4.80 | 0.00 |
| Citation quality | 4.25 | 5.00 | 0.75 |
| Readability ja | 4.25 | 4.75 | 0.50 |
| Original value | 3.00 | 4.00 | 1.00 |

Judgeの平均だけを見ると差の理由を失います。この例では`original_value`の差が最大です。各Result JSONの`validated_result`と`rationale`を読み、どのEvidenceを異なる基準で評価したのかを確認します。

```bash
jq '.judge.records[0] | {
  raw_aggregate_scores,
  adjusted_aggregate_scores,
  content_adjustments: .content_calibration.adjustments,
  citation_adjustments: .citation_calibration.adjustments
}' evaluation_results/combined_v2_4_0_candidate-reviewed_20260815_171045_641f9270c0f3493db7cfc9c217b65df6.json
```

## 11. 採用判断

高Scoreでも自動昇格は行いません。最低3〜5テーマを同一条件で実行し、次を人が確認します。

- Baselineより技術的正確性、有用性、再現性が改善したか
- Judge間のScore差をRationaleで説明できるか
- 架空の実測値、引用、URLが追加されていないか
- Reviewerが正しい記述を削除・改変していないか
- 約96 GBのPeak Memoryと実行時間を継続運用できるか
- 改善幅が大規模モデルの運用Costに見合うか

比較時は記事テーマ、Prompt、Evaluatorを揃えます。このCandidateはGenerator、Reviewer、Judge構成を同時に変更しているため、単一変数の因果比較としては扱わず、Pipeline全体のCandidateとして評価します。条件を満たすまでは、既存Registryの`production` aliasを変更しません。

## 12. トラブルシューティング

| 症状 | 原因の候補 | 確認・対処 |
|---|---|---|
| MLflowへ接続できない | Server未起動、Port違い | `curl -i http://127.0.0.1:5000/health`を実行する |
| 初回実行が長い | Model Weightを取得中 | Network、空きDisk容量、Hugging Face Cacheを確認する |
| Memory不足で停止する | Swallow 120Bの読込み余力不足 | 他Applicationを終了し、十分なUnified Memoryを確保する |
| 記事が途中で終わる | Reasoning込みのToken上限到達 | `generated_tokens_including_reasoning`を確認し、内容を再生成する |
| Generationの検査が失敗する | 見出し不足、順序違い、内容欠落 | `failed_prechecks`を確認し、構造問題だけReviewerへ渡す |
| Review後も検査が失敗する | Evidence消失、記事短縮、構造未修正 | 評価を止め、`diff -u`とReview JSONを確認する |
| 2つの評価が比較できない | 記事またはEvaluator Versionが不一致 | SHA-256と各Versionを揃えて再評価する |
| Judge Scoreが大きく異なる | 評価観点やEvidence解釈の差 | Aggregate平均ではなくSubscoreとRationaleを人が読む |

## 13. 完了チェック

- Generation、Review、Primary Evaluation、Independent Evaluationの4 Runがある
- Modelsに4役のExternal Modelがある
- Human Review Queueに校正済み記事があり、人の承認結果が保存されている
- PrimaryとIndependentの2 RunをGUIのCompareで比較した
- 各Stageの`source_run_id`を辿れる
- Generation MetadataにToken数、時間、Peak Memory、SHA-256がある
- Review後の`all_prechecks_passed`が`true`
- 2つのEvaluationが同一記事SHA-256を参照している
- Code Scorerの必須項目がすべて`1.0`
- Judgeごとの6軸Score、Subscore、Rationaleを確認した
- Judge間差と運用Costを含め、人が採否を判断した

この状態になれば、新しいマルチモデルPipelineの1回分の検証は完了です。次はテーマを変えて同じ条件で繰り返し、単発の高Scoreではなく安定性を確認します。
