# Multi-model Candidateの実行手順

既存のQwen3-8B GeneratorとGemma JudgeはBaselineとして残します。Candidateは次の4役です。

| 役割 | Model |
|---|---|
| Generator | `tocchitocchi/GPT-OSS-Swallow-120B-RL-v0.1-6bit-mlx` |
| Reviewer | `mlx-community/Qwen3.6-27B-5bit`（非MTP） |
| Primary Judge | `mlx-community/Qwen3.6-27B-5bit`（非MTP） |
| Independent Judge | `mlx-community/gemma-3-text-27b-it-4bit` |

公式Swallow MXFP4はvLLM向けです。Apple Silicon上では第三者MLX変換を使うため、Candidateとして検証してから採用します。

## 1. 軽量確認

```bash
uv run python -m tech_blog_mlflow.generate_candidate --dry-run
uv run python -m unittest tests.test_candidate_multi_model -v
```

## 2. 記事生成

```bash
uv run python -m tech_blog_mlflow.generate_candidate \
  --theme '記事のテーマ'
```

出力されたGeneration Run IDと記事Pathを控えます。GPT-OSSでは内部Reasoningとfinal本文が同じ生成枠を使うため、`max_tokens=10000`を使用します。記事本文の目標は5,800〜6,800文字、Length Guardは1,800〜7,000文字のままです。

## 3. レビュー・校正

```bash
uv run python -m evaluation.review_candidate \
  --article '<生成記事PATH>' \
  --source-run-id '<GENERATION_RUN_ID>' \
  --dry-run

uv run python -m evaluation.review_candidate \
  --article '<生成記事PATH>' \
  --source-run-id '<GENERATION_RUN_ID>'
```

校正前記事、指摘JSON、校正後記事は別々に保存されます。表示されたReview Run IDと校正後記事Pathを控えます。

## 4. 二重評価

```bash
uv run python -m evaluation.evaluate_candidate_dual \
  --article '<校正後記事PATH>' \
  --source-run-id '<REVIEW_RUN_ID>' \
  --dry-run

uv run python -m evaluation.evaluate_candidate_dual \
  --article '<校正後記事PATH>' \
  --source-run-id '<REVIEW_RUN_ID>'
```

Qwen Primary Judgeの後にModelを解放してGemma Independent Judgeを起動するため、2つを同時にメモリへ載せません。MLflow UIで`judge_role=primary`と`judge_role=independent`を比較してください。

## 5. 採用条件

- 3〜5テーマで同じPrompt、記事長目標、Code Scorerを使う
- SwallowではReasoning込み`max_tokens=10000`を固定する
- Baselineより技術的正確性・有用性・再現性が改善する
- QwenとGemmaのScore差が大きい記事は人手Reviewする
- 架空の実測値や引用が増えていない
- 生成時間とPeak Memoryが運用可能な範囲にある

条件を満たすまで、既存のRegistry alias `production`は変更しません。
