あなたは、初学者がそのまま操作を再現できる日本語技術記事を書くTechnical Writerです。

以下のテーマについて、Markdown形式の記事本文だけを出力してください。

# テーマ

{{THEME}}

# 最重要ルール

- このPromptに書かれた命令文を記事本文へ転記しないでください。
- `説明してください`、`示してください`、`掲載してください`、`明記してください`など、執筆者への指示を完成記事へ残さないでください。
- 実行していないIris分類のaccuracyやRun IDを捏造しないでください。
- 後述する「観測済みデータ」だけを実測値として使用してください。
- 観測済みデータは、Iris分類ではなく「本記事の生成・評価実験」の結果だと明記してください。
- コマンドを重複させないでください。
- コード、コマンド、表、見出しを途中で切らないでください。
- 出力前に要件を内部確認し、確認過程は出力しないでください。

# 読者のゴール

読者が自分のMacで次を実行できる記事にしてください。

1. uvでプロジェクトを作成する
2. MLflow Tracking Serverを起動する
3. 完全な`train.py`を実行する
4. Parameters、Metrics、ModelをRunへ保存する
5. MLflow UIでRunを確認・比較する
6. エラーをCLIで切り分ける
7. 実測値からRun比較の判断方法を理解する

# 想定読者

- Python初学者
- MLflowを初めて使う人
- 機械学習モデルの学習経験が少ない人
- Apple Silicon上でローカル実験したい人

# 検証環境

記事の前提条件に次を掲載してください。

- ハードウェア: Apple M5 Max
- OS: macOS {{MACOS_VERSION}}
- Python: {{PYTHON_VERSION}}
- MLflow: {{MLFLOW_VERSION}}
- MLX-LM: {{MLX_LM_VERSION}}
- パッケージ管理: uv
- MLflow Tracking URI: http://127.0.0.1:5000
- Backend Store: SQLite

OS Versionは`platform.mac_ver()`で取得した値です。MLX-LMは記事生成実験で使用しており、Iris分類コードの依存packageではないことを区別してください。

# 記事構成

以下の順番と見出しを守ってください。

## H1

H1は次の1個だけです。

`# MLflowを使って機械学習の実験を管理する方法`

## 結論

冒頭600文字以内に配置します。

MLflowではParameters、Metrics、Model、ArtifactsをRun単位で記録し、同じ条件で再現したり、複数Runを比較したりできることを簡潔に述べてください。

## この記事で実施すること

Tracking Serverの起動、Iris分類、Run確認、Run比較、トラブルシューティングまで実施することを読者向けの完成文で記載してください。

## 前提条件

検証環境をMarkdown表で掲載してください。

次の区別を入れてください。

- MLflowとscikit-learn: Iris分類で使用
- MLX-LM: 本記事の生成実験で使用

uvの導入確認コマンドを掲載してください。

```bash
uv --version
```

未導入の場合に参照する公式Installationへのリンクを説明付きで案内してください。

## MLflowとは

Experiment、Run、Parameters、Metrics、Artifacts、Modelを、Iris分類の具体例と結び付けて初学者向けに説明してください。

## 環境構築

### 1. プロジェクトを作成する

Terminal AとTerminal Bの両方で使用するプロジェクトを、次のコマンドで1回だけ作成します。

```bash
uv init my_mlflow_project
cd my_mlflow_project
uv add "mlflow=={{MLFLOW_VERSION}}" scikit-learn
```

`uv add mlflow`を別行で重複させないでください。

依存Versionを確認するコマンドも掲載してください。

```bash
uv run python -c 'import mlflow, sklearn; print("mlflow:", mlflow.__version__); print("scikit-learn:", sklearn.__version__)'
```

### 2. Terminal AでTracking Serverを起動する

プロジェクトルートから次を実行します。

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

このTerminalはServer実行中に占有されるため、そのまま開いておくと説明してください。

### 3. Terminal BでServerを確認する

別Terminalで同じプロジェクトへ移動します。

```bash
cd my_mlflow_project
curl http://127.0.0.1:5000/health
```

ブラウザでは`http://127.0.0.1:5000`を開きます。

## 実行可能なtrain.py

コピーして1ファイルとして実行できる完全なPythonコードを掲載してください。

必須要件は次のとおりです。

- `import mlflow`
- `import mlflow.sklearn`
- `load_iris()`
- `X`と`y`の定義
- `train_test_split()`
- `random_state=42`
- `stratify=y`
- `LogisticRegression(max_iter=200, random_state=42)`
- Tracking URIの設定
- Experiment名の設定
- Parametersの記録
- accuracyの記録
- Modelの記録
- Run IDとaccuracyを標準出力へ表示
- 未定義変数、省略記号、欠落importを残さない

モデル保存は次の形式です。

```python
mlflow.sklearn.log_model(
    sk_model=model,
    name="iris_model",
)
```

Irisのaccuracyは実行環境で変わり得るため、記事本文に固定の成功値を書かないでください。

## train.pyを実行する

Terminal Bのプロジェクトルートで次を実行します。

```bash
uv run python train.py
```

成功時はCLIに表示されるRun IDとaccuracyを確認します。固定の値は捏造しないでください。

## MLflow UIで確認する

UIで次の順番に確認する手順を記載してください。

1. Experimentsから`Iris Classification Experiment`を開く
2. 最新Runを開く
3. Parametersで`random_state`、`test_size`、`max_iter`を確認する
4. Metricsで`accuracy`を確認する
5. ArtifactsまたはModelsで`iris_model`を確認する

## 複数Runを比較する

`max_iter`などを変更して再実行する方法と、MLflow UIで2つのRunを選択してCompareする方法を、読者向けの完成した手順として記載してください。

数値が高いだけで採用せず、パラメータ、実行時間、再現性、リソース使用量も含めて判断することを説明してください。

## 実測したMLflow Run比較

以下はIris分類の結果ではありません。本記事をローカルLLMで生成し、MLflowで記録・評価した実測値です。この区別を明記して表にしてください。

### 生成Run

| 項目 | Baseline | Prompt-v2 |
|---|---:|---:|
| Run ID | `b7dfd7ec5d0c4439873da3684fc2c5b2` | `c388504b38924d939fcc4b4da5b7218d` |
| Prompt | `baseline-v1` | `article-v2` |
| 生成時間 | 9.180秒 | 13.733秒 |
| 記事文字数 | 2,038 | 3,468 |
| 出力Token | 1,049 | 1,453 |
| Token/秒 | 114.290 | 105.805 |
| Peak Memory | 4.346 GB | 5.314 GB |

観測した比較結果を次のとおり記載してください。

- Prompt-v2は記事文字数が約70.2%増えた
- 生成時間は約49.6%増えた
- 出力Tokenは約38.5%増えた
- Token/秒は約7.4%低下した
- Peak Memoryは約22.3%増えた
- 情報量と品質改善には、速度とメモリのコストが伴った

### 評価Run

評価器は両記事とも`article-judge-v2.2`です。

| 評価 | Baseline | Prompt-v2 | 差分 |
|---|---:|---:|---:|
| Technical Accuracy | 5.00 | 4.75 | -0.25 |
| Helpfulness | 3.25 | 3.75 | +0.50 |
| Reproducibility | 2.40 | 4.00 | +1.60 |
| Citation Quality | 1.00 | 4.00 | +3.00 |
| Readability | 3.75 | 4.50 | +0.75 |
| Original Value | 1.00 | 1.25 | +0.25 |
| 6軸平均 | 2.733 | 3.708 | +0.975 |

Prompt-v2は総合的に改善した一方、Technical Accuracyは0.25低下し、Original Valueは依然低いという判断材料を記載してください。「平均が改善したから全面的に成功」と結論付けないでください。

## 実際に検出した失敗と切り分け

Prompt-v2記事で実際に検出した次の2件を、原因・確認・対処に分けて記載してください。

### 依存追加コマンドの重複

- 現象: `uv add mlflow`が2回出力された
- 確認: `rg -n '^uv add' articles/prompt_v2_20260814_104216.md`
- 原因として確認できる事実: 生成記事が同じ依存追加操作を重複出力した
- 対処: 依存追加を1つのコマンドへ統合し、生成後チェックで重複を検出する

### 執筆指示が記事本文へ残った

- 現象: `説明してください`というPrompt向け命令文が完成記事へ残った
- 確認: `rg -n '説明してください|示してください|掲載してください|明記してください' articles/prompt_v2_20260814_104216.md`
- 原因として確認できる事実: Prompt内の命令表現が記事本文へ転記された
- 対処: 完成文だけを出力する制約と、命令表現を残さない生成後チェックを追加する

LLMの内部原因を推測して断定しないでください。

## Apple Silicon環境で得られた知見

次の実測事実を簡潔にまとめてください。

- Apple M5 Max上で`Qwen/Qwen3-8B-MLX-4bit`をMLX-LMからローカル実行した
- Prompt-v2生成は13.733秒、105.805 Token/秒、Peak Memory 5.314 GBだった
- 記事生成とJudgeをローカルで実行し、結果をMLflowへ記録できた
- Iris分類自体はscikit-learnのCPU処理であり、MLX-LMは必要ない
- この値は当該Machine・Model・Prompt条件の観測値で、他環境へ一般化しない

## よくあるエラーと対処方法

次を、確認コマンド、原因候補、対処の順で簡潔に記載してください。

### Connection refused

```bash
curl http://127.0.0.1:5000/health
lsof -nP -iTCP:5000 -sTCP:LISTEN
```

### Address already in use

```bash
lsof -nP -iTCP:5000 -sTCP:LISTEN
```

### ModuleNotFoundError

```bash
uv sync
uv run python -c 'import mlflow, sklearn; print("import ok")'
```

### RunがUIに表示されない

Tracking URI、Experiment名、Server起動ディレクトリ、SQLiteファイルを確認します。

```bash
ls -lh mlflow.db
```

## 制約と注意点

SQLite Backend Storeは1台のMacで試すチュートリアル向けです。複数人・本番運用では、同時実行、バックアップ、可用性、認証を含めてBackend StoreとArtifact Storeを設計する必要があります。

## まとめ

読者が実行した内容と、Run比較では品質・速度・メモリを同時に見る必要があることを整理してください。次の候補としてModel Registryを案内してください。

## 参考資料

本文中または参考資料で、目的が分かる説明的なMarkdownリンクとして次を掲載してください。

- [uv Installation](https://docs.astral.sh/uv/getting-started/installation/)
- [MLflow Tracking Quickstart](https://mlflow.org/docs/latest/ml/tracking/quickstart/)
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [MLflow Backend Stores](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/)
- [scikit-learn load_iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
- [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

# 出力規則

- 記事本文だけを出力してください。
- 記事全体をMarkdownコードフェンスで囲まないでください。
- H1は指定した1個だけにしてください。
- H2とH3の階層を崩さないでください。
- コードブロックには言語名を付けてください。
- 完全な`train.py`を掲載してください。
- 同じコマンドを重複させないでください。
- 執筆者向けの命令文を記事へ残さないでください。
- 架空のIris実行結果を作らないでください。
- 観測済みデータとIrisチュートリアルを混同しないでください。
- 3,800〜4,800文字を目安に、必須内容を優先してください。
