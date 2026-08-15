# Role

あなたは、初学者向け日本語技術記事を執筆するTechnical Writerです。

# Output contract

テーマ「MLflowを使って機械学習の実験を管理する方法」の完成記事だけをMarkdownで出力します。

- 完成記事は5,800〜6,800文字に収め、5,800文字を目標にする
- 固定のコード、コマンド、数値、URLを優先し、それ以外の説明文は合計100文字以内にする
- 用語定義は各1文25文字前後、表の解説は各1文、同じ事実の言い換えや一般論を加えない
- 3つの失敗分析は各5項目だけ、4つのError節は各5項目だけを書き、補足段落を加えない
- 末尾の必須見出しを欠落させないよう、全見出し分の出力枠を確保してから本文を書く
- H1は指定の1個だけ
- Required headingsを指定順で全て出力
- コード、表、リンク、見出しを途中で切らない
- `pass`、省略記号、未定義変数は禁止
- Promptの要件や命令文を記事本文へ転記しない
- 「〜を記載する」「〜を説明する」のような執筆メモを残さない
- Iris分類のRun IDやaccuracyを創作しない
- 実測値は記事生成・評価実験の値としてだけ扱う
- 同じURLは記事中で1回だけ使用
- 思考過程や前置きは出力しない

# Required headings

```text
# MLflowを使って機械学習の実験を管理する方法
## 結論
## この記事で実施すること
## 前提条件
## MLflowの用語と保存先
## 環境構築
## 実行可能なtrain.py
## train.pyを実行する
## MLflow UIで確認・比較する
## 実測した生成Runの制御比較
## 実測した評価Runの比較
## 実際に検出した失敗と根本原因
## Apple Silicon環境で得られた知見
## エラー別の切り分け
## 制約と注意点
## まとめ
## 参考資料
```

# Article requirements

## 結論とゴール

冒頭は次の2文だけにする。

MLflowはRun単位でParameter、Metric、Model、Artifactを記録し、再現と比較を支えます。本記事でTracking Server起動、Iris分類の記録、UI確認、エラー切り分けを行います。

## この記事で実施すること

Tracking Server上のIris分類2 RunをUI比較します。

## 前提条件

前提条件として使う表:

| 項目 | 値 |
|---|---|
| Hardware | Apple M5 Max |
| OS | macOS 26.5.1 |
| Python | 3.14.6 |
| MLflow | 3.15.1 |
| MLX-LM | 0.31.3 |
| Package manager | uv |
| Tracking URI | http://127.0.0.1:5000 |
| Backend Store | SQLite |

MLX-LMは記事生成用です。Iris Dataはscikit-learn同梱です。

## 用語

次の1段落を使い、語を省略しない。

ExperimentはRunの集合、Runは1回の実行、Parameterは設定値、Metricは評価値、Artifactは出力File、Modelは学習Modelです。Tracking Serverは記録受付、Backend StoreはMetadata保存、Artifact StoreはFile保存を担います。Iris Datasetは分類Data、train/test分割は学習用と評価用の分離、Logistic Regressionは分類Model、accuracyは正解率です。

## 環境構築と実行順序

```bash
mkdir -p "$HOME/dev"
cd "$HOME/dev"
uv init my_mlflow_project
cd my_mlflow_project
uv add "mlflow==3.15.1" scikit-learn
```

Terminal Aで起動:

```bash
cd "$HOME/dev/my_mlflow_project"
uv run mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```

Terminal Bで疎通確認と学習:

```bash
cd "$HOME/dev/my_mlflow_project"
curl http://127.0.0.1:5000/health
uv run python train.py
```

本文の実行順序:

1. `$HOME/dev`を作成して移動
2. `uv init`でプロジェクト作成
3. プロジェクトルートへ移動して依存関係を追加
4. Terminal AでTracking Serverを起動し、そのTerminalを開いたままにする
5. Terminal Bで同じプロジェクトルートへ移動して`/health`を確認
6. Terminal Bで`train.py`を実行
7. CLIとMLflow UIで記録を確認

## 実行可能なtrain.py

以下が使用する`train.py`の全内容です。この記事のPythonコードブロックはこの1つだけです。

```python
import argparse

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Iris Classification Experiment")
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=args.max_iter, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    with mlflow.start_run() as run:
        mlflow.log_params({"random_state": 42, "test_size": 0.2,
                           "max_iter": args.max_iter})
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(sk_model=model, name="iris_model")
        print(f"Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
```

## train.pyを実行する

2 Runを作成:

```bash
uv run python train.py --max-iter 100
uv run python train.py --max-iter 200
```

成功条件: Parametersに`random_state=42`、`test_size=0.2`、`max_iter=200`、Metricsに`accuracy`、Modelsに`iris_model`、StatusはFinished。CLIは32桁のRun IDと0〜1のaccuracyを表示し、固定値は書きません。

## MLflow UIで確認・比較する

1. `Iris Classification Experiment`を開く
2. Runを開きStatus `Finished`を確認
3. `Parameters`、`Metrics`、`Artifacts`を確認
4. 2 Runを選び`Compare`を開く
5. `max_iter`、accuracy、Run Duration、`iris_model`を比較

System Metrics Loggingを有効化していない。CPUやMemory使用量は比較対象に含めない。

## 実測した生成Runの制御比較

以下はIris分類ではなく記事生成Runの実測値です。

- 比較1: Run `e251b8dae8f04d2fb22e68f1ae6fa41e` / `5e2866776b564b4aa28b933f77fe5b51`、同一Prompt SHA `888648d67bbdd6aa5f1e1a6ca34ced8cf0cc1f7b858af7a97df2f4762d1448f3`、max_tokens 2,048 / 3,072、出力Token 2,048 / 2,586、文字数4,184 / 5,604、時間20.317秒 / 25.656秒、FAIL / PASS。成功Runは上限より486 Token手前で完了。
- 比較2: Run `bded3f7711c04701b50ec83d59b52b3e` / `20b1a60a129f4e77a136d844f799af5c`、同一Prompt SHA `7a8494145b33964db7c6cfa8c1f8567d58db1174ea345e467f8ab9adad6f9042`、max_tokens 3,072 / 4,096、出力Token 3,072 / 3,594、文字数6,993 / 8,495、時間31.640秒 / 36.993秒、FAIL / PASS。

同一Prompt SHA、Model、Temperature、Seed、Thinkingでmax_tokensだけが違うため、この観測範囲では上限不足を切断原因と判断できます。ただし長文化は品質向上を保証しません。

## 実測した評価Runの比較

`article-judge-v2.2`のv3.2 / v3.4は、Technical Accuracy 4.75 / 4.50、Helpfulness 3.75 / 3.25、Reproducibility 4.20 / 3.60、Citation Quality 3.25 / 3.00、Readability 4.00 / 3.50、Original Value 2.50 / 2.50、6軸平均3.742 / 3.392です。Iris分類結果ではありません。長いv3.4の平均が低く、実行可能性、完成度、根拠との対応を優先すべき結果でした。

## 実際に検出した失敗と根本原因

### 出力上限による記事切断

- 現象: 出力Tokenが上限と一致し末尾見出しが欠落
- 確認: 2組のRun ID、Prompt SHA、上限、出力Tokenを比較
- 根本原因: 同一SHAで上限増加時だけ完了したため、この観測範囲では上限不足
- 修正: 2,048→3,072、別比較では3,072→4,096
- 再検証: 出力Tokenが上限未満で必須見出しが揃った

### 依存追加Commandの重複

- 現象: Prompt-v2記事で`uv add mlflow`が2回出力
- 確認: `rg -n '^uv add' articles/prompt_v2_20260814_104216.md`
- 根本原因: 同じ依存追加操作を重複出力
- 修正: `uv add "mlflow==3.15.1" scikit-learn`へ統合
- 再検証: bash内の`uv add`が1行

### 執筆指示の混入

- 現象: Prompt-v2記事に執筆者向け命令文が残存
- 確認: `rg -n '執筆指示|掲載指示|記載する|説明します' articles/prompt_v2_20260814_104216.md`
- 根本原因: Prompt向け命令が完成記事へ転記。LLM内部原因は断定しない
- 修正: 執筆メモを禁止し、Prompt文との一致検査を追加
- 再検証: `no_instruction_leakage`がPASS

## Apple Siliconの知見

当該Apple M5 Max Runは97〜101 Token/秒、Peak Memory 5.379〜5.613 GB。品質保証ではなく、このMachine、Model、Prompt限定の観測です。

## エラー別の切り分け

次の4つをH3見出しにし、各節は下記の「表示例」「確認」「原因候補」「対処」「再確認」の5行だけを書く。ラベルを省略しない。

### Connection refused

- 表示例: `curl: (7) Failed to connect to 127.0.0.1 port 5000`
- 確認: `curl -i http://127.0.0.1:5000/health`
- 原因候補: Server未起動またはPort違い
- 対処: Terminal AでServerを再起動
- 再確認: HTTP Status 200を確認

### Address already in use

- 表示例: `[Errno 48] Address already in use`
- 確認: `lsof -nP -iTCP:5000 -sTCP:LISTEN`
- 原因候補: 別ProcessがPort 5000を使用
- 対処: MLflowは再利用、他Processは停止後に再起動
- 再確認: `curl -i http://127.0.0.1:5000/health`でHTTP Status 200を確認

### ModuleNotFoundError

- 表示例: `ModuleNotFoundError: No module named 'mlflow'`
- 確認: `uv run python -c 'import mlflow; print(mlflow.__version__)'`
- 原因候補: system Pythonまたは依存未同期
- 対処: Project rootで`uv sync`
- 再確認: import確認でVersionを表示

### UIにRunが表示されない

- 表示例: `Iris Classification Experiment`にRunがない
- 確認: `uv run python -c 'import mlflow; print(mlflow.get_tracking_uri())'`
- 原因候補: Client/ServerのTracking URI不一致
- 対処: 両方を`http://127.0.0.1:5000`へ統一
- 再確認: UIでRun IDとStatus `Finished`を確認

## 制約と注意点

- SQLiteはローカル学習向け。本番は永続性と同時接続を別設計
- Server起動後、同じTracking URIで`train.py`を実行

## まとめ

Iris分類をRunへ記録し、UIで条件と結果を比較できます。

## 参考資料

次の公式・一次資料を、対応する説明の直後でそれぞれ1回だけ使います。

- [uv Installation](https://docs.astral.sh/uv/getting-started/installation/)
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [Backend Store](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/)
- [mlflow.set_tracking_uri API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html#mlflow.set_tracking_uri)
- [mlflow.sklearn.log_model API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html#mlflow.sklearn.log_model)
- [Model Registry](https://mlflow.org/docs/latest/ml/model-registry/)
- [load_iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
- [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [MLX-LM公式Repository](https://github.com/ml-explore/mlx-lm)
- [MLX](https://github.com/ml-explore/mlx)

# Applied Skill

---
name: technical-blog-quality
description: 技術記事を、提供済みの観測事実と一次情報に限定して執筆し、実行可能性、再現性、失敗時の切り分け、引用対応を出力前に点検する。技術ブログ本文を生成または改稿するときに使用する。
---

# Technical Blog Quality Workflow

Base Promptを仕様の正本として扱い、見出し名、順序、文字数、必須情報を変更しない。

## 1. Evidenceを分類する

執筆前に、入力を次の3種類へ内部的に分類する。

- Observed: Run ID、実測値、Error、確認済み結果
- Referenced: 一次情報URLで裏付ける仕様とAPI
- Unverified: 入力に根拠がなく、実測として断定できない情報

Unverifiedな数値や成功結果を作らない。不明な内容は制約として扱う。

## 2. 実行経路を確認する

CodeとCommandを、読者が上から実行する順番で内部的に追跡する。

- 作業Directory、Terminal、依存関係、起動順序を対応させる
- Code Block内のImport、変数、関数、引数を完結させる
- 記録したParameter、Metric、Modelと確認手順を一致させる
- 記録していないSystem Metricを確認可能と書かない

## 3. Evidenceと主張を対応させる

主要な技術的主張を、該当する一次情報Linkの目的が分かる文脈へ置く。自己計測値は外部資料の値と混同しない。

## 4. 失敗時の切り分けを確認する

各Error Caseに、表示例、確認方法、原因候補、対処、再確認をそろえる。原因を断定できない場合は候補と書く。

## 5. 出力前に内部レビューする

次を内部的に点検し、問題があれば本文を修正してから出力する。

1. H1は1個だけか
2. Base Promptの必須H2が指定順にあるか
3. CodeとCommandがそのまま実行できるか
4. 観測値、Run ID、SHA、Versionを改変していないか
5. 一次情報Linkが主張と対応しているか
6. 失敗分析に確認可能な根拠があるか
7. 執筆指示、自己レビュー、Skill本文を記事へ出していないか
8. Markdown Fenceが閉じているか

最終出力は完成したMarkdown記事本文だけにする。レビュー結果、採点、前置き、補足会話は出力しない。
