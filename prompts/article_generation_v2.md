あなたは、初学者が実際に操作を再現できる日本語技術記事を書くTechnical Writerです。

以下のテーマについて、Markdown形式の技術記事を作成してください。

# テーマ

{{THEME}}

# この記事の目的

読者がMLflowの概要を理解するだけでなく、自分のMacで以下を実行できる状態にしてください。

1. MLflow Tracking Serverを起動する
2. 実行可能な機械学習コードを動かす
3. Parameters、Metrics、ModelをMLflowへ保存する
4. MLflow UIでRunを確認する
5. 複数Runを比較する
6. よくあるエラーを切り分ける

# 想定読者

- Python初学者
- MLflowを初めて使う人
- 機械学習モデルの学習経験が少ない人
- Apple Silicon上でローカル実験したい人

# 検証環境

記事の「前提条件」に以下を明記してください。

- ハードウェア: Apple M5 Max
- OS: macOS {{MACOS_VERSION}}
- Python: {{PYTHON_VERSION}}
- MLflow: {{MLFLOW_VERSION}}
- MLX-LM: {{MLX_LM_VERSION}}
- パッケージ管理: uv
- MLflow Tracking URI: http://127.0.0.1:5000
- Backend Store: SQLite

# 必須構成

次の順番を守ってください。

# MLflowを使って機械学習の実験を管理する方法

## 結論

記事の冒頭600文字以内に置いてください。

MLflowを使うことで、Parameters、Metrics、Model、ArtifactsをRun単位で記録し、実験同士を比較できることを簡潔に説明してください。

## この記事で実施すること

この記事を最後まで進めると何ができるかを説明してください。

## 前提条件

検証環境をMarkdown表で掲載してください。

必要なソフトウェアとバージョンを明記してください。

## MLflowとは

以下の用語を初学者向けに説明してください。

- Experiment
- Run
- Parameters
- Metrics
- Artifacts
- Model

## 環境構築

番号付きのH3見出しを使用してください。

### 1. プロジェクトを作成する

`uv init`、仮想環境、必要packageの導入方法を示してください。

### 2. MLflow Tracking Serverを起動する

次の構成を使用してください。

- SQLite Backend Store
- Host: 127.0.0.1
- Port: 5000

コマンドを省略せず掲載してください。

### 3. Serverの起動を確認する

`curl`とブラウザの両方の確認方法を示してください。

## 実行可能なサンプルコード

コードは、コピーして1ファイルとして実行できる完全な内容にしてください。

次の条件を満たしてください。

- scikit-learnのIris datasetを使用する
- `X`と`y`をコード内で定義する
- `train_test_split()`で学習用とテスト用に分割する
- `LogisticRegression`を使用する
- `random_state`を固定する
- MLflow Tracking URIを明示する
- Experiment名を設定する
- `mlflow.start_run()`を使用する
- Parametersを記録する
- accuracyをMetricsへ記録する
- Modelを記録する
- Run IDを表示する
- 省略記号を使用しない
- 未定義変数を残さない

MLflow 3系の書き方に合わせて、モデル保存では次の形式を使用してください。

mlflow.sklearn.log_model(
    sk_model=model,
    name="iris_model",
)

コードファイル名は`train.py`としてください。

## サンプルコードを実行する

実行コマンドと、成功時に確認すべき内容を説明してください。

実行していない数値や架空のaccuracyを、実測結果として断定しないでください。

## MLflow UIで確認する

以下をどこで確認するか説明してください。

- Experiment
- Run
- Parameters
- Metrics
- Model
- Artifacts

## 複数Runを比較する

モデルパラメータを変更して再実行し、MLflow UIで比較する流れを説明してください。

## よくあるエラーと対処方法

最低でも次を掲載してください。

### Connection refused

- MLflow Serverが起動しているか
- Tracking URIが正しいか
- Port 5000がLISTENしているか

### Address already in use

Port 5000を使用中のProcessを確認する方法を示してください。

### ModuleNotFoundError

`uv run`で実行しているか、依存packageが入っているか確認する方法を示してください。

### RunがUIに表示されない

- Experiment名
- Tracking URI
- 接続先のSQLite
- 起動ディレクトリ

を確認するように説明してください。

## 制約と注意点

ローカルSQLite構成はチュートリアルには適するが、複数人による本番運用では構成を検討する必要があることを説明してください。

断定しすぎず、詳細は公式ドキュメントを参照するよう案内してください。

## まとめ

読者が実行した内容と、次に試すべき内容を整理してください。

## 参考資料

以下の公式情報をMarkdownリンクとして掲載してください。

- MLflow Tracking Quickstart
  https://mlflow.org/docs/latest/ml/tracking/quickstart/

- MLflow Tracking
  https://mlflow.org/docs/latest/ml/tracking/

- MLflow Model Registry
  https://mlflow.org/docs/latest/ml/model-registry/

# 出力規則

- 記事本文だけを出力してください。
- Markdownコードフェンスで記事全体を囲まないでください。
- H1は1個だけにしてください。
- H2とH3の階層を崩さないでください。
- コードブロックには言語名を付けてください。
- コマンドとPythonコードを省略しないでください。
- 未定義変数を作らないでください。
- 架空の実行結果を捏造しないでください。
- 2,500〜4,000文字程度に収めてください。
- 一般論だけで終わらず、読者が手元で確認できる内容にしてください。
