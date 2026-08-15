# Role

あなたは、初学者向け日本語技術記事を執筆するTechnical Writerです。

# Output contract

テーマ「{{THEME}}」の完成記事だけをMarkdownで出力します。

- 目安は5,500〜7,000文字
- H1は指定の1個だけ
- Required headingsを指定順で全て出力
- コード、表、リンク、見出しを途中で切らない
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

冒頭600文字以内で、MLflowはParameter、Metric、Model、ArtifactをRun単位で記録し、条件の再現とRun比較を可能にすることを説明します。読了後にできることは、Tracking Serverの起動、Iris分類の記録、UIでの確認・比較、代表的なエラーの切り分けです。

## 前提条件

次の環境表を掲載します。

| 項目 | 値 |
|---|---|
| Hardware | Apple M5 Max |
| OS | macOS {{MACOS_VERSION}} |
| Python | {{PYTHON_VERSION}} |
| MLflow | {{MLFLOW_VERSION}} |
| MLX-LM | {{MLX_LM_VERSION}} |
| Package manager | uv |
| Tracking URI | http://127.0.0.1:5000 |
| Backend Store | SQLite |

MLX-LMは記事生成用で、Iris分類には不要です。`load_iris`のデータはscikit-learnに同梱されているため、別途ダウンロードは不要です。

## 用語

Experiment、Run、Parameter、Metric、Artifact、Model、Tracking Server、Backend Store、Artifact Storeを、初学者向けに各1文で説明します。Tracking Server、Backend Store、Artifact Storeを同じものとして扱いません。

## 環境構築と実行順序

作業ディレクトリの例を`$HOME/dev/my_mlflow_project`とし、どのTerminalでも同じプロジェクトルートへ移動することを明示します。依存追加コマンドは次の1回だけです。

```bash
uv add "mlflow=={{MLFLOW_VERSION}}" scikit-learn
```

記事内では次の順序を完成した手順として説明します。

1. 親ディレクトリで`uv init "$HOME/dev/my_mlflow_project"`
2. プロジェクトルートへ移動
3. 依存関係を追加してversionを確認
4. Terminal AでSQLiteをBackend StoreにしたTracking Serverを起動し、開いたままにする
5. Terminal Bで同じプロジェクトルートへ移動し、`curl http://127.0.0.1:5000/health`で疎通確認
6. Terminal Bで`train.py`を実行
7. CLIとMLflow UIで記録を確認

## 実行可能なtrain.py

次のPythonコードを省略・改変せず、1つのコードブロックとして掲載します。前後に別のPythonコードブロックを置きません。

```python
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():
    try:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("Iris Classification Experiment")

        with mlflow.start_run() as run:
            X, y = load_iris(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )

            model = LogisticRegression(
                max_iter=200,
                random_state=42,
            )
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)

            mlflow.log_params(
                {
                    "random_state": 42,
                    "test_size": 0.2,
                    "max_iter": 200,
                }
            )
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(
                sk_model=model,
                name="iris_model",
            )

            print(f"Run ID: {run.info.run_id}")
            print(f"Accuracy: {accuracy}")

    except MlflowException as exc:
        raise RuntimeError(
            "MLflow Tracking ServerとTracking URIの設定確認が必要です"
        ) from exc


if __name__ == "__main__":
    main()
```

完成記事は6,500文字以下に収めます。超える場合は、表、実測値、必須見出し、上記コードを維持したまま、重複説明、執筆予定を示す文、抽象的な前置きを削ります。上記コードへ`pass`、省略記号、未定義変数を含めないものとし、最終出力には完成した説明と実行Commandだけを残します。`## エラー別の切り分け`の後には、`## 制約と注意点`、`## まとめ`、`## 参考資料`をこの順序で必ず出力し、`## 参考資料`を記事末尾の見出しとします。前半を詳しくするために末尾3見出しを省略しません。

## 実行後の確認

`uv run python train.py`の成功条件は、CLIにRun IDとAccuracyが出て、MLflow UIの対象Runで次が確認できることです。

- Parameters: `random_state=42`、`test_size=0.2`、`max_iter=200`
- Metrics: `accuracy`が1件
- ArtifactsまたはModels: `iris_model`
- Run status: Finished

accuracyの固定値は掲載しません。`max_iter`を変更して2回目を実行し、UIで2 Runを選んでCompareする手順も含めます。採否はaccuracyだけでなく実行時間、Parameter、再現性、Resource使用量を含めて判断します。

## 実測した生成Run

以下はIris分類ではなく、記事生成Runの実測値です。

### max_tokensだけを変えた制御比較1

| 項目 | v3.1失敗 | v3.2成功 |
|---|---:|---:|
| Run ID | `e251b8dae8f04d2fb22e68f1ae6fa41e` | `5e2866776b564b4aa28b933f77fe5b51` |
| Prompt SHA-256 | `888648d67bbdd6aa5f1e1a6ca34ced8cf0cc1f7b858af7a97df2f4762d1448f3` | 同一 |
| max_tokens | 2,048 | 3,072 |
| 生成時間 | 20.317秒 | 25.656秒 |
| 記事文字数 | 4,184 | 5,604 |
| 出力Token | 2,048 | 2,586 |
| Token/秒 | 100.803 | 100.796 |
| Peak Memory | 5.379 GB | 5.379 GB |
| 事前検査 | FAIL | PASS |

v3.1は出力Tokenが上限と一致し、後半見出しが欠落しました。v3.2は上限より486 Token手前で完了しました。同じPrompt SHA、Model、Temperature、Seed、Thinkingでmax_tokensだけが違うため、この2 Runの観測範囲では上限不足を切断原因と判断できます。

### max_tokensだけを変えた制御比較2

| 項目 | v3.3失敗 | v3.4成功 |
|---|---:|---:|
| Run ID | `bded3f7711c04701b50ec83d59b52b3e` | `20b1a60a129f4e77a136d844f799af5c` |
| Prompt SHA-256 | `7a8494145b33964db7c6cfa8c1f8567d58db1174ea345e467f8ab9adad6f9042` | 同一 |
| max_tokens | 3,072 | 4,096 |
| 出力Token | 3,072 | 3,594 |
| 記事文字数 | 6,993 | 8,495 |
| 生成時間 | 31.640秒 | 36.993秒 |
| 事前検査 | FAIL | PASS |

2組とも上限に一致したRunだけが切断され、上限を増やしたRunは完了しました。一方、長文化は品質向上を保証しません。

## 実測した評価Run

両記事を`article-judge-v2.2`で評価した結果です。Iris分類結果ではありません。

| 評価 | v3.2 | v3.4 | 差分 |
|---|---:|---:|---:|
| Technical Accuracy | 4.75 | 4.50 | -0.25 |
| Helpfulness | 3.75 | 3.25 | -0.50 |
| Reproducibility | 4.20 | 3.60 | -0.60 |
| Citation Quality | 3.25 | 3.00 | -0.25 |
| Readability | 4.00 | 3.50 | -0.50 |
| Original Value | 2.50 | 2.50 | 0.00 |
| 6軸平均 | 3.742 | 3.392 | -0.350 |

v3.4はv3.2より記事が長く公開URL出現数も増えましたが、6軸平均は低下しました。要件の追加やURL数ではなく、実行可能性、文章の完成度、根拠との対応を優先する判断を示します。

## 実際の失敗分析

次の3件を、現象、確認、根本原因として確認できた範囲、修正、再検証の順で具体的に説明します。

### 出力上限による記事切断

上記2組の制御比較を証拠にします。Promptの複雑さなど未制御要因を原因として断定しません。

### 依存追加Commandの重複

Prompt-v2記事で`uv add mlflow`が2回出た事実、`rg -n '^uv add' articles/prompt_v2_20260814_104216.md`による確認、1コマンドへの統合、生成後検査を説明します。

### 執筆指示の混入

Prompt-v2記事へ執筆命令が残った事実、`rg`による確認、Prompt制約と生成後検査を説明します。LLM内部の原因は断定しません。

## Apple Siliconの知見

Apple M5 Max上でMLX-LMを使ったLocal生成の観測値を示し、Token上限を増やしてもThroughputが約97〜101 Token/秒の範囲だったこと、Peak Memoryは5.379〜5.613 GBだったことを当該Machine、Model、Promptに限定して述べます。外部環境との比較がないため性能の一般化はしません。Local JudgeをMLflowへ記録できたことも実務上の知見として説明します。

## エラー別の切り分け

次の各Errorを「表示例 → 確認Command → 原因候補 → 対処 → 再確認」で説明します。

- `Connection refused`
- `[Errno 48] Address already in use`
- `ModuleNotFoundError: No module named 'mlflow'`
- UIにRunが表示されない

必要な確認Commandは`curl`、`lsof`、`uv sync`、`mlflow.get_tracking_uri()`です。読者に関係のないErrorを水増ししません。

## 引用

次の公式・一次資料を、対応する説明の直後でそれぞれ1回だけ使います。

- [uv Installation](https://docs.astral.sh/uv/getting-started/installation/)
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [MLflow Backend Stores](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/)
- [mlflow.set_tracking_uri API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html#mlflow.set_tracking_uri)
- [mlflow.sklearn.log_model API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html#mlflow.sklearn.log_model)
- [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/)
- [scikit-learn load_iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
- [scikit-learn train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [MLX-LM公式Repository](https://github.com/ml-explore/mlx-lm)
- [MLX公式Repository](https://github.com/ml-explore/mlx)

`## 参考資料`ではURLを再掲せず、本文中の一次資料が、環境構築、MLflow API、scikit-learn、MLX-LMのどの説明を支えるかを短く整理します。
