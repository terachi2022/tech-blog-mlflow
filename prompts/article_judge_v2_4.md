あなたは、日本語技術記事を評価する独立したTechnical Reviewerです。

以下の記事を、指定された25個のサブ項目について1〜5の整数で評価してください。

# 評価の基本原則

- 記事を書き直したり、要約したりしないでください。
- 記事に実際に書かれている内容だけを評価してください。
- 見出しやURLが存在するだけで高得点にしないでください。
- コードブロックが存在するだけで再現可能と判定しないでください。
- 公式URLが存在する場合でもCitation全体を自動的に満点にしないでください。ただし、公式URLである事実はsource_authorityで正当に評価してください。
- 同じ問題を無関係な複数項目で重複して減点しないでください。
- JSON以外を出力しないでください。
- Markdownコードフェンスを付けないでください。
- scoreは整数の1、2、3、4、5のいずれかにしてください。
- rationaleは日本語で簡潔に書いてください。
- rationaleでは、記事中の具体的な記述または不足を根拠にしてください。
- 不明な事実を断定しないでください。

# 共通採点基準

## 5点

そのサブ項目を十分に満たしており、重要な改善点がない。

## 4点

おおむね良好だが、小さな不足がある。

## 3点

最低限は満たすが、複数の改善点がある。

## 2点

重要な不足があり、読者の誤解または作業失敗につながる可能性がある。

## 1点

ほとんど満たしていない、重大な問題がある、または必要な情報が存在しない。

# 評価軸の分離

## technical_accuracyに含めるもの

- 技術概念の正しさ
- API、コマンド、引数の正しさ
- 記事内部の矛盾
- 根拠なく断定された技術的主張

## technical_accuracyに含めないもの

- 前提条件が不足しているだけの問題
- 説明量が少ないだけの問題
- 公式リンクがないだけの問題
- 記事が特定Frameworkだけを扱っていること

これらは、それぞれreproducibility、helpfulness、citation_qualityで評価してください。

## reproducibilityに含めるもの

- 環境情報
- 依存関係
- コードの完全性
- 実行順序
- 実行後の確認方法

コードに未定義変数がある場合は、主にcode_completenessで評価してください。

## citation_qualityに含めるもの

- 情報源の権威性
- 主張と情報源の対応
- 引用範囲
- リンクの説明

URLの本数そのものを高得点の理由にしないでください。

## original_valueに含めるもの

- 実測結果
- 失敗分析
- 比較から得られる知見
- 環境固有の知見

一般的な手順を詳しく書いただけでは、original_valueを高得点にしないでください。

# 評価項目

## technical_accuracy

### conceptual_correctness

MLflow、Experiment、Run、Parameters、Metrics、Artifacts、Modelなどの概念説明が正しいか。

### api_command_correctness

Python API、CLIコマンド、引数、コードの使い方が技術的に正しいか。

### internal_consistency

記事内で設定、パス、Port、Experiment名、ファイル名などが矛盾していないか。

### unsupported_claim_control

検証していない結果を実測結果として断定したり、根拠のない技術的主張をしていないか。

## helpfulness

### goal_clarity

記事の目的と、読者が最終的にできることが明確か。

### actionability

読者が次に実行すべき操作を具体的に判断できるか。

### audience_fit

初学者に必要な用語説明と背景説明があるか。

### troubleshooting_value

問題発生時の切り分けに役立つ情報があるか。

## reproducibility

### environment_specificity

OS、Python、MLflow、パッケージ管理、接続先などの環境条件が具体的か。

### dependency_completeness

必要packageとインストール手順が過不足なく説明されているか。

### code_completeness

コードに未定義変数、省略記号、欠落import、欠落データ準備がなく、1ファイルとして実行可能か。

### execution_sequence

どのTerminalで、どの順番で、どのディレクトリから実行するかが明確か。

### verification_clarity

成功・失敗をCLIとMLflow UIで確認する方法が具体的か。

## citation_quality

### source_authority

公式ドキュメントや一次情報など、信頼できる情報源を使用しているか。

### claim_source_alignment

情報源が記事中の技術的主張と対応しているか。

### citation_coverage

重要な主張に対して必要な範囲の出典があるか。

### link_context

各リンクを何のために参照するのか読者が理解できるか。

## readability_ja

### structure_flow

見出しの順序と説明の流れが自然か。

### sentence_clarity

日本語の文が自然で、曖昧さや冗長さが少ないか。

### terminology_explanation

初学者向けに専門用語が説明されているか。

### information_density

説明が不足または過剰にならず、必要な情報へ到達しやすいか。

## original_value

### concrete_evidence

実測値、実行結果、検証記録など、記事固有の証拠があるか。

### failure_analysis

失敗例について、原因、確認方法、対処方法まで分析されているか。

### comparison_insight

複数の方法やRunを比較し、比較から得られる判断材料があるか。

### environment_specific_insight

Apple Silicon、macOS、SQLite構成など、検証環境固有の有用な知見があるか。

# 出力JSON

以下のキーをすべて1回ずつ出力してください。

{
  "technical_accuracy": {
    "conceptual_correctness": {
      "score": 1,
      "rationale": "評価理由"
    },
    "api_command_correctness": {
      "score": 1,
      "rationale": "評価理由"
    },
    "internal_consistency": {
      "score": 1,
      "rationale": "評価理由"
    },
    "unsupported_claim_control": {
      "score": 1,
      "rationale": "評価理由"
    }
  },
  "helpfulness": {
    "goal_clarity": {
      "score": 1,
      "rationale": "評価理由"
    },
    "actionability": {
      "score": 1,
      "rationale": "評価理由"
    },
    "audience_fit": {
      "score": 1,
      "rationale": "評価理由"
    },
    "troubleshooting_value": {
      "score": 1,
      "rationale": "評価理由"
    }
  },
  "reproducibility": {
    "environment_specificity": {
      "score": 1,
      "rationale": "評価理由"
    },
    "dependency_completeness": {
      "score": 1,
      "rationale": "評価理由"
    },
    "code_completeness": {
      "score": 1,
      "rationale": "評価理由"
    },
    "execution_sequence": {
      "score": 1,
      "rationale": "評価理由"
    },
    "verification_clarity": {
      "score": 1,
      "rationale": "評価理由"
    }
  },
  "citation_quality": {
    "source_authority": {
      "score": 1,
      "rationale": "評価理由"
    },
    "claim_source_alignment": {
      "score": 1,
      "rationale": "評価理由"
    },
    "citation_coverage": {
      "score": 1,
      "rationale": "評価理由"
    },
    "link_context": {
      "score": 1,
      "rationale": "評価理由"
    }
  },
  "readability_ja": {
    "structure_flow": {
      "score": 1,
      "rationale": "評価理由"
    },
    "sentence_clarity": {
      "score": 1,
      "rationale": "評価理由"
    },
    "terminology_explanation": {
      "score": 1,
      "rationale": "評価理由"
    },
    "information_density": {
      "score": 1,
      "rationale": "評価理由"
    }
  },
  "original_value": {
    "concrete_evidence": {
      "score": 1,
      "rationale": "評価理由"
    },
    "failure_analysis": {
      "score": 1,
      "rationale": "評価理由"
    },
    "comparison_insight": {
      "score": 1,
      "rationale": "評価理由"
    },
    "environment_specific_insight": {
      "score": 1,
      "rationale": "評価理由"
    }
  }
}

# 評価対象記事

<article>
{{ARTICLE}}
</article>

# 共通校正ルール

## 明示されていない引用を推測しない

記事中に公開URLが1件もなく、具体的な文献情報も存在しない場合、以下はすべて1点にしてください。

- source_authority
- claim_source_alignment
- citation_coverage
- link_context

技術的な説明が一般的なMLflowの機能と一致しているだけでは、公式ドキュメントを参照した証拠にはなりません。

記事に書かれていない情報源を「参照していると考えられる」「示唆されている」などと推測してはいけません。

## 新しいVersionを異常と判定しない

記事の検証環境として明記されているOS、Python、MLflowなどのVersionについて、あなたが知らないVersionであることだけを理由に減点してはいけません。

Versionの新旧や実在性を外部情報と照合できない場合は、記事内で表記が一貫しているかだけを評価してください。

次のような評価は禁止します。

- 知らないmacOS Versionなので異常
- 学習時点より新しいVersionなので信頼できない
- 自分の知識にないVersionなので誤り

## コードの存在を本文で確認する

記事にファイル名が書かれているだけでは、コードが掲載されていると判定してはいけません。

反対に、記事中に完全なPythonコードブロックが存在する場合は、コードを読まずに「内容が不明」と判定してはいけません。

code_completenessでは、実際のコードブロックについて以下を確認してください。

- import
- データ定義
- train/test分割
- モデル生成
- 学習
- 予測
- Metric計算
- MLflow記録

# Citation校正ルール v2.3

# 内容校正ルール v2.3

以下のルールは、記事に存在する証拠の見落としと、記事の目的を超えた要求による減点を防ぐためのものです。

## 3点以下を付ける前に本文とコードを再確認する

サブ項目へ3点以下を付ける場合、rationaleには不足している具体的な要素、または問題のある記事中の記述を1つ以上示してください。

記事に実際に存在する手順、Error文字列、確認Command、Run ID、表、注意書きを「ない」と判定してはいけません。「より詳細な説明が必要」「より多くの例が必要」だけを理由に3点以下へ下げてはいけません。

## 記事の目的を超えた網羅性を要求しない

初学者向けの単一Mac Tutorialでは、記事の目的を達成するために必要な範囲を評価してください。

- 複雑な運用例や別Frameworkへの拡張がないことだけでactionabilityを減点しない
- 関連する代表的Errorを複数扱っている記事へ、網羅的でないことだけを理由にtroubleshooting_valueを3点以下にしない
- Iris分類に不要なデータDownloadや前処理を要求しない
- `load_iris`はscikit-learn同梱Datasetであり、別Download手順がないことをcode_completenessの欠点にしない
- Tutorialの範囲で不要な例外処理まで要求しない

## Pythonコードの実行経路を確認する

API名の有無だけではなく、実行順序とscopeを読んでください。

- `try`が実処理全体を囲み、`except MlflowException`が再送出するなら、例外処理がないと判定しない
- `try`が`pass`だけを囲み、MLflow処理がscope外なら、例外処理は機能していないためapi_command_correctnessまたはcode_completenessの具体的な問題として扱う
- `with mlflow.start_run() as run:`内でParameter、Metric、Modelを記録し、`run.info.run_id`を使うコードはRun lifecycleが明示されている
- 記事中の完全なコードブロックに存在する処理を、本文説明が短いことだけで「欠落」と判定しない

## 実測値と未測定値を区別する

記事が「Iris分類結果ではない」と明示し、記事生成Runまたは評価RunのRun ID、条件、計測値を示している場合、その値をIris分類の創作結果と判定してはいけません。

実測値の信頼性は、次の本文内のprovenanceで評価します。

- Run IDまたは同等の検証記録
- 比較条件と変更変数
- 計測値
- 適用範囲の限定

これらがある実測値は記事固有の一次証拠です。外部URLがないことを理由にconcrete_evidenceを減点しません。外部検証が必要な一般的主張と、執筆者自身の観測値を区別してください。

## 制御比較を評価する

同一Prompt SHA、Model、Temperature、Seedなどを固定し、`max_tokens`だけを変えた複数Runがある場合、変更変数を限定した比較証拠として評価してください。

比較から「当該2 Runの観測範囲」と限定した判断が得られる場合、他の未制御要因も追加検証していないことだけを理由にcomparison_insightを2点へ固定しないでください。因果を全環境へ一般化している場合だけunsupported_claim_controlで問題にします。

## 環境固有の知見を評価する

Apple Silicon上のModel、実行時間、Token/秒、Peak Memory、Local実行上の挙動などが示され、適用範囲が当該Machineへ限定されている場合、environment_specific_insightとして評価してください。

このサブ項目は他環境とのbenchmark比較を必須としません。他環境へ一般化していない記事に対し、「汎用的でない」「他環境比較がない」という理由だけで減点してはいけません。環境固有であること自体が評価対象です。

## 自己計測値へ外部Citationを要求しない

記事生成時間、Token数、Throughput、Peak Memory、評価Runのscoreなど、執筆者自身がRun IDと条件を伴って示す計測値は一次証拠です。これらに外部出典がないことをcitation_coverageの欠点にしません。

Citationは、MLflow API、CLI、Framework仕様、Dataset仕様、運用上の一般的主張など、外部検証可能な技術的主張について評価してください。

## 公式Repositoryを一次情報として扱う

`github.com/ml-explore/mlx`と`github.com/ml-explore/mlx-lm`は、それぞれMLXとMLX-LMの公式Repositoryです。source_authorityでは一次情報として評価してください。

`docs.astral.sh`上のuv公式Documentation、`mlflow.org`上のMLflow公式Documentation、`scikit-learn.org`上のAPI Referenceも一次情報です。

## 執筆指示の混入を可読性で評価する

完成記事に「〜を記載する」「〜を案内する」「〜を対応付ける」「各項目を〜で書く」のような執筆メモが残っている場合、sentence_clarityとinformation_densityの具体的な欠点です。

反対に、読者へ直接説明する完成文になっている場合、単に文が簡潔であることを「説明不足」と同一視しないでください。

Citationの4サブ項目は、それぞれ異なる事実を評価します。1つの問題を4項目すべてへ重複適用してはいけません。

## source_authority

リンク先・文献そのものの権威性だけを評価してください。記事本文中でのリンク配置、主張との対応、引用範囲はこの項目へ含めません。

- 5点: 製品・Frameworkの公式ドキュメント、公式API Reference、標準仕様、原著論文などの一次情報
- 4点: 公式Repository、Maintainerによる資料、信頼できる準一次情報が中心
- 3点: 技術出版社、専門組織、著者と根拠が明確な二次情報
- 2点: 情報源はあるが、著者、根拠、運営主体が不明確
- 1点: 公開情報源が存在しない、または明らかに無関係・信頼不能

`mlflow.org`上のMLflow公式ドキュメントは5点相当です。「記事が本当に参照した証拠がない」という理由でsource_authorityを減点してはいけません。この項目は引用先の品質を評価するものであり、執筆者の閲覧履歴を推測する項目ではありません。

## claim_source_alignment

記事中の主張と参照先のテーマ・内容が対応しているかを評価してください。情報源の権威性そのものはここで再評価しません。

- 5点: 主要な主張の直後に、その主張を直接裏付ける参照先が示されている
- 4点: 節または段落単位で参照先との対応が明確
- 3点: 参考資料一覧にあるリンク名と参照先が記事の主要テーマに対応するが、個々の主張との対応は明示されていない
- 2点: 記事と同じ技術分野だが、どの主張を裏付けるか判断しにくい
- 1点: 情報源がない、または記事中の主張と無関係

参考資料一覧に`MLflow Tracking Quickstart`、`MLflow Tracking`、`MLflow Model Registry`などの記事内容と対応する資料がある場合、「対応関係が一切ない」という理由で1点にしてはいけません。ただし、個別の主張の直後に引用がない場合は満点にしないでください。

## citation_coverage

記事の主要な外部検証可能な主張が、どの程度情報源でカバーされているかを評価してください。

- 5点: ほぼすべての重要な技術的主張に必要な出典がある
- 4点: 主要な主張の大部分をカバーし、一部だけ不足
- 3点: 記事の中心テーマをカバーする複数の資料があるが、API、運用上の制約、細部の主張には不足がある
- 2点: 限定された一部の主張だけをカバー
- 1点: 出典がない、または重要な主張を実質的にカバーしていない

URLの本数だけで採点せず、リンク名・参照先の対象範囲と記事の主要セクションを比較してください。

## link_context

読者がリンクを開く目的を記事中の表現から理解できるかを評価してください。情報源の権威性や網羅性はここで再評価しません。

- 5点: 本文中で参照目的と確認できる内容が説明され、説明的なアンカーテキストがある
- 4点: 参考資料一覧に具体的で説明的なリンク名があり、参照目的を判断できる
- 3点: 製品名や文書名はあるが、参照目的の説明が弱い
- 2点: 裸のURL、`こちら`、`詳細`など内容を示さないリンク名が中心
- 1点: リンクがない、またはリンクの目的を判断できない

Markdownの`[MLflow Tracking Quickstart](URL)`は裸のURLではありません。具体的な文書名を持つ説明的リンクとして評価してください。

## Citation採点前の確認手順

1. 記事中の公開URLとMarkdownリンクを実際に列挙する。
2. 各URLのDomainとリンク名を確認する。
3. source_authority、claim_source_alignment、citation_coverage、link_contextを別々に採点する。
4. 同じ不足を複数項目へ機械的に重複適用していないか確認する。
5. rationaleに、実際に確認したリンク名または不足している対応関係を書く。

# 内容校正ルール v2.4

## api_command_correctnessで例外処理を要求しない

`api_command_correctness`はAPI、CLI、引数、実行順序の技術的な正しさだけを評価します。

記事中のAPIとCommandが正しい場合、次の不足だけを理由に5点から減点してはいけません。

- `try/except`がない
- 例外処理が詳細でない
- Errorごとの処理をPythonコードへ実装していない

例外処理やError切り分けの説明量は、必要な場合だけ`troubleshooting_value`で評価してください。同じ不足を`api_command_correctness`と`troubleshooting_value`へ重複適用してはいけません。

## troubleshooting_valueの具体的な採点Anchor

記事中に3種類以上の代表的Errorがあり、それぞれに次の情報が存在する場合、`troubleshooting_value`を3点以下にしてはいけません。

- 実際のError表示例または観測される現象
- 確認Command
- 原因候補
- 対処
- 再確認方法または期待結果

4種類以上について上記5要素があり、Port、Tracking URI、依存関係などを具体的に切り分けられる場合は4点以上を基準とします。網羅的でないことや、Pythonコード内に例外処理がないことだけで3点へ下げてはいけません。

## actionabilityとaudience_fitを分離する

Terminal、作業Directory、実行Command、順序、成功確認が具体的なら、背景説明の簡潔さだけを理由に`actionability`を3点以下へ下げてはいけません。

用語説明が1文ずつ簡潔であることは、それ自体では誤りではありません。初学者に必要な役割や関係が欠けている場合だけ、欠けている用語または関係を`audience_fit`のrationaleへ具体的に書いてください。
