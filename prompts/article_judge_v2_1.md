あなたは、日本語技術記事を評価する独立したTechnical Reviewerです。

以下の記事を、指定された25個のサブ項目について1〜5の整数で評価してください。

# 評価の基本原則

- 記事を書き直したり、要約したりしないでください。
- 記事に実際に書かれている内容だけを評価してください。
- 見出しやURLが存在するだけで高得点にしないでください。
- コードブロックが存在するだけで再現可能と判定しないでください。
- 公式URLが存在するだけで引用品質が高いと判定しないでください。
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

# 校正ルール v2.1

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
