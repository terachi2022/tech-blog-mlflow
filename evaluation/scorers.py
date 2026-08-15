import ipaddress
import re
from urllib.parse import urlparse

from mlflow.entities import Feedback
from mlflow.genai import scorer


def _code_block_count(text: str) -> int:
    """Markdown fenced code blockの数を返す。"""
    return text.count("```") // 2


def _h2_count(text: str) -> int:
    """H2見出し数を返す。"""
    return len(
        re.findall(
            r"^##\s+.+$",
            text,
            flags=re.MULTILINE,
        )
    )


def _strip_fenced_code(text: str) -> str:
    """Markdown fenced codeを構造検査から除外する。"""
    return re.sub(
        r"```.*?```",
        "",
        text,
        flags=re.DOTALL,
    )


def _numbered_step_count(text: str) -> int:
    """番号付きH3とMarkdown番号Listの手順数を返す。"""
    prose = _strip_fenced_code(text)

    numbered_h3 = re.findall(
        r"^###\s+\d+[.)]\s+.+$",
        prose,
        flags=re.MULTILINE,
    )

    numbered_list = re.findall(
        r"^\s*\d+[.)]\s+\S.*$",
        prose,
        flags=re.MULTILINE,
    )

    return len(numbered_h3) + len(
        numbered_list
    )


def _public_external_urls(text: str) -> list[str]:
    """
    本文中の外部公開URLを取得する。

    localhost / loopback / private IP は除外する。
    """
    urls = re.findall(
        r"https?://[^\s)>\"']+",
        text,
    )

    result: list[str] = []

    for url in urls:
        # Markdownや句読点の末尾を簡易除去
        url = url.rstrip(".,;:!?。、）]}>")

        parsed = urlparse(url)
        host = parsed.hostname

        if not host:
            continue

        if host.lower() == "localhost":
            continue

        try:
            ip = ipaddress.ip_address(host)

            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
            ):
                continue

        except ValueError:
            # 単一Labelや内部向けSuffixは公開外部URLとして扱わない。
            normalized_host = host.lower().rstrip(".")
            if (
                "." not in normalized_host
                or normalized_host.endswith(
                    (".local", ".internal", ".localhost")
                )
            ):
                continue

        result.append(url)

    return list(dict.fromkeys(result))


def _has_explicit_version(text: str) -> bool:
    """
    主要技術について明示的なバージョン番号が
    記載されているかを判定する。

    例:
      MLflow 3.15.1
      Python 3.14.6
      MLX-LM 0.31.3
      Qwen3 8B はバージョン番号ではないため対象外
    """
    tech_names = (
        r"(?:"
        r"MLflow|"
        r"Python|"
        r"MLX(?:-LM)?|"
        r"macOS|"
        r"Ubuntu|"
        r"Rocky\s+Linux|"
        r"RHEL|"
        r"CUDA"
        r")"
    )

    version_number = r"v?\d+\.\d+(?:\.\d+)?"

    pattern = (
        rf"\b{tech_names}"
        rf"\s*"
        rf"(?:version|バージョン|ver\.?)?"
        rf"\s*[:=]?\s*"
        rf"{version_number}\b"
    )

    return bool(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
    )


def _has_prerequisites(text: str) -> bool:
    pattern = (
        r"(前提条件|前提環境|動作環境|"
        r"必要条件|環境要件|Prerequisites)"
    )

    return bool(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
    )


def _has_failure_cases(text: str) -> bool:
    pattern = (
        r"(エラー|失敗例|トラブル|注意点|"
        r"制約|制限事項|よくある問題|"
        r"うまくいかない)"
    )

    return bool(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
    )


@scorer
def has_h1(outputs: str) -> Feedback:
    matched = bool(
        re.search(
            r"^#\s+.+$",
            outputs,
            flags=re.MULTILINE,
        )
    )

    return Feedback(
        value=matched,
        rationale=(
            "H1タイトルがあります。"
            if matched
            else "H1タイトルがありません。"
        ),
    )


@scorer
def conclusion_near_top(outputs: str) -> Feedback:
    """
    「## 結論」が冒頭600文字以内にあるかを判定。
    """
    position = outputs.find("## 結論")

    matched = (
        position >= 0
        and position <= 600
    )

    return Feedback(
        value=matched,
        rationale=(
            f"結論の開始位置は{position}文字目です。"
            if position >= 0
            else "「## 結論」がありません。"
        ),
    )


@scorer
def code_block_count(outputs: str) -> int:
    """Markdownコードブロック数。"""
    return _code_block_count(outputs)


@scorer
def public_external_link_count(outputs: str) -> int:
    """
    重複を除いた公開外部URL数。

    localhost等は除外する。
    """
    return len(
        _public_external_urls(outputs)
    )


@scorer
def has_version_info(outputs: str) -> Feedback:
    """
    MLflow 3.15.1 等の明示的な
    バージョン記載を評価する。
    """
    matched = _has_explicit_version(outputs)

    return Feedback(
        value=matched,
        rationale=(
            "主要技術の明示的なバージョン記載があります。"
            if matched
            else
            "主要技術の明示的なバージョン記載がありません。"
        ),
    )


@scorer
def has_prerequisites(outputs: str) -> Feedback:
    """前提条件・前提環境が明記されているか。"""
    matched = _has_prerequisites(outputs)

    return Feedback(
        value=matched,
        rationale=(
            "前提条件または環境情報があります。"
            if matched
            else
            "前提条件・環境情報が明示されていません。"
        ),
    )


@scorer
def has_failure_cases(outputs: str) -> Feedback:
    """
    エラー例、失敗例、制約、注意点などが
    記述されているか。
    """
    matched = _has_failure_cases(outputs)

    return Feedback(
        value=matched,
        rationale=(
            "失敗例・注意点・制約の記載があります。"
            if matched
            else
            "失敗例・注意点・制約の記載がありません。"
        ),
    )


@scorer
def structure_score(outputs: str) -> Feedback:
    """
    技術記事の基本構造を0〜1で評価。

    H1                      0.2
    冒頭に結論              0.2
    H2が4個以上              0.2
    code blockが2個以上      0.2
    まとめ/結論/おわりに      0.2
    """
    score = 0.0
    reasons: list[str] = []

    # H1
    if re.search(
        r"^#\s+.+$",
        outputs,
        flags=re.MULTILINE,
    ):
        score += 0.2
        reasons.append("H1あり")

    # 冒頭の結論
    position = outputs.find("## 結論")

    if 0 <= position <= 600:
        score += 0.2
        reasons.append("結論が冒頭")

    # H2
    h2_count = _h2_count(outputs)

    if h2_count >= 4:
        score += 0.2
        reasons.append(
            f"H2={h2_count}"
        )

    # Code Block
    blocks = _code_block_count(outputs)

    if blocks >= 2:
        score += 0.2
        reasons.append(
            f"code block={blocks}"
        )

    # まとめ
    if re.search(
        r"^##\s+(まとめ|結論|おわりに)",
        outputs,
        flags=re.MULTILINE,
    ):
        score += 0.2
        reasons.append("まとめあり")

    return Feedback(
        value=round(score, 2),
        rationale=", ".join(reasons),
    )


@scorer
def reproducibility_proxy(outputs: str) -> Feedback:
    """
    記事を読者が再現しやすいかを
    コードベースで近似評価する。

    technical accuracyそのものではない。

    Code block 3個以上      0.25
    前提条件                 0.20
    バージョン               0.20
    手順3個以上              0.20
    エラー/注意/制約          0.15
    """
    score = 0.0
    reasons: list[str] = []

    blocks = _code_block_count(outputs)

    if blocks >= 3:
        score += 0.25
        reasons.append(
            f"コード例={blocks}"
        )

    if _has_prerequisites(outputs):
        score += 0.20
        reasons.append(
            "前提条件あり"
        )

    if _has_explicit_version(outputs):
        score += 0.20
        reasons.append(
            "バージョンあり"
        )

    numbered_steps = _numbered_step_count(
        outputs
    )

    if numbered_steps >= 3:
        score += 0.20
        reasons.append(
            f"手順={numbered_steps}"
        )

    if _has_failure_cases(outputs):
        score += 0.15
        reasons.append(
            "失敗・制約あり"
        )

    return Feedback(
        value=round(score, 2),
        rationale=(
            ", ".join(reasons)
            if reasons
            else
            "再現性を示す要素を検出できませんでした。"
        ),
    )


@scorer
def article_length_chars(outputs: str) -> int:
    """記事の文字数。"""
    return len(outputs)
