"""ASR 文本规范化：CER/WER 前 ref/hyp 双方同口径处理。

zh/ja/ko/yue → CER 口径：去空白与标点，CJK 逐字，拉丁小写。
en → WER 口径：小写、去标点、折叠空白。
"""
import re

_CJK = r"㐀-鿿豈-﫿一-鿿぀-ヿ가-힯"
_LATIN = r"a-z0-9"

_PUNCT = re.compile(r"[^\w\s" + _CJK + r"]", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text: str, lang: str) -> str:
    t = text.lower()
    # 撇号直接删除而非替换为空格（didn't → didnt），
    # 否则 ref 裂成两个词、不吐撇号的 hyp 被冤判一个错误
    t = t.replace("'", "").replace("’", "")
    t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    if lang == "en":
        return t
    # CER 语言：去掉全部空格，拉丁字母保持小写连续
    return t.replace(" ", "")


def error_rate(ref: str, hyp: str, lang: str) -> float:
    """对**未规范化**的 ref/hyp 计算该语言口径的错误率。"""
    import jiwer

    r, h = normalize(ref, lang), normalize(hyp, lang)
    if not r:
        return 0.0 if not h else 1.0
    if lang == "en":
        return jiwer.wer(r, h)
    return jiwer.cer(r, h)
