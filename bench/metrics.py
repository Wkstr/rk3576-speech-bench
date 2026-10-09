"""指标汇总：错误率 + Wilson 95% 置信区间 + 延迟分位。

Wilson 口径（发布数字用）：把编辑距离的错误数近似为二项计数，
p = 总错误数 / 总参考单元数（zh 字、en 词），对 p 算 Wilson 95% 区间。
注意这是近似——编辑操作不独立，区间偏窄；仅用于公开报告的保守标注。
"""
import math


def wilson(errors: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """返回 (中心值, 下界, 上界)。total=0 时全 0。"""
    if total == 0:
        return 0.0, 0.0, 0.0
    p = errors / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center, max(0.0, center - half), min(1.0, center + half)


def percentiles(xs: list[float], ps=(50, 95)) -> dict[str, float]:
    if not xs:
        return {}
    xs = sorted(xs)
    out = {"mean": round(sum(xs) / len(xs), 4)}
    for p in ps:
        k = min(len(xs) - 1, max(0, math.ceil(p / 100 * len(xs)) - 1))
        out[f"p{p}"] = round(xs[k], 4)
    return out


def edit_counts(ref: str, hyp: str, lang: str) -> tuple[int, int]:
    """返回 (错误数, 参考单元数)。输入为**未规范化**文本。"""
    import jiwer

    from textnorm import normalize

    r, h = normalize(ref, lang), normalize(hyp, lang)
    if lang == "en":
        m = jiwer.process_words(r, h)
        total = len(r.split())
    else:
        m = jiwer.process_characters(r, h)
        total = len(r)
    return m.substitutions + m.deletions + m.insertions, total
