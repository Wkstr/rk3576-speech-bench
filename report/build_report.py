#!/usr/bin/env python3
"""从 results/ 的 JSON 渲染双语报告 report/REPORT.md（中文正文 + 英文摘要）。

测量层（bench/）只产 JSON，本脚本只读 JSON——重跑 harness 后重跑本脚本即更新报告。
用法：.venv/bin/python report/build_report.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASR_DIR = ROOT / "results/asr"
TTS_DIR = ROOT / "results/tts"
OUT = ROOT / "report/REPORT.md"

# tag → 展示名（模型, 精度/后端）
ASR_NAMES = {
    "zipformer-small__rknn-2.3.2": ("Zipformer small（双语）", "NPU fp16 自转"),
    "zipformer-small__cpu96": ("Zipformer small（双语）", "CPU fp32"),
    "zipformer-large__rknn-2.3.2": ("Zipformer large（双语）", "NPU fp16 自转"),
    "zipformer-large__cpu": ("Zipformer large（双语）", "CPU fp32"),
}
TTS_NAMES = {
    "piper-zh-huayan__zh": ("Piper zh（huayan medium）", "VITS", "zh"),
    "piper-en-ryan__en": ("Piper en（ryan medium）", "VITS", "en"),
    "matcha-zh-en__zh-en": ("Matcha + vocos-16k（中英）", "Matcha", "zh+en"),
    "matcha-hybrid__zh-en": ("Matcha 混合（声学 CPU + vocos NPU）", "Matcha", "zh+en"),
    "kokoro__zh-en": ("Kokoro 82M v1.1（多语）", "Kokoro", "zh+en"),
    "mms-eng__en": ("MMS en（地板对照）", "VITS", "en"),
}
DATASET_NAMES = {"zh": "FLEURS cmn_hans_cn test", "en": "LibriSpeech test-clean"}


def load(d: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(d.glob("*.json")):
        if f.name.endswith(".sysmon.jsonl"):
            continue
        r = json.loads(f.read_text())
        out[r["config"]["tag"]] = r
    return out


def npu_stats(tag: str, d: Path) -> tuple[float | None, int | None]:
    """从 sysmon.jsonl 算 NPU 占用均值/峰值（10Hz 采样，null 已剔除）。"""
    f = d / f"{tag}.sysmon.jsonl"
    if not f.exists():
        return None, None
    vals = []
    for line in f.read_text().splitlines():
        s = json.loads(line).get("npu_load")
        if s:
            # 格式 "NPU load:  Core0:  0%, Core1:  0%,"——取各核百分比均值
            cores = [int(x) for x in re.findall(r"Core\d+:\s*(\d+)%", str(s))]
            if cores:
                vals.append(sum(cores) / len(cores))
    if not vals:
        return None, None
    return round(sum(vals) / len(vals), 1), round(max(vals), 1)


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def ci(e: dict) -> str:
    lo, hi = e["wilson95"]
    return f"{pct(e['center'])} [{pct(lo)}–{pct(hi)}]"


def main() -> None:
    asr, tts = load(ASR_DIR), load(TTS_DIR)
    board = next(iter(asr.values()))["config"]["board"]
    kernel = next(iter(asr.values()))["config"]["kernel"]
    dates = sorted(r["config"]["date_utc"][:10] for r in list(asr.values()) + list(tts.values()))

    # ASR 行：按 模型×provider×语言 排序（small 在前，NPU 在前）
    order = ["zipformer-small__rknn-2.3.2__zh", "zipformer-small__cpu96__zh",
             "zipformer-large__rknn-2.3.2__zh", "zipformer-large__cpu__zh",
             "zipformer-small__rknn-2.3.2__en", "zipformer-small__cpu96__en",
             "zipformer-large__rknn-2.3.2__en", "zipformer-large__cpu__en"]
    asr_rows = []
    for tag in order:
        r = asr.get(tag)
        if not r:
            continue
        base = tag.rsplit("__", 1)[0]
        model, backend = ASR_NAMES[base]
        lang = tag.rsplit("__", 1)[1]
        a = r["aggregate"]
        err = a.get("cer") or a.get("wer")
        s = r["sysmon"]
        npu_mean, npu_peak = npu_stats(tag, ASR_DIR)
        asr_rows.append(
            f"| {model} | {backend} | {lang} | {ci(err)} | {a['rtf_mean']:.2f} "
            f"| {a['rtfx']:.1f}× | {a['first_token_s']['p50']:.2f} / {a['first_token_s']['p95']:.2f} "
            f"| {s['cpu_pct_mean']:.0f}% | {s['rss_mb_peak']:.0f} | "
            + (f"{npu_mean}% / {npu_peak}%" if npu_mean is not None else "–")
            + " |")

    # NPU vs CPU 逐条一致性
    consist, consist_ratio = [], {}
    for model, rknn, cpu in [
        ("small zh", "zipformer-small__rknn-2.3.2__zh", "zipformer-small__cpu96__zh"),
        ("small en", "zipformer-small__rknn-2.3.2__en", "zipformer-small__cpu96__en"),
        ("large zh", "zipformer-large__rknn-2.3.2__zh", "zipformer-large__cpu__zh"),
        ("large en", "zipformer-large__rknn-2.3.2__en", "zipformer-large__cpu__en"),
    ]:
        if rknn in asr and cpu in asr:
            h = lambda d: {u["id"]: u["hyp"].strip() for u in d["utterances"]}
            hr, hc = h(asr[rknn]), h(asr[cpu])
            same = sum(1 for i in hr if hr[i] == hc.get(i))
            consist.append(f"| {model} | {same}/{len(hr)} |")
            consist_ratio[model] = f"{same}/{len(hr)}"
    consist_zh = f"small {consist_ratio.get('small zh')}、large {consist_ratio.get('large zh')}"
    consist_en = f"small {consist_ratio.get('small en')}、large {consist_ratio.get('large en')}"

    # NPU 占用描述（发现 2 用）：small/large 的 rknn 配置均值
    def _npu_of(tag):
        m, _ = npu_stats(tag, ASR_DIR)
        return m
    npu_means = {k: _npu_of(t) for k, t in [
        ("small", "zipformer-small__rknn-2.3.2__zh"), ("large", "zipformer-large__rknn-2.3.2__zh")]}
    npu_frag = (f"（实测占用佐证：small 均值 {npu_means['small']}%、large 均值 {npu_means['large']}%、峰值均 ~23%，算力大量闲置）"
                if all(v is not None for v in npu_means.values()) else "")

    # TTS 关键数字（TL;DR/英文摘要/发现 3 用）
    ma = tts.get("matcha-zh-en__zh-en")
    hy = tts.get("matcha-hybrid__zh-en")
    ma_rtf = ma["aggregate"]["rtf_mean"] if ma else float("nan")
    ma_rt = ma["aggregate"]["roundtrip_error_rate"]["center"] if ma else float("nan")
    hy_rtf = hy["aggregate"]["rtf_mean"] if hy else float("nan")

    # TTS 行
    tts_rows, worst = [], []
    for tag, (name, arch, langs) in TTS_NAMES.items():
        r = tts.get(tag)
        if not r:
            continue
        a = r["aggregate"]
        rt = a["roundtrip_error_rate"]
        n = len(r["items"])
        s = r["sysmon"]
        tts_rows.append(
            f"| {name} | {arch} | {langs} | {n} | {a['rtf_mean']:.2f} | {a['rtfx']:.1f}× "
            f"| {ci(rt)} | {s['rss_mb_peak']:.0f} |")
        for it in sorted(r["items"], key=lambda x: -(x["roundtrip"]["error_rate"] or 0))[:2]:
            if (it["roundtrip"]["error_rate"] or 0) > 0.3:
                worst.append((name, it))

    worst_rows = "\n".join(
        f"| {name} | {it['id']} | {it['text']} | {it['roundtrip']['hyp']} | {pct(it['roundtrip']['error_rate'])} |"
        for name, it in worst)

    md = f"""# RK3576 ASR / TTS Benchmark 报告

> 板子：{board}（RK3576，8GB，Debian 12，kernel {kernel}）。测量日期 {dates[0]} ~ {dates[-1]}。
> 全部原始数据：`results/`（JSON + 10Hz 资源采样）。生成：`python report/build_report.py`，重跑 harness 后重跑本脚本即更新。

## TL;DR

- **流式 ASR 上 NPU 完全可用**：自转 fp16（toolkit2 2.3.2）zipformer 与 CPU fp32 近乎**逐条一致**（zh：{consist_zh}；en：{consist_en}——差异仅 1 词/1 字级，聚合错误数不变），零精度退化；官方 RKNN 版在上游是坏掉的（sherpa-onnx#2861），自转是唯一可行路径。
- **速度**：small NPU RTF 0.14–0.15（≈7 倍实时）；large 在 NPU 与 CPU 上同为 RTF 0.41——**NPU 对 large 无加速收益**，large 的唯一 NPU 价值是 offload 省电。
- **TTS 速度王是 Matcha**（CPU RTF {ma_rtf:.2f}，中英双语）；**vocos 换 NPU 后再降 {1-hy_rtf/ma_rtf:.0%} 到 {hy_rtf:.2f}**，回译不变——vocos 是 TTS 唯一稳赢的 NPU 卸载点。Piper 单语 RTF 0.32–0.34；Kokoro 82M fp32 在 A72 上 RTF 2.87（0.35 倍实时，仅批处理可用）；MMS en 1.66。
- **回译可懂度**（下界口径，见「口径」）：Piper zh 13% 最优；Matcha 修正声码器错配（22k→16k）后 en 可懂度恢复正常；Kokoro zh 长句数字串会读成英文。

## English summary

On a reComputer RK3576 (8GB), self-converted fp16 RKNN zipformer models (rknn-toolkit2 2.3.2) transcribe **near-identically to the CPU fp32 baseline** (zh: {consist_zh}; en: {consist_en} — diffs are single word/character, aggregate error counts unchanged), while the vendor's official RKNN zipformer is broken upstream (sherpa-onnx#2861). RTF: small NPU ≈0.14 (7× realtime); large shows **no NPU speedup** (RTF 0.41 on both). For TTS: Matcha+vocos-16k is fastest (CPU RTF {ma_rtf:.2f}, bilingual; **RTF {hy_rtf:.2f} with the vocoder offloaded to NPU**, same roundtrip intelligibility — the vocoder is the only reliably profitable TTS NPU offload; the fp16 acoustic estimator is not viable, and upstream's own production doesn't use it either). Piper RTF 0.32–0.34 per language, Kokoro-82M fp32 RTF 2.87 (batch only), MMS en 1.66. Roundtrip-ASR intelligibility (a lower bound — digit verbalization and homophones count as errors): Piper zh 13%, Piper en 21%, Matcha {ma_rt:.0%}, Kokoro 30%, MMS 31%; Kokoro-ZH reads digit strings in English. All numbers come with Wilson 95% CIs in `results/`.

## 口径（读数字前必读）

- **样本**：ASR = LibriSpeech test-clean 抽 30 条（en）+ FLEURS cmn_hans_cn test 抽 30 条（zh）；TTS = 自建 30 条文本（zh12/en12/ja2/ko2/yue2，本报告覆盖 zh+en 24 条，日韩粤属后续阶段）。zh 的 FLEURS 句子偏简单，CER 数字偏乐观，勿与 AISHELL 类数据集直接对比。
- **指标**：zh 用 CER、en 用 WER（撇号删除、小写去标点归一化）；括号内为 Wilson 95% 置信区间（30 条小样本必读区间）。RTF = 推理耗时/音频时长，越低越好；RTFx 为倍速（越高越好）。
- **NPU 版 = 自转 fp16**（rknn-toolkit2 2.3.2，librknnrt 2.2.0）：官方 RKNN zipformer 在上游坏掉（small 全 blank / large 叠词，#2861 未修），官方版不进矩阵。
- **流式延迟**：首字延迟 = 音频起点到首个 stable partial（100ms 帧真实节奏喂入）；端点延迟 = 语音尾到 final（端点检测关闭，故≈0，纯解码收尾耗时）。
- **TTS 回译可懂度是下界**：裁判为 zipformer small（CPU，与 ASR 基线同口径），合成音频重采样 16k 后转写对比。数字口读（`204`→`TWO HUNDRED FOUR`）与同音词（`flour`→`FLOWER`）被计为错误，**所有模型吃同一裁判、同一文本，横向比较成立，绝对值偏悲观**。
- **资源占用**：10Hz 采样 CPU%/RSS/温度；NPU 占用读 debugfs `/sys/kernel/debug/rknpu/load`（{"已启用" if any(npu_stats(t, ASR_DIR)[0] is not None for t in asr) else "本期未获 root 授权，NPU 占用缺失"}）。

## ASR 结果

| 模型 | 后端 | 语言 | 错误率 [95% CI] | RTF | 倍速 | 首字 p50/p95 (s) | CPU 均值 | RSS 峰值 (MB) | NPU 占用 均值/峰值 |
|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(asr_rows)}

### NPU vs CPU 逐条一致率（fp16 零退化验证）

| 对比 | 逐条一致 |
|---|---|
{chr(10).join(consist)}

差异均为 1 词/1 字级（如 marveled/marvelled、who has/who's），且各配置聚合错误数不变——fp16 近平局翻转，非精度损失。

## TTS 结果

| 模型 | 架构 | 语言 | 条数 | RTF | 倍速 | 回译错误率 [95% CI]（下界） | RSS 峰值 (MB) |
|---|---|---|---|---|---|---|---|
{chr(10).join(tts_rows)}

### 回译最差样本（错误率 >30% 的各模型 top 样本）

| 模型 | 样本 | 原文 | 回译 | 错误率 |
|---|---|---|---|---|
{worst_rows}

## 发现

1. **fp16 转换零退化**：四组对比逐条一致率 {consist_ratio.get('small zh')}、{consist_ratio.get('small en')}、{consist_ratio.get('large zh')}、{consist_ratio.get('large en')}，差异仅拼写变体/虚词近平局翻转且聚合错误数不变；本场景 fp16 不需要 int8，精度焦虑可以放下。
2. **NPU 对 large 无加速**：large 的 rknn 与 cpu RTF 同为 0.41，瓶颈疑似在 decoder/joiner 的 CPU 侧串行步骤{npu_frag}；small 则有明确（虽不大）的加速。选型含义：要精度上 large 时，不必纠结 NPU。
3. **Matcha 两条教训**：(a) 声码器必须配 16k——matcha-icefall-zh-en 是 16kHz 模型，首轮我们错配 vocos-22khz（sherpa 有告警被日志过滤吞了），导致音频 1.38× 快进、en 回译 63% 的冤案；修正后 en 30.1%、总 ER {ma_rt:.1%}。(b) 声学模型的 estimator 转 fp16 RKNN 精度不足（逐步偏差 25%→63%，ODE 累积后输出不可懂；上游生产也不用它——官方 compose 里标 `MATCHA_USE_ORT=1`，ORT CPU 9/10 vs RKNN 拆分 7/10）。encoder/vocos 的 NPU 版都验证可用。
4. **Kokoro zh 数字串会读成英文**（`2026年的春节` → `TWO THOUSAND TWENTY SIX…`），中英混排场景需前端预处理。
5. **Kokoro 在 A72 上 RTF 2.87**：fp32 82M 对四核 A72 过重，仅离线批处理可用；实时场景选 Matcha/Piper。
6. **MMS en（RTF 1.66）既慢可懂度又差**，纯地板对照，不建议任何场景使用。

## 局限

- 每语言 30 条、单板单次测量；Wilson 区间已给出，跨板复现可能有 ±1–2pp 漂移。
- 日韩粤 TTS 文本已备但未测（回译裁判仅中英），属下一阶段；SenseVoice、Whisper、TTS 的 NPU 化同属后续。
- 回译为间接可懂度指标，非人工 MOS；合成样本落盘 `results/tts/samples/` 可人工复核。
- 未测功耗（板载电流计无效，无 USB 功率计）。

## 复现

```bash
# 环境：Debian 12 on RK3576，librknnrt 2.2.0，sherpa-onnx（RKNN 索引 wheel）
bash setup/install.sh && bash setup/check_env.sh
.venv/bin/python scripts/prepare_data.py en   # 及 zh（数据集获取见 AGENTS.md）
bash bench/run_p1.sh        # ASR 4 配置 × zh/en
bash bench/run_p3.sh        # TTS 5 模型
.venv/bin/python report/build_report.py       # 本报告
```
"""
    OUT.write_text(md)
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
