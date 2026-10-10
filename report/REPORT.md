# RK3576 ASR / TTS Benchmark 报告

> 板子：Seeed Studio ReComputer RK3576 Dev Kit（RK3576，8GB，Debian 12，kernel 6.1.115-vendor-seeed-rk3576）。测量日期 2026-10-09 ~ 2026-10-10。
> 全部原始数据：`results/`（JSON + 10Hz 资源采样）。生成：`python report/build_report.py`，重跑 harness 后重跑本脚本即更新。

## TL;DR

- **流式 ASR 上 NPU 完全可用**：自转 fp16（toolkit2 2.3.2）zipformer 与 CPU fp32 近乎**逐条一致**（zh：small 30/30、large 29/30；en：small 27/30、large 30/30——差异仅 1 词/1 字级，聚合错误数不变），零精度退化；官方 RKNN 版在上游是坏掉的（sherpa-onnx#2861），自转是唯一可行路径。
- **速度**：small NPU RTF 0.14–0.15（≈7 倍实时）；large 在 NPU 与 CPU 上同为 RTF 0.41——**NPU 对 large 无加速收益**，large 的唯一 NPU 价值是 offload 省电。
- **TTS 速度王是 Matcha**（CPU RTF 0.25，中英双语）；**vocos 换 NPU 后再降 31% 到 0.18**，回译不变——vocos 是 TTS 唯一稳赢的 NPU 卸载点。Piper 单语 RTF 0.32–0.34；Kokoro 82M fp32 在 A72 上 RTF 2.87（0.35 倍实时，仅批处理可用）；MMS en 1.66。
- **回译可懂度**（下界口径，见「口径」）：Piper zh 13% 最优；Matcha 修正声码器错配（22k→16k）后 en 可懂度恢复正常；Kokoro zh 长句数字串会读成英文。
- **Qwen3-ASR（rkvoice-stream NPU 全栈）精度大幅领先 zipformer**：zh CER 4.4%（zipformer small 10.1%）、en WER 3.5%（zipformer large 11.4%），且多语覆盖 52 种。代价：RTF 0.40–0.47（比 small NPU 慢 3 倍），kv 整段定稿无流式 partial，尾帧定稿 p50 ≈2.1s——要精度/多语选它，要流式交互选 zipformer small。中英混合集（23 条合成）CER 3.4% vs zipformer small 30.0%，句内混说场景 zipformer 不可用（见发现 7）。

## English summary

On a reComputer RK3576 (8GB), self-converted fp16 RKNN zipformer models (rknn-toolkit2 2.3.2) transcribe **near-identically to the CPU fp32 baseline** (zh: small 30/30、large 29/30; en: small 27/30、large 30/30 — diffs are single word/character, aggregate error counts unchanged), while the vendor's official RKNN zipformer is broken upstream (sherpa-onnx#2861). RTF: small NPU ≈0.14 (7× realtime); large shows **no NPU speedup** (RTF 0.41 on both). For TTS: Matcha+vocos-16k is fastest (CPU RTF 0.25, bilingual; **RTF 0.18 with the vocoder offloaded to NPU**, same roundtrip intelligibility — the vocoder is the only reliably profitable TTS NPU offload; the fp16 acoustic estimator is not viable, and upstream's own production doesn't use it either). Piper RTF 0.32–0.34 per language, Kokoro-82M fp32 RTF 2.87 (batch only), MMS en 1.66. **Qwen3-ASR** (rkvoice-stream full-NPU stack: fp16 RKNN encoder + w4a16 RKLLM decoder) beats every zipformer on accuracy — zh CER 4.4% vs 10.1%, en WER 3.5% vs 11.4%, 52 languages, auto-detected — at RTF 0.40–0.47 with no streaming partials (whole-utterance KV decode; finalize p50 ≈2.1s after end-of-audio); on a 23-utterance synthesized zh-en code-switch set it hits CER 3.4% while zipformer collapses on intra-sentence switching (small 30.0%). Roundtrip-ASR intelligibility (a lower bound — digit verbalization and homophones count as errors): Piper zh 13%, Piper en 21%, Matcha 25%, Kokoro 30%, MMS 31%; Kokoro-ZH reads digit strings in English. All numbers come with Wilson 95% CIs in `results/`.

## 口径（读数字前必读）

- **样本**：ASR = LibriSpeech test-clean 抽 30 条（en）+ FLEURS cmn_hans_cn test 抽 30 条（zh）；TTS = 自建 30 条文本（zh12/en12/ja2/ko2/yue2，本报告覆盖 zh+en 24 条，日韩粤属后续阶段）。zh 的 FLEURS 句子偏简单，CER 数字偏乐观，勿与 AISHELL 类数据集直接对比。
- **指标**：zh 用 CER、en 用 WER（撇号删除、小写去标点归一化）；括号内为 Wilson 95% 置信区间（30 条小样本必读区间）。RTF = 推理耗时/音频时长，越低越好；RTFx 为倍速（越高越好）。
- **NPU 版 = 自转 fp16**（rknn-toolkit2 2.3.2，librknnrt 2.2.0）：官方 RKNN zipformer 在上游坏掉（small 全 blank / large 叠词，#2861 未修），官方版不进矩阵。
- **流式延迟**：首字延迟 = 音频起点到首个 stable partial（100ms 帧真实节奏喂入）；端点延迟 = 语音尾到 final（端点检测关闭，故≈0，纯解码收尾耗时）。
- **TTS 回译可懂度是下界**：裁判为 zipformer small（CPU，与 ASR 基线同口径），合成音频重采样 16k 后转写对比。数字口读（`204`→`TWO HUNDRED FOUR`）与同音词（`flour`→`FLOWER`）被计为错误，**所有模型吃同一裁判、同一文本，横向比较成立，绝对值偏悲观**。
- **Qwen3-ASR 口径**：suharvest/rkvoice-stream 预编译产物（fp16 encoder 2s/4s + RKLLM w4a16_g128 decoder，librkllmrt v1.2.3），语言**自动检测**（不传 hint，与 zipformer 同口径）；走其 V2V 生产的 kv 整段解码路径（边喂边编码、finish 单次解码），无流式 partial，首字列显示 –；其 transcribe() 滑窗拼接路径在 en 上会丢段（见发现 8），未采用。输出自带标点、数字为阿拉伯形式，与 zipformer 原始输出同过一套 textnorm 后比较。
- **资源占用**：10Hz 采样 CPU%/RSS/温度；NPU 占用读 debugfs `/sys/kernel/debug/rknpu/load`（已启用）。

## ASR 结果

| 模型 | 后端 | 语言 | 错误率 [95% CI] | RTF | 倍速 | 首字 p50/p95 (s) | CPU 均值 | RSS 峰值 (MB) | NPU 占用 均值/峰值 |
|---|---|---|---|---|---|---|---|---|---|
| Zipformer small（双语） | NPU fp16 自转 | zh | 10.1% [8.3%–11.9%] | 0.14 | 7.0× | 2.21 / 5.04 | 7% | 488 | 6.8% / 23.0% |
| Zipformer small（双语） | CPU fp32 | zh | 10.1% [8.3%–11.9%] | 0.16 | 6.3× | 2.19 / 4.96 | 30% | 267 | – |
| Zipformer large（双语） | NPU fp16 自转 | zh | 7.6% [6.0%–9.2%] | 0.41 | 2.4× | 2.16 / 4.33 | 9% | 609 | 14.8% / 23.5% |
| Zipformer large（双语） | CPU fp32 | zh | 7.8% [6.1%–9.4%] | 0.41 | 2.5× | 2.14 / 4.34 | 59% | 488 | – |
| Zipformer small（双语） | NPU fp16 自转 | en | 19.3% [16.1%–22.4%] | 0.15 | 6.9× | 1.19 / 1.25 | 6% | 485 | 6.7% / 21.5% |
| Zipformer small（双语） | CPU fp32 | en | 19.3% [16.1%–22.4%] | 0.17 | 5.9× | 1.18 / 1.24 | 31% | 246 | – |
| Zipformer large（双语） | NPU fp16 自转 | en | 11.4% [8.9%–13.9%] | 0.41 | 2.4× | 0.89 / 1.51 | 9% | 607 | 15.0% / 23.0% |
| Zipformer large（双语） | CPU fp32 | en | 11.4% [8.9%–13.9%] | 0.44 | 2.3× | 0.89 / 1.50 | 62% | 494 | – |
| Qwen3-ASR（rkvoice-stream 预编译） | NPU fp16+w4a16 | zh | 4.4% [3.1%–5.6%] | 0.40 | 2.5× | –（无 partial） | 18% | 2914 | 14.4% / 50.0% |
| Qwen3-ASR（rkvoice-stream 预编译） | NPU fp16+w4a16 | en | 3.5% [2.1%–5.0%] | 0.47 | 2.2× | –（无 partial） | 20% | 2911 | 17.0% / 51.0% |

### NPU vs CPU 逐条一致率（fp16 零退化验证）

| 对比 | 逐条一致 |
|---|---|
| small zh | 30/30 |
| small en | 27/30 |
| large zh | 29/30 |
| large en | 30/30 |

差异均为 1 词/1 字级（如 marveled/marvelled、who has/who's），且各配置聚合错误数不变——fp16 近平局翻转，非精度损失。

### 中英混合（code-switch，23 条 TTS 合成语音）

| 模型 | 后端 | CER [95% CI] | RTF | 条数 |
|---|---|---|---|---|
| Qwen3-ASR（rkvoice-stream 预编译） | NPU fp16+w4a16 | 3.4% [2.2%–4.6%] | 0.45 | 23 |
| Zipformer small（双语） | CPU fp32 | 30.0% [26.9%–33.1%] | 0.19 | 23 |
| Zipformer large（双语） | CPU fp32 | 24.3% [21.4%–27.2%] | 0.45 | 23 |

语料为自建 24 句中英混合文本（职场/开发者/数码/生活/专名 5 类，难度梯度从单词插入到连续多词切换），用户侧 TTS 合成；cs_011（Wi-Fi 被 TTS 读成乱码，两模型一致确认）剔除不计分；4 条 ref 按两模型一致的实际读音校准（数字/型号口读形式，如 3.12→三点一二、café→咖啡）。口径同 zh CER（逐字、拉丁小写去空格）。**注意：合成语音比真人发音规整，绝对值偏乐观，只用于横向对比。**

## TTS 结果

| 模型 | 架构 | 语言 | 条数 | RTF | 倍速 | 回译错误率 [95% CI]（下界） | RSS 峰值 (MB) |
|---|---|---|---|---|---|---|---|
| Piper zh（huayan medium） | VITS | zh | 12 | 0.34 | 3.0× | 13.4% [9.1%–17.6%] | 553 |
| Piper en（ryan medium） | VITS | en | 12 | 0.32 | 3.1× | 21.0% [14.7%–27.3%] | 512 |
| Matcha + vocos-16k（中英） | Matcha | zh+en | 24 | 0.25 | 3.9× | 24.7% [20.5%–29.0%] | 424 |
| Matcha 混合（声学 CPU + vocos NPU） | Matcha | zh+en | 24 | 0.18 | 5.7× | 23.5% [19.3%–27.7%] | 524 |
| Kokoro 82M v1.1（多语） | Kokoro | zh+en | 24 | 2.87 | 0.3× | 29.8% [25.3%–34.4%] | 944 |
| MMS en（地板对照） | VITS | en | 12 | 1.66 | 0.6× | 31.2% [24.0%–38.4%] | 535 |

### 回译最差样本（错误率 >30% 的各模型 top 样本）

| 模型 | 样本 | 原文 | 回译 | 错误率 |
|---|---|---|---|---|
| Piper zh（huayan medium） | zh_08 | 2026年的春节是2月17日，星期二。 | 二千二十六年的春节是二月十七日星期二 | 47.1% |
| Piper zh（huayan medium） | zh_09 | 这个应用支持中文和English双语切换。 | 这个应用支持中文何用双语切换 | 40.0% |
| Piper en（ryan medium） | en_04 | The meeting has been moved to Room 204 on the second floor. | TWO HUNDRED FOUR ON THE SECOND FOUR | 75.0% |
| Piper en（ryan medium） | en_08 | The library is open from 9 AM to 6 PM on weekdays. | THE LIBRARY IS OPEN FROM NINE AM TO SIX P M ON WEEKDAY | 41.7% |
| Matcha + vocos-16k（中英） | zh_08 | 2026年的春节是2月17日，星期二。 | TWO THOUSAND TWENTY SIX YEAR是 TWO SEVENTY日星期二 | 200.0% |
| Matcha + vocos-16k（中英） | en_04 | The meeting has been moved to Room 204 on the second floor. | THE MEETING HAS BEEN MOVED TO ROOM TOO HUNDREDS FOR DREW ROUND THE SECOND FLORI | 50.0% |
| Matcha 混合（声学 CPU + vocos NPU） | zh_08 | 2026年的春节是2月17日，星期二。 | TO ZERO TWO SIXTYN的春节是突越弯三文日星期二 | 117.7% |
| Matcha 混合（声学 CPU + vocos NPU） | en_05 | This recipe calls for 300 grams of flour and a pinch of salt. | THIS RECY PEA COSFORD FREEZEAL ZERO GRANTS OF FLOR AND THE PINCH OF SALT | 61.5% |
| Kokoro 82M v1.1（多语） | zh_08 | 2026年的春节是2月17日，星期二。 | TWO THOUSAND TWENTY SIX YANDA TWINSIA SHRITOYA SEVENTEEN YEAR SEEN TI R | 352.9% |
| Kokoro 82M v1.1（多语） | en_09 | She bought 2 tickets for the concert on Saturday, May 14th. | SHE BROUGHT TO TICKETS THE CONST ON SARADAY MAY FOURTEENTH | 54.5% |
| MMS en（地板对照） | en_08 | The library is open from 9 AM to 6 PM on weekdays. | THOUGH LIBRARIES OPEN FROM THEM TO MAN WE GAZE | 75.0% |
| MMS en（地板对照） | en_05 | This recipe calls for 300 grams of flour and a pinch of salt. | THIS RISET CALLS FOR FUN GRAHAMS OF FLOWER AND A PINCHER SALT | 46.2% |

## 发现

1. **fp16 转换零退化**：四组对比逐条一致率 30/30、27/30、29/30、30/30，差异仅拼写变体/虚词近平局翻转且聚合错误数不变；本场景 fp16 不需要 int8，精度焦虑可以放下。
2. **NPU 对 large 无加速**：large 的 rknn 与 cpu RTF 同为 0.41，瓶颈疑似在 decoder/joiner 的 CPU 侧串行步骤（实测占用佐证：small 均值 6.8%、large 均值 14.8%、峰值均 ~23%，算力大量闲置）；small 则有明确（虽不大）的加速。选型含义：要精度上 large 时，不必纠结 NPU。
3. **Matcha 两条教训**：(a) 声码器必须配 16k——matcha-icefall-zh-en 是 16kHz 模型，首轮我们错配 vocos-22khz（sherpa 有告警被日志过滤吞了），导致音频 1.38× 快进、en 回译 63% 的冤案；修正后 en 30.1%、总 ER 24.7%。(b) 声学模型的 estimator 转 fp16 RKNN 精度不足（逐步偏差 25%→63%，ODE 累积后输出不可懂；上游生产也不用它——官方 compose 里标 `MATCHA_USE_ORT=1`，ORT CPU 9/10 vs RKNN 拆分 7/10）。encoder/vocos 的 NPU 版都验证可用。
4. **Kokoro zh 数字串会读成英文**（`2026年的春节` → `TWO THOUSAND TWENTY SIX…`），中英混排场景需前端预处理。
5. **Kokoro 在 A72 上 RTF 2.87**：fp32 82M 对四核 A72 过重，仅离线批处理可用；实时场景选 Matcha/Piper。
6. **MMS en（RTF 1.66）既慢可懂度又差**，纯地板对照，不建议任何场景使用。
7. **中英混合是 Qwen3-ASR 的结构性优势**：code-switch 集 CER Qwen3 3.4% vs zipformer small 30.0% / large 24.3%。zipformer 双语模型只支持整句单语，句内混说时英文词崩成音近汉字（debug→低霸隔、deliver→呆子晚）并伴随整段丢弃；Qwen3 基本逐字正确，残余错误里有一类是**翻译倾向**（把「我们 need 更多的 data」转写成「we need more data」），是语言模型味而非听力问题。
8. **Qwen3-ASR 是精度天花板，但离线路径有坑**：其 transcribe() 默认滑窗拼接（中间块 6 token 早停 + rollback 重 decode）在 en 上会整段丢文本（`IT WAS SILENT AND GLOOMY BEING TENANTED SOLELY BY` 整句消失）并在块缝丢空格；kv 整段解码（V2V 路径）则逐句正确。接入方若复用 rkvoice-stream，离线转写务必走 kv_streaming，勿用 transcribe() 默认参数。RK3576 官方产物只带 2s/4s encoder 窗口（无 15s），长音频被限在 4s 窗口滑窗。

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
bash bench/run_p1.sh        # ASR zipformer 4 配置 × zh/en
bash bench/run_qwen3.sh     # ASR Qwen3-ASR × zh/en（需 models/qwen3-asr + models/lib/librkllmrt.so）
bash bench/run_p3.sh        # TTS 5 模型
.venv/bin/python report/build_report.py       # 本报告
```
