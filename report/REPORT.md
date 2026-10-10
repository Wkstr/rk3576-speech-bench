# RK3576 ASR / TTS Benchmark 报告

> 板子：Seeed Studio ReComputer RK3576 Dev Kit（RK3576，8GB，Debian 12，kernel 6.1.115-vendor-seeed-rk3576）。测量日期 2026-10-09 ~ 2026-10-10。
> 全部原始数据：`results/`（JSON + 10Hz 资源采样）。生成：`python report/build_report.py`，重跑 harness 后重跑本脚本即更新。

## TL;DR

- **流式 ASR 上 NPU 完全可用**：自转 fp16（toolkit2 2.3.2）zipformer 与 CPU fp32 近乎**逐条一致**（zh：small 30/30、large 29/30；en：small 27/30、large 30/30——差异仅 1 词/1 字级，聚合错误数不变），零精度退化；官方 RKNN 版在上游是坏掉的（sherpa-onnx#2861），自转是唯一可行路径。
- **速度**：small NPU RTF 0.14–0.15（≈7 倍实时）；large 在 NPU 与 CPU 上同为 RTF 0.41——**NPU 对 large 无加速收益**，large 的唯一 NPU 价值是 offload 省电。
- **TTS 速度王是 Matcha**（RTF 0.25，中英双语）；Piper 单语 RTF 0.32–0.34；Kokoro 82M fp32 在 A72 上 RTF 2.87（0.35 倍实时，仅批处理可用）；MMS en 1.66。
- **回译可懂度**（下界口径，见「口径」）：Piper zh 13% 最优；Matcha en 与 Kokoro zh 长句有真实可懂度问题。

## English summary

On a reComputer RK3576 (8GB), self-converted fp16 RKNN zipformer models (rknn-toolkit2 2.3.2) transcribe **near-identically to the CPU fp32 baseline** (zh: small 30/30、large 29/30; en: small 27/30、large 30/30 — diffs are single word/character, aggregate error counts unchanged), while the vendor's official RKNN zipformer is broken upstream (sherpa-onnx#2861). RTF: small NPU ≈0.14 (7× realtime); large shows **no NPU speedup** (RTF 0.41 on both). For TTS on CPU: Matcha+vocos is fastest (RTF 0.25, bilingual), Piper RTF 0.32–0.34 per language, Kokoro-82M fp32 RTF 2.87 (batch only), MMS en 1.66. Roundtrip-ASR intelligibility (a lower bound — digit verbalization and homophones count as errors): Piper zh 13%, Piper en 21%, Kokoro 30%, MMS 31%, Matcha 37%, with genuine intelligibility failures on Matcha-EN and Kokoro-ZH long sentences. All numbers come with Wilson 95% CIs in `results/`.

## 口径（读数字前必读）

- **样本**：ASR = LibriSpeech test-clean 抽 30 条（en）+ FLEURS cmn_hans_cn test 抽 30 条（zh）；TTS = 自建 30 条文本（zh12/en12/ja2/ko2/yue2，本报告覆盖 zh+en 24 条，日韩粤属后续阶段）。zh 的 FLEURS 句子偏简单，CER 数字偏乐观，勿与 AISHELL 类数据集直接对比。
- **指标**：zh 用 CER、en 用 WER（撇号删除、小写去标点归一化）；括号内为 Wilson 95% 置信区间（30 条小样本必读区间）。RTF = 推理耗时/音频时长，越低越好；RTFx 为倍速（越高越好）。
- **NPU 版 = 自转 fp16**（rknn-toolkit2 2.3.2，librknnrt 2.2.0）：官方 RKNN zipformer 在上游坏掉（small 全 blank / large 叠词，#2861 未修），官方版不进矩阵。
- **流式延迟**：首字延迟 = 音频起点到首个 stable partial（100ms 帧真实节奏喂入）；端点延迟 = 语音尾到 final（端点检测关闭，故≈0，纯解码收尾耗时）。
- **TTS 回译可懂度是下界**：裁判为 zipformer small（CPU，与 ASR 基线同口径），合成音频重采样 16k 后转写对比。数字口读（`204`→`TWO HUNDRED FOUR`）与同音词（`flour`→`FLOWER`）被计为错误，**所有模型吃同一裁判、同一文本，横向比较成立，绝对值偏悲观**。
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

### NPU vs CPU 逐条一致率（fp16 零退化验证）

| 对比 | 逐条一致 |
|---|---|
| small zh | 30/30 |
| small en | 27/30 |
| large zh | 29/30 |
| large en | 30/30 |

差异均为 1 词/1 字级（如 marveled/marvelled、who has/who's），且各配置聚合错误数不变——fp16 近平局翻转，非精度损失。

## TTS 结果

| 模型 | 架构 | 语言 | 条数 | RTF | 倍速 | 回译错误率 [95% CI]（下界） | RSS 峰值 (MB) |
|---|---|---|---|---|---|---|---|
| Piper zh（huayan medium） | VITS | zh | 12 | 0.34 | 3.0× | 13.4% [9.1%–17.6%] | 553 |
| Piper en（ryan medium） | VITS | en | 12 | 0.32 | 3.1× | 21.0% [14.7%–27.3%] | 512 |
| Matcha + vocos（中英） | Matcha | zh+en | 24 | 0.25 | 4.0× | 36.7% [32.0%–41.5%] | 413 |
| Kokoro 82M v1.1（多语） | Kokoro | zh+en | 24 | 2.87 | 0.3× | 29.8% [25.3%–34.4%] | 944 |
| MMS en（地板对照） | VITS | en | 12 | 1.66 | 0.6× | 31.2% [24.0%–38.4%] | 535 |

### 回译最差样本（错误率 >30% 的各模型 top 样本）

| 模型 | 样本 | 原文 | 回译 | 错误率 |
|---|---|---|---|---|
| Piper zh（huayan medium） | zh_08 | 2026年的春节是2月17日，星期二。 | 二千二十六年的春节是二月十七日星期二 | 47.1% |
| Piper zh（huayan medium） | zh_09 | 这个应用支持中文和English双语切换。 | 这个应用支持中文何用双语切换 | 40.0% |
| Piper en（ryan medium） | en_04 | The meeting has been moved to Room 204 on the second floor. | TWO HUNDRED FOUR ON THE SECOND FOUR | 75.0% |
| Piper en（ryan medium） | en_08 | The library is open from 9 AM to 6 PM on weekdays. | THE LIBRARY IS OPEN FROM NINE AM TO SIX P M ON WEEKDAY | 41.7% |
| Matcha + vocos（中英） | zh_08 | 2026年的春节是2月17日，星期二。 | TO SUSAN TWENTY SIX IN THE春节是托运三问天日星期二 | 152.9% |
| Matcha + vocos（中英） | en_03 | My flight leaves at seven thirty tomorrow morning. | 帅气才三万三他们 | 100.0% |
| Kokoro 82M v1.1（多语） | zh_08 | 2026年的春节是2月17日，星期二。 | TWO THOUSAND TWENTY SIX YANDA TWINSIA SHRITOYA SEVENTEEN YEAR SEEN TI R | 352.9% |
| Kokoro 82M v1.1（多语） | en_09 | She bought 2 tickets for the concert on Saturday, May 14th. | SHE BROUGHT TO TICKETS THE CONST ON SARADAY MAY FOURTEENTH | 54.5% |
| MMS en（地板对照） | en_08 | The library is open from 9 AM to 6 PM on weekdays. | THOUGH LIBRARIES OPEN FROM THEM TO MAN WE GAZE | 75.0% |
| MMS en（地板对照） | en_05 | This recipe calls for 300 grams of flour and a pinch of salt. | THIS RISET CALLS FOR FUN GRAHAMS OF FLOWER AND A PINCHER SALT | 46.2% |

## 发现

1. **fp16 转换零退化**：四组对比逐条一致率 30/30、27/30、29/30、30/30，差异仅拼写变体/虚词近平局翻转且聚合错误数不变；本场景 fp16 不需要 int8，精度焦虑可以放下。
2. **NPU 对 large 无加速**：large 的 rknn 与 cpu RTF 同为 0.41，瓶颈疑似在 decoder/joiner 的 CPU 侧串行步骤（实测占用佐证：small 均值 6.8%、large 均值 14.8%、峰值均 ~23%，算力大量闲置）；small 则有明确（虽不大）的加速。选型含义：要精度上 large 时，不必纠结 NPU。
3. **Matcha en 可懂度差**：英文句被裁判听成中文音节（如 `My flight leaves…` → `帅气才三万三他们`），该模型的英文前端/训练偏中文，英文场景勿选。
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
