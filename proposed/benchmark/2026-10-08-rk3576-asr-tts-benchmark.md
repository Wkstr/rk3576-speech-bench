# RK3576 ASR/TTS Benchmark 方案

- 日期：2026-10-08
- 状态：proposed（已于本会话经 grilling + think 两轮确认，用户批准）
- 硬件：reComputer RK3576 devkit（8GB RAM，Debian 12 / Armbian，内核 6.1.115-vendor-seeed-rk3576，可用磁盘 20GB）

## Building

本 repo 一套 Python 可复现 benchmark pipeline：配置驱动的模型/数据集注册表 + 统一测量 harness（准确率/延迟/资源三线并行）+ JSON 结果层 + 双语报告与零构建静态站点。公开内容向，任何人一块 reComputer RK3576 即可复现全部数字。

## Not building

不测功耗、不做声音克隆、不做 Qwen3-ASR、不做多档量化实验、不做 MCD 等重型音质指标、不引入第二模型运行时。

## 评测范围（grilling 锁定）

**场景**：ASR 流式优先，TTS 首包延迟敏感；核心中英，广度日/韩/粤。

**ASR 清单**（+ CPU fp32 基线）：

| 模型 | 来源 | 流式 | 语言 |
|---|---|---|---|
| zipformer small bilingual | sherpa RKNN 现成 58M | ✅ | 中英 |
| zipformer bilingual large (02-20) | sherpa RKNN 现成 146M | ✅ | 中英 |
| SenseVoice 20s | sherpa RKNN 现成 | ❌（VAD 分段） | 中英日韩粤 |
| Whisper base | rknn_model_zoo 自转 | ❌ | 多语言对照 |

**TTS 清单**：Piper VITS（多语言 CPU 基线）→ Matcha+Vocos（RKNN 主力）→ Kokoro-82M（CPU 先行，NPU 进阶）→ MMS-TTS（多语言地板）。不做声音克隆。

**数据**：zh/en 各随机 30 条，日/韩/粤各 10 条（FLEURS），TTS 固定 30 条文本；不做真实场景自录。发布数字带 Wilson 95% 置信区间。

**指标**：CER(中日韩粤)/WER(英)、RTF/RTFx、ASR 首字延迟+端点延迟、TTS 首包延迟、峰值内存/CPU/NPU/温度、ASR 回译可懂度、ASR→TTS V2V 串联延迟。不测功耗。

**产出**：pipeline + 双语报告（中文正文+英文摘要）+ 静态单页零构建对比站（可排序表格 + Chart.js + 音频试听，GitHub Pages）。

## Approach

```
models.yaml ──┐                ┌─→ results/asr/*.json ─┐
datasets.yaml ┼→ bench/ ────── ┼─→ results/tts/*.json ─┼→ report/ (md 双语)
  (注册表)     │  run_asr.py    ├─→ results/tts/samples/ │   site/ (静态单页)
  下载脚本 ────┘  run_tts.py    │    *.wav (试听)        │   GitHub Pages
                  sysmon.py ───┴─→ 资源时间序列 ─────────┘
                  (10Hz: CPU/RSS/NPU/温度)
```

测量层只产 JSON/wav，呈现层只读 JSON，重跑自动更新站点。

## Key decisions

1. Python + sherpa-onnx API 做 harness，拒绝手动 CLI 抄数（不可复现），拒绝 MLPerf 式重框架（30 条样本用不上）。
2. 同一 harness 切 provider（rknn ↔ cpu fp32），天然双口径；量化损失 = 两边 CER 差。
3. 流式延迟按真实时间节奏喂 100ms 帧：音频起点→首个 stable partial = 首字延迟；语音尾→final = 端点延迟（测法借鉴 rkvoice-stream 的 TTFB/V2V 口径，实测验证其宣称的 120ms/700ms）。
4. 30 条样本的公开数字必须带 Wilson 95% 置信区间；同时报 RTFx（借自 HF Open ASR Leaderboard）。
5. 高风险项（Whisper 自转、Matcha/Kokoro RKNN 化）排在 P4，失败只损失对照项，主线数据早已落盘。
6. librknnrt 锁 2.2.0（sherpa-onnx 与 2.3.x/2.4.x 有已知崩溃）；HF 下载默认走 `HF_ENDPOINT=https://hf-mirror.com`，wget -c 断点续传，产物本地缓存。

## 阶段划分（各自独立可交付）

| 阶段 | 内容 | 出口 |
|---|---|---|
| P1 地基 | setup 脚本（librknnrt 2.2.0 + sherpa-onnx + 环境自检）、下载/采样脚本、ASR harness + sysmon；zipformer small/large × rknn/cpu × zh/en | 第一批真实 JSON |
| P2 多语言 | SenseVoice（VAD 分段）+ FLEURS 日/韩/粤子集 | 多语言数据 |
| P3 TTS | TTS harness（TTFB/RTF/样本落盘/回译可懂度）× Piper、Matcha、Kokoro(CPU)、MMS-TTS | TTS 数据齐 |
| P4 补全 | Whisper 板上自转（toolkit2 aarch64；失败→社区预转换→仅 CPU 基线并标注）、Matcha/Kokoro NPU 尝试、V2V 串联 | 矩阵完整 |
| P5 呈现 | 报告生成（模板+JSON 渲染 md 双语）、静态单页、GitHub Pages | 发布 |

## 最脆弱假设

sherpa-onnx + librknnrt 2.2.0 能在 seeed 定制内核上正常初始化 NPU。不成立则 NPU 主线垮 → P1 第一步自检（驱动版本、NPU 冒烟、debugfs load 读数）；失败按 seeed wiki 升驱动 0.9.8 或换匹配 librknnrt；最坏降级纯 CPU benchmark 并显著声明（CPU 基线本就是矩阵一半，价值打折不归零）。

## Unknowns（显式挂起）

1. sherpa-onnx 是否有 cp311 aarch64 wheel —— P1 第一小时验证，无则 Miniforge 起 py3.10（官方示例即 cp310）。
2. Matcha/Kokoro 的 RKNN 模型从 rkvoice-stream 提取还是自转 —— P4 开工时定。
