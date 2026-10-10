# RK3576 ASR / TTS Benchmark

[![在线站点](https://img.shields.io/badge/site-GitHub%20Pages-0d9488?logo=github)](https://wkstr.github.io/rk3576-speech-bench/)
[![platform](https://img.shields.io/badge/platform-RK3576%20NPU%20%2B%20Cortex--A72-blue)](https://www.seeedstudio.com/)
[![harness](https://img.shields.io/badge/harness-sherpa--onnx%20%2B%20librknnrt%202.2.0-orange)](https://github.com/k2-fsa/sherpa-onnx)

[English](README.md) | 中文

在瑞芯微 RK3576 开发板（reComputer，8GB）上对端侧语音识别（ASR）与语音合成（TTS）做的可复现实测：同一份 harness 切换 NPU（自转 fp16 RKNN）与 CPU（fp32）双口径，所有数字带 Wilson 95% 置信区间，原始 JSON 与合成样本全部入库。

**👉 在线对比站点（可排序表格 + 图表 + TTS 试听，中英文切换）：<https://wkstr.github.io/rk3576-speech-bench/>**

**📄 完整报告：[report/REPORT.md](report/REPORT.md)**（中文正文 + 英文摘要，口径/局限/复现命令全在其中）

## 结果速览（数据快照 2026-10-10，最新以站点为准）

### ASR（流式 zipformer，30 条/语言）

| 模型 | 后端 | zh CER | en WER | RTF (zh / en) |
|---|---|---|---|---|
| small（双语） | **NPU fp16 自转** | 10.1% | 19.3% | 0.14 / 0.15 |
| small（双语） | CPU fp32 | 10.1% | 19.3% | 0.16 / 0.17 |
| large（双语） | **NPU fp16 自转** | 7.6% | 11.4% | 0.41 / 0.41 |
| large（双语） | CPU fp32 | 7.8% | 11.4% | 0.41 / 0.44 |

要点：NPU 自转 fp16 与 CPU 输出近乎逐条一致（zh 30/30、29/30；en 27/30、30/30）——fp16 零退化；NPU 对 large 无加速（瓶颈在 CPU 侧 decode）。官方 RKNN zipformer 在上游是坏掉的（[sherpa-onnx#2861](https://github.com/k2-fsa/sherpa-onnx/issues/2861)），本仓库 NPU 数字来自自转模型（toolkit2 2.3.2）。

### TTS（CPU fp32，自建文本）

| 模型 | 语言 | RTF | 回译可懂度错误率（下界） |
|---|---|---|---|
| Matcha + vocos | zh+en | **0.25** | 36.7% |
| Piper (huayan) | zh | 0.34 | **13.4%** |
| Piper (ryan) | en | 0.32 | 21.0% |
| MMS | en | 1.66 | 31.2% |
| Kokoro 82M v1.1 | zh+en | 2.87 | 29.8% |

要点：Matcha 最快但英文可懂度差；Piper 均衡推荐；Kokoro 在四核 A72 上仅批处理可用；MMS 仅作地板对照。回译为下界口径（数字口读/同音词计为错误，横向比较公平），详见报告「口径」节。Matcha 混合版（声学 CPU + vocos NPU）RTF 可达 0.18。

## 目录结构

```
bench/    测量 harness（run_asr.py / run_tts.py / sysmon.py / run_p1.sh / run_p3.sh）
results/  原始结果 JSON + 10Hz 资源采样（呈现层唯一数据源）
report/   build_report.py → REPORT.md（双语）
site/     build_site.py → index.html（自包含单页）+ audio/（84 条 TTS 试听样本）
models.yaml / datasets.yaml  模型与数据集注册表
```

## 复现

```bash
bash setup/install.sh && bash setup/check_env.sh     # 环境（librknnrt 2.2.0 + sherpa-onnx）
.venv/bin/python scripts/prepare_data.py en          # 及 zh；数据集获取见 AGENTS.md
bash bench/run_p1.sh                                 # ASR 4 配置 × zh/en
bash bench/run_p3.sh                                 # TTS 5 模型
.venv/bin/python report/build_report.py              # 报告
.venv/bin/python site/build_site.py --copy-audio 99  # 站点
```

数据/模型来源：[LibriSpeech test-clean](https://www.openslr.org/12)、[FLEURS cmn_hans_cn](https://huggingface.co/datasets/google/fleurs)、[sherpa-onnx 模型仓库](https://github.com/k2-fsa/sherpa-onnx)、[rknn-toolkit2 2.3.2](https://github.com/airockchip/rknn-toolkit2)。
