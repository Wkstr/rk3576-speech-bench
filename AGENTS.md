# RK3576 ASR/TTS Benchmark — 项目约定

在 reComputer RK3576 (8GB, Debian 12) 上评测可跑的 ASR/TTS 模型，产出 JSON 结果 + 双语报告 + 静态对比站点。
完整方案：`proposed/benchmark/2026-10-08-rk3576-asr-tts-benchmark.md`
阶段计划：`.planning/2026-10-08-rk3576-asr-tts-benchmark/task_plan.md`
当前交接：`.agents/handoffs/CURRENT.md`（新会话先读它）

## ⚠️ 下载约定（用户硬性要求）

**所有下载由用户手动完成。agent 不直接下载文件**，只提供 URL + 目标路径，等用户放好后再继续。

- 模型 tarball → 解压到 `models/<name>/<variant>/`（`tar xf <file> -C <dir> --strip-components=1`，目录结构是 `encoder.rknn` 等直接在 variant 目录下）
- 散文件 → 放到指定路径
- `models/_cache/` 里可能有中断的残留 tarball，同名覆盖或删除均可
- 唯一例外：`setup/install.sh` 里的 pip 安装和 librknnrt 下载已由脚本完成，不算手动范围

### 当前需要（解 P1 阻塞）

| 文件 | URL | 放到 |
|---|---|---|
| rknn-toolkit2 arm64 wheel（35MB） | https://github.com/airockchip/rknn-toolkit2/raw/master/rknn-toolkit2/packages/arm64/rknn_toolkit2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl | `~/.local/rknn/wheels/`（新建目录） |

用途：板上用 toolkit2 2.3.2 重转 zipformer RKNN 模型（官方版系 2.1.0 Beta 期转换，在 RK3576 上输出异常，见 sherpa-onnx#2861）。到位后 agent 建 `.venv-rknn` 安装并跑 `scripts/convert_rknn.py`。

已到位：librknnrt 2.3.2 / zipformer small CPU / SenseVoice RKNN / silero VAD。librknnrt 2.1.0 不需要（#2861 证实换版本无效）。

### 可提前下（P1 后半程 ~ P2 用）

| 文件 | URL | 放到 |
|---|---|---|
| zipformer large CPU | https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2 | 解压到 `models/zipformer-large-bilingual/cpu/` |
| SenseVoice CPU | https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2 | 解压到 `models/sensevoice/cpu/` |
| LibriSpeech test-clean | https://www.openslr.org/resources/12/test-clean.tar.gz | `data/_cache/test-clean.tar.gz`（不解压，脚本处理） |

注：zipformer large RKNN 版暂缓——上游 issue #2861 报告其有“叠词”输出，是否纳入矩阵待用户拍板。

## 环境

```bash
cd /home/jy/asrtts-benchmark
export LD_LIBRARY_PATH="$HOME/.local/rknn/lib"   # 跑 sherpa 必须
.venv/bin/python ...                              # python 一律用 venv 里的
```

- 自检 `bash setup/check_env.sh`；安装/修复 `bash setup/install.sh`（幂等）
- NPU 详细日志：`export RKNN_LOG_LEVEL=5`
- 磁盘紧张（/ 剩 ~20GB）：数据集只采样子集，大压缩包用完即删

## 评测口径（已定，别回退）

- 报告双语（中文正文 + 英文摘要）；样本 zh/en 各 30、日韩粤各 10；数字带 Wilson 95% 置信区间
- NPU int8 官方版 vs CPU fp32 基线，同 harness 切 provider
- 不测功耗；不做声音克隆；不做 Qwen3-ASR
