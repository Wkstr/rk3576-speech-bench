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

### P1 状态（2026-10-09）

自转 zipformer RKNN（toolkit2 2.3.2，fp16）板上验证通过：small（chunk96）5/5 冒烟正确、RTF 0.15–0.19，en 30 条 WER 与 CPU 逐条一致（fp16 零退化）；large（chunk32）冒烟输出连贯。正式 NPU 版 = `models/<name>/rknn-2.3.2/`；small 官方 `rknn/` 版（#2861，输出全 blank）仅留作报告对照，large 官方版从未下载。small 的 CPU 对照须用 `cpu/96/` 子目录同 chunk 口径。决策与验证细节：`.agents/notes/implemented/process/2026-10-09-zipformer-rknn-self-conversion.md`。

librknnrt 2.1.0 不需要（#2861 证实换版本无效）。arm64 版 toolkit2 wheel 在 `~/.local/rknn/wheels/`（板上转换方案已弃用，留作后备）。

### 当前需要

| 文件 | URL / 来源 | 放到 |
|---|---|---|
| FLEURS 中文 test 音频（525MB） | https://hf-mirror.com/datasets/google/fleurs/resolve/main/data/cmn_hans_cn/audio/test.tar.gz | `data/_cache/fleurs-zh/test.tar.gz`（不解压） |
| FLEURS 中文 test 转写（0.5MB） | https://hf-mirror.com/datasets/google/fleurs/resolve/main/data/cmn_hans_cn/test.tsv | `data/_cache/fleurs-zh/test.tsv` |
| Piper zh（64MB） | https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-zh_CN-huayan-medium.tar.bz2 | 解压到 `models/piper-zh-huayan/cpu/` |
| Piper en（64MB） | https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-ryan-medium.tar.bz2 | 解压到 `models/piper-en-ryan/cpu/` |
| Matcha 中英（75MB） | https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-en.tar.bz2 | 解压到 `models/matcha-zh-en/cpu/` |
| vocos 声码器（51MB，Matcha 用） | https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx | `models/vocos/vocos-22khz-univ.onnx`（散文件） |
| Kokoro 多语（347MB） | https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_1.tar.bz2 | 解压到 `models/kokoro/cpu/` |
| MMS en（102MB） | https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mms-eng.tar.bz2 | 解压到 `models/mms-eng/cpu/` |

zh 源已从 AISHELL-1 换成 FLEURS cmn_hans_cn test（用户拍板 2026-10-09：AISHELL 太大；FLEURS 与 P2 日韩粤同源）。
已到位：zipformer small CPU+自转 RKNN / large CPU+自转 RKNN（`rknn-2.3.2/`）/ LibriSpeech en 30 条已采样。SenseVoice 全线移出当前范围（用户拍板，后续需要再补）。MMS 官方只打包了 eng，无 zh。

### 可选提权（改善资源曲线）

`sudo chmod o+r /sys/kernel/debug/rknpu/load` —— 不解则 sysmon 的 npu_load 全为 null（其余 CPU/RSS/温度不受影响）。

注：zipformer large 自转 RKNN 已到位（用户 2026-10-09 转好，放 `rknn-2.3.2/`）；官方版叠词问题（#2861）与我们无关，但 large 冒烟出现过一次双词重复，报告阶段在语料上定量再下结论。

## 环境

```bash
cd /home/jy/asrtts-benchmark
export LD_LIBRARY_PATH="$HOME/.local/rknn/lib"   # 跑 sherpa 必须
.venv/bin/python ...                              # python 一律用 venv 里的
```

- 自检 `bash setup/check_env.sh`；安装/修复 `bash setup/install.sh`（幂等）
- NPU 详细日志：`export RKNN_LOG_LEVEL=5`
- 磁盘紧张（/ 剩 ~20GB）：数据集只采样子集，大压缩包用完即删

## 提交约定

- 提交信息用**英文** + Conventional Commits 简洁风（caveman-commit：只说意图，why 优先于 what，无 AI 署名）
- 仓库：github.com/Wkstr/rk3576-speech-bench（私有，P5 发布时翻公开）

## 评测口径（已定，别回退）

- 报告双语（中文正文 + 英文摘要）；样本 zh/en 各 30、日韩粤各 10（P2 顺延中）；数字带 Wilson 95% 置信区间
- NPU 自转 fp16（toolkit2 2.3.2）vs CPU fp32 基线，同 harness 切 provider（官方 RKNN 版坏掉，仅作对照，见 #2861）
- 不测功耗；不做声音克隆；不做 Qwen3-ASR
