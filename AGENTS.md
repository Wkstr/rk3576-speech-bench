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

**全部下载已到位**（2026-10-10）：FLEURS zh 已采样 30 条（tarball 已删）；TTS 五模型已解压到 `models/<name>/cpu/` + vocos 在 `models/vocos/`。P2 恢复时需另下 FLEURS ja_jp/ko_kr/yue_hant_hk（同 zh 链接换语言码）。

zh 源已从 AISHELL-1 换成 FLEURS cmn_hans_cn test（用户拍板 2026-10-09：AISHELL 太大；FLEURS 与 P2 日韩粤同源）。
已到位：zipformer small CPU+自转 RKNN / large CPU+自转 RKNN（`rknn-2.3.2/`）/ LibriSpeech en 30 条 + FLEURS zh 30 条已采样 / TTS 五模型。SenseVoice 全线移出当前范围（用户拍板，后续需要再补）。MMS 官方只打包了 eng，无 zh。

### P4 状态（2026-10-10，部分启动）

Matcha NPU 化实测闭环（note：`.agents/notes/implemented/testing/2026-10-10-matcha-npu-vocos-hybrid.md`）：estimator fp16 RKNN 精度不足不可用（上游生产也不用）；**混合版转正**：声学 CPU + vocos NPU（`--type matcha-hybrid`，RTF 0.18 vs CPU 0.25，回译一致）。**声码器错配修正**：matcha 是 16kHz 模型，首轮 CPU 数据错配 vocos-22khz（音频 1.38× 快进）已用 16k 重测。板上注意：rknn-toolkit-lite2 只认 `/usr/lib/librknnrt.so`（用户已软链到 2.3.2；sherpa 走 LD_LIBRARY_PATH 的 2.2.0 不受影响）；espeak C API 用 piper-phonemize 自带库。Kokoro NPU 有上游预编译包（HF harvestsu/seeed-local-voice-rk-artifacts），未启动。

### P5 状态（2026-10-10）

已发布：repo 已翻公开，站点 https://wkstr.github.io/rk3576-speech-bench/（gh-pages 分支，`git subtree push --prefix site origin gh-pages` 更新）。报告 `report/REPORT.md`（`report/build_report.py` 从 results JSON 渲染，重跑后需重跑生成器）。站点音频已拍板全量入库 `site/audio/`（84 条，~14MB）。遗留：NPU 占用待用户 sudo 提权后重测 4 个 rknn 配置 + 重生成报告/站点。

### P3 状态（2026-10-10）

TTS 矩阵跑完（`bash bench/run_p3.sh`）：RTF matcha 0.25 < piper-en 0.32 < piper-zh 0.34 < mms 1.66 < kokoro 2.87；roundtrip ER 13%–37%（口径=可懂度下界，数字口读/同音词不归一化）。回译截尾坑（尾部需补 1s 静音）与口径依据：`.agents/notes/implemented/testing/2026-10-10-tts-roundtrip-harness.md`。kokoro 只跑 zh+en（裁判是中英双语，ja/ko/yue 回译无意义，属 P2）。

### 可选提权（改善资源曲线）

`sudo chmod o+r /sys/kernel/debug/rknpu/load` —— 不解则 sysmon 的 npu_load 全为 null（其余 CPU/RSS/温度不受影响）。无 sudo 替代路径已排除：`/sys/class/devfreq/27700000.npu/load` 虽免 root 可读，但本板内核下恒报 `100@950000000Hz`（空闲与推理中不变，2026-10-10 实测），不可用。

注：zipformer large 自转 RKNN 已到位并语料级验收（2026-10-10）：en WER 11.4% 与 cpu 完全相同且 30/30 逐条一致，叠词定量 0 次（此前 smoke 的 ALWAYS ALWAYS 为孤立事件）；zh CER 7.57% vs cpu 7.76%。large 的 rknn 与 cpu RTF 同为 0.41（NPU 对 large 无加速），报告如实呈现。

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
