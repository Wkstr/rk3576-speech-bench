# Matcha + Vocos → RKNN（RK3576）PC 转换说明

本目录 4 个脚本原样 vendor 自 [rkvoice-stream](https://github.com/suharvest/rkvoice-stream)（Apache 2.0，`models/tts/matcha/`），未改动。
方案要点：Matcha 声学模型与 Vocos 声码器都转 RKNN fp16 跑 NPU；ODE 累加环和 ISTFT 留在 CPU（fp32）保精度。
文本前端（lexicon/FST/espeak-ng）板上已有现成的（`models/matcha-zh-en/cpu/`），不在本流程内。

> 板上集成（rknn-toolkit-lite2 + run_tts.py rknn provider）在转换产物到位后由 agent 做，见文末「产出物落位」。

## 0. 环境（x86 PC）

复用你转 zipformer 的 rknn 环境（toolkit2 2.3.x），补装/钉版：

```bash
source <你的 rknn venv>/bin/activate
pip install "onnx==1.16.1" onnx-simplifier onnxruntime numpy
```

- `onnx` 必须钉 1.16.1（高版本与 toolkit2 2.3.x 不兼容，上游文档明示）
- 未验证过的组合：toolkit2 2.3.0（上游钉版）vs 我们的 2.3.2——先用现有 2.3.2 试，失败再降 2.3.0

## 1. 下载输入文件（放 `~/matcha-data/`，脚本默认路径）

| 文件 | 来源 | 说明 |
|---|---|---|
| `model-steps-3.onnx` | 板上 `models/matcha-zh-en/cpu/model-steps-3.onnx` scp 过去即可；或重新下 [sherpa-onnx 发布包](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-en.tar.bz2) | Matcha 声学模型（75MB） |
| `vocos-16khz-univ.onnx` | https://hf-mirror.com/csukuangfj/sherpa-onnx-vocoders/resolve/main/vocos-16khz-univ.onnx | ⚠️ 是 **16kHz** 版，不是板上 22kHz 那个 |

## 2. 转换三步（按序）

```bash
cd <本目录 scripts/matcha-rknn>

# ① 修声学模型的动态算子（默认 s64 bucket：seq_len=80, x_len=64 → 单条最长 ~9.6s 音频 / 64 音素）
python fix_matcha_rknn.py \
  --input ~/matcha-data/model-steps-3.onnx \
  --output ~/matcha-data/model-steps-3-rknn-ready.onnx
# 通过标志：末尾打印 "Shape MATCH!"、"Cosine similarity vs original" ≈ 1.000000、"ORT OK"
# 想要更长句（~20s）：加 --seq-len 160 --x-len 140（s140 bucket，慢一倍）；
# ⚠️ 不要 --seq-len 256：超 RK3576 NPU SDP attention 内存上限，推理会失败（上游实测）

# ② 拆 encoder + estimator 并转 RKNN fp16（拆分是为绕开 ODE 累加的 fp16 精度丢失）
python split_matcha_rknn.py --all \
  --onnx ~/matcha-data/model-steps-3-rknn-ready.onnx \
  --target rk3576
# 通过标志："PASS: Split model is numerically identical"（max diff < 1e-3）+ 两个 .rknn 导出 OK
# 产出（与输入同目录）：matcha-encoder-fp16.rknn、matcha-estimator-fp16.rknn、
#                     time_emb_step0/1/2.npy（以及中间产物 matcha-encoder/estimator.onnx）

# ③ 转 Vocos 声码器（fp16，600 帧定长输入；rk3576 不需要 Clip 补丁，脚本只给 rk3588 打）
python convert_vocos_16khz_rknn.py
# 通过标志：导出 vocos-16khz-univ-fp16.rknn（~27MB）
```

## 3. 产出物落位（scp 回板子）

```bash
# 板上目标目录
ssh jy@<板子> mkdir -p /home/jy/asrtts-benchmark/models/matcha-zh-en/rknn
cd ~/matcha-data
scp matcha-encoder-fp16.rknn matcha-estimator-fp16.rknn \
    time_emb_step0.npy time_emb_step1.npy time_emb_step2.npy \
    vocos-16khz-univ-fp16.rknn \
    jy@<板子>:/home/jy/asrtts-benchmark/models/matcha-zh-en/rknn/
```

放好通知 agent 即可，剩下的（rknn-toolkit-lite2 安装、run_tts.py rknn provider、矩阵跑分）在板上完成。

## 预期性能（上游 RK3576 实测，s64 + vocos-600）

Matcha RKNN ~430ms + Vocos RKNN ~80ms + ISTFT ~50ms（CPU）+ 前端 <10ms（CPU），
典型句 ~570ms 合出 7.6s 音频，**RTF ~0.07**（对照：我们 CPU 全量实测 RTF 0.25）。

## 已知坑（上游文档 + 脚本注释）

1. 声学模型有动态 `Range`/`Slice`/`Ceil`/`RandomNormalLike`——fix 脚本已全部静态化处理（probe-first，先探原模型再改，避免级联 shape 污染）
2. 整模型直接转 fp16 会因 ODE 累加丢精度（mel 能量塌陷）——所以拆 encoder/estimator、ODE 环留 CPU fp32
3. vocos 输出是 STFT 三分量（mag/x/y），不是波形——板上做 ISTFT（CPU，~50ms）
4. RK3588 的 vocos 有 Clip 空 min 全零 bug，脚本只对 `TARGET=rk3588` 打补丁；rk3576 不受影响
5. bucket 即最长音素数：s64 = 64 音素约 9.6s 音频，超出要分句（板上 engine 侧处理）

## 参考

- 上游方案文档：rkvoice-stream `docs/matcha-zh-en-tts.md`（注意其路径与性能表偏旧，以 `rkvoice_stream/backends/tts/matcha.py` 头注释为准）
- sherpa-onnx 模型页：https://k2-fsa.github.io/sherpa/onnx/tts/all/Chinese-English/matcha-icefall-zh-en.html
