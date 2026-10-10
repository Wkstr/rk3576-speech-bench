#!/usr/bin/env bash
# P3 TTS 矩阵：piper-zh / piper-en / matcha / kokoro / mms × 各自语言（全 CPU fp32）。
# kokoro 只跑 zh+en：回译裁判 zipformer small 是中英双语，ja/ko/yue 回译无意义（属 P2）。
# 用法：bash bench/run_p3.sh [limit]   # limit 非空则每模型只跑前 N 条（冒烟）
set -euo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH="$HOME/.local/rknn/lib"
PY=.venv/bin/python
LIMIT=${1:-0}

PZ=models/piper-zh-huayan/cpu
PE=models/piper-en-ryan/cpu
MA=models/matcha-zh-en/cpu
KO=models/kokoro/cpu
MM=models/mms-eng/cpu
VOC=models/vocos/vocos-22khz-univ.onnx

run() { # tag type langs... -- extra args 往 run_tts.py 透传
  local tag=$1 type=$2; shift 2
  echo "=== $tag ==="
  $PY bench/run_tts.py --type "$type" --tag "$tag" --limit "$LIMIT" "$@"
}

run piper-zh-huayan__zh vits \
  --langs zh \
  --model "$PZ/zh_CN-huayan-medium.onnx" --tokens "$PZ/tokens.txt" --data-dir "$PZ/espeak-ng-data"

run piper-en-ryan__en vits \
  --langs en \
  --model "$PE/en_US-ryan-medium.onnx" --tokens "$PE/tokens.txt" --data-dir "$PE/espeak-ng-data"

run matcha-zh-en__zh-en matcha \
  --langs zh en \
  --model "$MA/model-steps-3.onnx" --vocoder "$VOC" --tokens "$MA/tokens.txt" \
  --lexicon "$MA/lexicon.txt" --data-dir "$MA/espeak-ng-data"

run kokoro__zh-en kokoro \
  --langs zh en \
  --model "$KO/model.onnx" --voices "$KO/voices.bin" --tokens "$KO/tokens.txt" \
  --lexicon "$KO/lexicon-us-en.txt,$KO/lexicon-zh.txt" \
  --data-dir "$KO/espeak-ng-data" --dict-dir "$KO/dict"

run mms-eng__en vits \
  --langs en \
  --model "$MM/model.onnx" --tokens "$MM/tokens.txt"

echo "全部完成: results/tts/"
