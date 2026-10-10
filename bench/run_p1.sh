#!/usr/bin/env bash
# P1 跑分矩阵：zipformer small(rknn-2.3.2 / cpu96) + large(cpu) × zh/en。
# 用法：bash bench/run_p1.sh [limit] [langs]  # limit 非空则每配置只跑前 N 条（冒烟）；langs 默认 "en zh"
set -euo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH="$HOME/.local/rknn/lib"
PY=.venv/bin/python
LIMIT=${1:-0}
LANGS=${2:-"en zh"}

S=models/zipformer-small-bilingual
L=models/zipformer-large-bilingual

run() { # tag provider manifest encoder decoder joiner tokens
  local tag=$1 provider=$2 manifest=$3 enc=$4 dec=$5 joi=$6 tok=$7
  echo "=== $tag ==="
  $PY bench/run_asr.py --provider "$provider" --tag "$tag" --manifest "$manifest" \
    --encoder "$enc" --decoder "$dec" --joiner "$joi" --tokens "$tok" \
    --limit "$LIMIT"
}

for LANG in $LANGS; do
  M="data/librispeech-test-clean/manifest.jsonl"
  [ "$LANG" = zh ] && M="data/fleurs-zh/manifest.jsonl"
  [ -f "$M" ] || { echo "[skip] $M 不存在"; continue; }

  # small NPU（自转 chunk96）
  run "zipformer-small__rknn-2.3.2__${LANG}" rknn "$M" \
    "$S/rknn-2.3.2/encoder.rknn" "$S/rknn-2.3.2/decoder.rknn" "$S/rknn-2.3.2/joiner.rknn" "$S/rknn-2.3.2/tokens.txt"
  # small CPU 基线（同 chunk96 口径）
  run "zipformer-small__cpu96__${LANG}" cpu "$M" \
    "$S/cpu/96/encoder-epoch-99-avg-1.onnx" "$S/cpu/96/decoder-epoch-99-avg-1.onnx" \
    "$S/cpu/96/joiner-epoch-99-avg-1.onnx" "$S/cpu/96/tokens.txt"
  # large CPU 基线（无 chunk 变体，root fp32）
  run "zipformer-large__cpu__${LANG}" cpu "$M" \
    "$L/cpu/encoder-epoch-99-avg-1.onnx" "$L/cpu/decoder-epoch-99-avg-1.onnx" \
    "$L/cpu/joiner-epoch-99-avg-1.onnx" "$L/cpu/tokens.txt"
  # large NPU（自转 chunk32，2026-10-09 转入）
  run "zipformer-large__rknn-2.3.2__${LANG}" rknn "$M" \
    "$L/rknn-2.3.2/encoder.rknn" "$L/rknn-2.3.2/decoder.rknn" "$L/rknn-2.3.2/joiner.rknn" "$L/rknn-2.3.2/tokens.txt"
done
echo "全部完成: results/asr/"
