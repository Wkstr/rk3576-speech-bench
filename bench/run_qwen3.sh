#!/usr/bin/env bash
# Qwen3-ASR 跑分：rknn+rkllm（w4a16_g128）× zh/en，双 pass（RTF + 定稿延迟）。
# 用法：bash bench/run_qwen3.sh [limit] [langs]  # limit 非空则每配置只跑前 N 条（冒烟）
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LIMIT=${1:-0}
LANGS=${2:-"en zh"}

for LANG in $LANGS; do
  M="data/librispeech-test-clean/manifest.jsonl"
  [ "$LANG" = zh ] && M="data/fleurs-zh/manifest.jsonl"
  [ -f "$M" ] || { echo "[skip] $M 不存在"; continue; }
  echo "=== qwen3-asr__rknn-w4a16__${LANG} ==="
  $PY bench/run_asr_qwen3.py --manifest "$M" \
    --tag "qwen3-asr__rknn-w4a16__${LANG}" --limit "$LIMIT"
done
echo "全部完成: results/asr/"
