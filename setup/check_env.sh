#!/usr/bin/env bash
# RK3576 ASR/TTS benchmark — 环境自检（只读，不安装任何东西）
# 出口码：0 = 全部通过；1 = 有 FAIL 项
set -uo pipefail

PASS=0; FAIL=0
ok()   { echo "  ✅ $1"; }
warn() { echo "  ⚠️  $1"; }
bad()  { echo "  ❌ $1"; FAIL=1; }

echo "== 基本信息 =="
echo "  kernel : $(uname -r)"
echo "  arch   : $(uname -m)   ($(tr '\0' ' ' < /proc/device-tree/compatible 2>/dev/null | awk '{print $1" "$2}'))"
echo "  python : $(python3 --version 2>&1)"
echo "  disk   : $(df -h / | awk 'NR==2{print $4" free / "$2}')"
echo "  mem    : $(free -h | awk '/^Mem:/{print $7" avail / "$2}')"

echo "== NPU =="
if id -nG | grep -qw render; then ok "当前用户在 render 组（NPU 推理免 sudo）"; else bad "用户不在 render 组"; fi
if ls /dev/dri/renderD* >/dev/null 2>&1; then ok "render 节点存在: $(ls /dev/dri/renderD* | tr '\n' ' ')"; else bad "无 /dev/dri/renderD* 节点"; fi
if [ -r /sys/kernel/debug/rknpu/version ]; then
  ok "rknpu 驱动版本: $(cat /sys/kernel/debug/rknpu/version)"
else
  warn "debugfs 不可读，无法确认 rknpu 驱动版本（可选：sudo cat /sys/kernel/debug/rknpu/version）"
fi

echo "== 运行时依赖 =="
RKNN_LIB="${RKNNRT_LIB_PATH:-$HOME/.local/rknn/lib/librknnrt.so}"
if [ -f "$RKNN_LIB" ]; then
  ver=$(strings "$RKNN_LIB" 2>/dev/null | grep "librknnrt version" || true)
  ok "librknnrt: $RKNN_LIB  ($ver)"
else
  warn "librknnrt 未安装（setup/install.sh 会装到 $RKNN_LIB）"
fi
if python3 -c "import sherpa_onnx" 2>/dev/null; then
  ok "sherpa_onnx: $(python3 -c 'import sherpa_onnx; print(sherpa_onnx.__version__)' 2>/dev/null)"
elif [ -x "$(dirname "$0")/../.venv/bin/python" ] && "$(dirname "$0")/../.venv/bin/python" -c "import sherpa_onnx" 2>/dev/null; then
  ok "sherpa_onnx (venv): $("$(dirname "$0")/../.venv/bin/python" -c 'import sherpa_onnx; print(sherpa_onnx.__version__)')"
else
  warn "sherpa_onnx 未安装（setup/install.sh 会装）"
fi

echo "== 网络 =="
for u in https://pypi.org https://github.com https://raw.githubusercontent.com https://hf-mirror.com; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 -I "$u" 2>/dev/null)
  if [ "${code:-000}" != "000" ]; then ok "$u -> $code"; else warn "$u 不可达"; fi
done

echo "== 温度传感器 =="
if sensors >/dev/null 2>&1; then ok "lm-sensors 可用（npu/soc/CPU 温度可采）"; else warn "lm-sensors 不可用"; fi

echo
if [ "$FAIL" -eq 0 ]; then echo "自检通过（警告项不阻塞）"; else echo "存在 FAIL 项，请先修复"; exit 1; fi
