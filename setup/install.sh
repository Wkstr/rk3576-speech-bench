#!/usr/bin/env bash
# RK3576 ASR/TTS benchmark — 安装（幂等，可重复执行）
# 1) librknnrt 2.2.0 → ~/.local/rknn/lib/（用户态，免 sudo；锁 2.2.0 避开 sherpa-onnx 与 2.3.x/2.4.x 的已知崩溃）
# 2) .venv + sherpa-onnx 1.13.8 + benchmark 依赖
set -euo pipefail
cd "$(dirname "$0")/.."

RKNN_DIR="$HOME/.local/rknn/lib"
RKNN_URL="https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.2.0/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so"

echo "== 1/3 librknnrt 2.2.0 =="
mkdir -p "$RKNN_DIR"
lib_ver() { strings "$1" 2>/dev/null | grep "librknnrt version" || true; }
if [ -f "$RKNN_DIR/librknnrt.so" ] && lib_ver "$RKNN_DIR/librknnrt.so" | grep -q "2.2.0"; then
  echo "  已存在且版本正确，跳过"
else
  curl -fL --retry 3 -C - -o "$RKNN_DIR/librknnrt.so" "$RKNN_URL"
  ver=$(lib_ver "$RKNN_DIR/librknnrt.so")
  echo "  $ver"
  echo "$ver" | grep -q "2.2.0" || { echo "版本校验失败"; exit 1; }
fi

echo "== 2/3 python venv + 依赖 =="
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv || { echo "缺 python3-venv：sudo apt install python3.11-venv"; exit 1; }
fi
.venv/bin/pip install --upgrade -q pip
# 注意：PyPI 的 sherpa-onnx wheel 不含 RKNN provider，必须走官方 RKNN 索引（-cn 为国内镜像）
.venv/bin/pip install -q --force-reinstall --no-deps \
  "sherpa-onnx==1.13.8" -f https://k2-fsa.github.io/sherpa/onnx/rk-npu-cn.html
.venv/bin/pip install -q psutil pyyaml soundfile numpy jiwer editdistance
.venv/bin/python -c "import sherpa_onnx; print('  sherpa_onnx', sherpa_onnx.__version__)"
.venv/bin/python -c "import sherpa_onnx, subprocess; print(subprocess.run(['ldd', sherpa_onnx.__file__.replace('__init__.py','')+'lib/libsherpa-onnx-python-api.so'], capture_output=True, text=True).stdout)" 2>/dev/null | grep -q rknn && echo "  RKNN provider: OK" || echo "  RKNN provider: 待 NPU 冒烟确认"

echo "== 3/3 使用方式 =="
cat <<EOF
运行 benchmark 前 export：
  export RKNNRT_LIB_PATH="$RKNN_DIR/librknnrt.so"
  export LD_LIBRARY_PATH="$RKNN_DIR:\${LD_LIBRARY_PATH:-}"
（bench/ 脚本会自动设置，这里仅供手动调试用）
EOF
echo "安装完成。下一步：bash setup/check_env.sh"

