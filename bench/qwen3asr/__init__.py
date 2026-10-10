"""Qwen3-ASR RKNN+RKLLM engine，vendor 自 suharvest/rkvoice-stream（Apache-2.0）。

上游: https://github.com/suharvest/rkvoice-stream
  rkvoice_stream/backends/asr/qwen3/ @ ac65833 (2026-09-27)
仅 vendor 离线/流式转写所需 7 个模块（config/mel/utils/encoder/decoder/stream/engine），
跳过 chunk_confirm.py、streaming.py、matmul_decoder.py 及对 rkvoice_stream 包其余部分的依赖。
"""

from .engine import Qwen3ASREngine

__all__ = ["Qwen3ASREngine"]
