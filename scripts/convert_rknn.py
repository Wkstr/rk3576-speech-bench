#!/usr/bin/env python3
# Adapted from icefall: egs/librispeech/ASR/pruned_transducer_stateless7_streaming/export_rknn.py
# Copyright (c) 2025 Xiaomi Corporation (authors: Fangjun Kuang) — Apache 2.0
# Source: https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming/export_rknn.py
#
# 用途：把 sherpa-onnx CPU 版 ONNX（encoder/decoder/joiner）转换为 RKNN（target rk3576）。
# 官方 rk3576 zipformer RKNN 模型在 RK3576 上输出异常（small 全空 / large 叠词，
# 见 k2-fsa/sherpa-onnx#2861），疑似 2024-08 toolkit 2.1.0 Beta 期转换问题；
# 本脚本用 rknn-toolkit2 2.3.2 在板上重转。
#
# 运行环境：独立 venv（.venv-rknn），依赖见 rknn-toolkit2 arm64_requirements_cp311.txt。
# 用法示例：
#   .venv-rknn/bin/python scripts/convert_rknn.py --target-platform rk3576 \
#     --in-encoder  models/zipformer-small-bilingual/cpu/64/encoder-epoch-99-avg-1.onnx \
#     --in-decoder  models/zipformer-small-bilingual/cpu/64/decoder-epoch-99-avg-1.onnx \
#     --in-joiner   models/zipformer-small-bilingual/cpu/64/joiner-epoch-99-avg-1.onnx \
#     --out-encoder models/zipformer-small-bilingual/rknn-2.3.2/encoder.rknn \
#     --out-decoder models/zipformer-small-bilingual/rknn-2.3.2/decoder.rknn \
#     --out-joiner  models/zipformer-small-bilingual/rknn-2.3.2/joiner.rknn

import argparse
import logging
from pathlib import Path
from typing import List

from rknn.api import RKNN

logging.basicConfig(level=logging.WARNING)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--target-platform", type=str, default="rk3576")
    parser.add_argument("--in-encoder", type=str, required=True)
    parser.add_argument("--in-decoder", type=str, required=True)
    parser.add_argument("--in-joiner", type=str, required=True)
    parser.add_argument("--out-encoder", type=str, required=True)
    parser.add_argument("--out-decoder", type=str, required=True)
    parser.add_argument("--out-joiner", type=str, required=True)
    return parser


def export_rknn(rknn, filename):
    ret = rknn.export_rknn(filename)
    if ret != 0:
        exit(f"Export rknn model to {filename} failed!")
    print(f"Exported {filename}")


def init_model(filename: str, target_platform: str, custom_string=None):
    rknn = RKNN(verbose=False)
    rknn.config(target_platform=target_platform, custom_string=custom_string)
    if not Path(filename).is_file():
        exit(f"{filename} does not exist")

    ret = rknn.load_onnx(model=filename)
    if ret != 0:
        exit(f"Load model {filename} failed!")

    # 与官方一致：不做量化（fp16）。sherpa-onnx 官方 RKNN zipformer 即 fp16
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        exit(f"Build model {filename} failed!")

    return rknn


class MetaData:
    def __init__(
        self,
        model_type: str,
        attention_dims: List[int],
        encoder_dims: List[int],
        T: int,
        left_context_len: List[int],
        decode_chunk_len: int,
        cnn_module_kernels: List[int],
        num_encoder_layers: List[int],
        context_size: int,
    ):
        self.model_type = model_type
        self.attention_dims = attention_dims
        self.encoder_dims = encoder_dims
        self.T = T
        self.left_context_len = left_context_len
        self.decode_chunk_len = decode_chunk_len
        self.cnn_module_kernels = cnn_module_kernels
        self.num_encoder_layers = num_encoder_layers
        self.context_size = context_size

    def to_str(self) -> str:
        def to_s(ll):
            return ",".join(list(map(str, ll)))

        s = f"model_type={self.model_type}"
        s += ";attention_dims=" + to_s(self.attention_dims)
        s += ";encoder_dims=" + to_s(self.encoder_dims)
        s += ";T=" + str(self.T)
        s += ";left_context_len=" + to_s(self.left_context_len)
        s += ";decode_chunk_len=" + str(self.decode_chunk_len)
        s += ";cnn_module_kernels=" + to_s(self.cnn_module_kernels)
        s += ";num_encoder_layers=" + to_s(self.num_encoder_layers)
        s += ";context_size=" + str(self.context_size)
        assert len(s) < 1024, (s, len(s))
        return s


def get_meta_data(encoder: str, decoder: str):
    import onnxruntime

    session_opts = onnxruntime.SessionOptions()
    session_opts.inter_op_num_threads = 1
    session_opts.intra_op_num_threads = 1

    m_encoder = onnxruntime.InferenceSession(
        encoder, sess_options=session_opts, providers=["CPUExecutionProvider"]
    )
    m_decoder = onnxruntime.InferenceSession(
        decoder, sess_options=session_opts, providers=["CPUExecutionProvider"]
    )

    encoder_meta = m_encoder.get_modelmeta().custom_metadata_map
    decoder_meta = m_decoder.get_modelmeta().custom_metadata_map
    print("encoder meta:", encoder_meta)
    print("decoder meta:", decoder_meta)

    def to_int_list(s):
        return list(map(int, s.split(",")))

    return MetaData(
        model_type=encoder_meta["model_type"],
        attention_dims=to_int_list(encoder_meta["attention_dims"]),
        encoder_dims=to_int_list(encoder_meta["encoder_dims"]),
        T=int(encoder_meta["T"]),
        left_context_len=to_int_list(encoder_meta["left_context_len"]),
        decode_chunk_len=int(encoder_meta["decode_chunk_len"]),
        cnn_module_kernels=to_int_list(encoder_meta["cnn_module_kernels"]),
        num_encoder_layers=to_int_list(encoder_meta["num_encoder_layers"]),
        context_size=int(decoder_meta["context_size"]),
    )


def main():
    args = get_parser().parse_args()
    print(vars(args))

    meta = get_meta_data(args.in_encoder, args.in_decoder)
    print("metadata:", meta.to_str())

    encoder = init_model(
        args.in_encoder,
        custom_string=meta.to_str(),
        target_platform=args.target_platform,
    )
    decoder = init_model(args.in_decoder, target_platform=args.target_platform)
    joiner = init_model(args.in_joiner, target_platform=args.target_platform)

    export_rknn(encoder, args.out_encoder)
    export_rknn(decoder, args.out_decoder)
    export_rknn(joiner, args.out_joiner)

    encoder.release()
    decoder.release()
    joiner.release()


if __name__ == "__main__":
    main()
