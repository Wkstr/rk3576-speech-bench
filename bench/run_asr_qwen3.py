#!/usr/bin/env python3
"""Qwen3-ASR（RKNN encoder + RKLLM decoder）benchmark harness，与 run_asr.py 同口径。

引擎 vendor 自 suharvest/rkvoice-stream（bench/qwen3asr/，Apache-2.0）。
与 sherpa 系的差异：
  - 语言自动检测（不传 hint，与 zipformer 同口径）；
  - 两个 pass 都用 kv_streaming 模式（边喂边编码、finish 单次整段解码，其 V2V 生产路径）。
    冒烟实测 transcribe() 的滑窗文本拼接在 en 上丢段/丢缝空格（中间块 6 token 早停
    + rollback 重 decode 预算耗尽所致），kv 整段解码无此问题且精度最高；
  - A pass 整段一次喂入测 RTF；B pass 100ms 帧真实节奏喂入，endpoint = 尾帧后
    finish() 墙钟（尾部 encode + 全量 decode，对应其 Finalize 口径）；
  - kv 模式无流式 partial，first_token_s 恒为 null（首词≈定稿，报告注明）；
  - apply_itn_flag=False：板上无 chinese_itn 依赖且与 zipformer 原始输出同口径。

用法（driver 见 bench/run_qwen3.sh）：
  .venv/bin/python bench/run_asr_qwen3.py --manifest data/fleurs-zh/manifest.jsonl \
    --tag qwen3-asr__rknn-w4a16__zh --limit 5
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from metrics import edit_counts, percentiles, wilson  # noqa: E402
from sysmon import Sysmon  # noqa: E402
from run_asr import board_info  # noqa: E402

CHUNK_MS = 100
SR = 16000
ENC_CHUNK_S = 4.0   # kv 模式 feed 时的编码粒度（4s encoder 窗口）
MAX_NEW_TOKENS = 256  # 15s 英文约 80 token，留足余量


def _kv_stream(engine):
    return engine.create_stream(language=None, chunk_size=ENC_CHUNK_S,
                                kv_streaming=True, max_new_tokens=MAX_NEW_TOKENS)


def pass_throughput(engine, samples: np.ndarray):
    """整段喂入（kv：全部 encode + 单次 decode）。返回 (text, wall_s)。"""
    stream = _kv_stream(engine)
    t0 = time.monotonic()
    stream.feed_audio(samples)
    res = stream.finish(apply_itn_flag=False)
    wall = time.monotonic() - t0
    return res["text"].strip(), wall


def pass_realtime(engine, samples: np.ndarray):
    """100ms 帧真实节奏喂 kv 流。返回 (text, endpoint_s)。kv 无 partial，不测首词。"""
    stream = _kv_stream(engine)
    chunk = SR * CHUNK_MS // 1000
    t0 = time.monotonic()

    for off in range(0, len(samples), chunk):
        stream.feed_audio(samples[off:off + chunk])
        lag = t0 + (off + chunk) / SR - time.monotonic()
        if lag > 0:
            time.sleep(lag)

    t_audio_end = time.monotonic()
    res = stream.finish(apply_itn_flag=False)
    endpoint = time.monotonic() - t_audio_end
    return res["text"].strip(), endpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tag", required=True, help="结果文件名 results/asr/<tag>.json")
    ap.add_argument("--out", default=str(ROOT / "results/asr"))
    ap.add_argument("--model-dir", default=str(ROOT / "models/qwen3-asr"))
    ap.add_argument("--lib", default=str(ROOT / "models/lib/librkllmrt.so"))
    ap.add_argument("--decoder-quant", default="w4a16_g128")
    ap.add_argument("--encoder-sizes", default="2,4")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用）")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[run] {args.tag}: {len(rows)} 条, decoder={args.decoder_quant}, "
          f"kv_streaming enc_chunk={ENC_CHUNK_S}s, 语言=auto", flush=True)

    from qwen3asr import Qwen3ASREngine
    engine = Qwen3ASREngine(
        args.model_dir, platform="rk3576", lib_path=args.lib,
        decoder_quant=args.decoder_quant,
        encoder_sizes=[int(s) for s in args.encoder_sizes.split(",")],
        enabled_cpus=4, verbose=False)
    print(f"[engine] ready, max_chunk={engine.max_chunk_seconds}s", flush=True)

    mon = Sysmon()
    mon.start()

    utts, n_text_diff = [], 0
    try:
        for i, row in enumerate(rows):
            samples, sr = sf.read(str(ROOT / row["wav"]), dtype="float32")
            assert sr == SR, f"{row['wav']}: {sr} != {SR}"
            dur = len(samples) / SR

            hyp_a, wall = pass_throughput(engine, samples)
            hyp_b, ep = pass_realtime(engine, samples)
            ft = None  # kv 模式无流式 partial
            if hyp_a != hyp_b:
                n_text_diff += 1

            errors, units = edit_counts(row["text"], hyp_a, row["lang"])
            utts.append(dict(
                id=row["id"], lang=row["lang"], duration=round(dur, 2),
                ref=row["text"], hyp=hyp_a,
                errors=errors, ref_units=units,
                error_rate=round(errors / units, 4) if units else None,
                rtf=round(wall / dur, 4),
                first_token_s=round(ft, 3) if ft is not None else None,
                endpoint_s=round(ep, 3),
            ))
            print(f"  [{i + 1}/{len(rows)}] {row['id']}: er={utts[-1]['error_rate']} "
                  f"rtf={utts[-1]['rtf']} ep={utts[-1]['endpoint_s']}", flush=True)
    finally:
        series = mon.stop()
        engine.close()

    total_err = sum(u["errors"] for u in utts)
    total_units = sum(u["ref_units"] for u in utts)
    total_dur = sum(u["duration"] for u in utts)
    center, lo, hi = wilson(total_err, total_units)
    metric_name = "wer" if rows[0]["lang"] == "en" else "cer"
    aggregate = {
        metric_name: {"center": round(center, 4), "wilson95": [round(lo, 4), round(hi, 4)],
                      "errors": total_err, "units": total_units},
        "rtf_mean": round(sum(u["rtf"] for u in utts) / len(utts), 4),
        "rtfx": round(total_dur / sum(u["rtf"] * u["duration"] for u in utts), 2),
        "first_token_s": percentiles([u["first_token_s"] for u in utts if u["first_token_s"] is not None]),
        "endpoint_s": percentiles([u["endpoint_s"] for u in utts]),
        "text_diff_ab": n_text_diff,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "config": {
            "tag": args.tag, "provider": "rknn+rkllm",
            "model": "Qwen3-ASR (rkvoice-stream artifacts, w4a16_g128 decoder + fp16 encoder 2s/4s)",
            "model_dir": args.model_dir, "decoder_quant": args.decoder_quant,
            "encoder_sizes": args.encoder_sizes, "language": "auto",
            "stream_mode": "kv_streaming", "enc_chunk_s": ENC_CHUNK_S,
            "max_new_tokens": MAX_NEW_TOKENS,
            "num_threads": 4, "chunk_ms": CHUNK_MS,
            "decoding_method": "greedy (top_k=1)", "manifest": args.manifest,
            **board_info(),
            "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "sysmon": mon.summary(),
        "aggregate": aggregate,
        "utterances": utts,
    }
    (out / f"{args.tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    with (out / f"{args.tag}.sysmon.jsonl").open("w") as f:
        for s in series:
            f.write(json.dumps(s) + "\n")
    print(f"[done] {out / (args.tag + '.json')}")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
