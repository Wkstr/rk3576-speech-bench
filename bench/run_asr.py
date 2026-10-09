#!/usr/bin/env python3
"""ASR benchmark harness：同一代码切 provider（rknn/cpu），流式双_pass 测量。

每条音频跑两遍：
  A 吞吐 pass：整段一次喂入，墙钟 → RTF / RTFx（纯算力口径）
  B 实时 pass：100ms 帧按真实节奏喂，记首字延迟（首个稳定 partial）与端点延迟（尾帧→final）

用法（driver 见 bench/run_p1.sh）：
  .venv/bin/python bench/run_asr.py --provider rknn \
    --encoder M/encoder.rknn --decoder M/decoder.rknn --joiner M/joiner.rknn --tokens M/tokens.txt \
    --manifest data/librispeech-test-clean/manifest.jsonl --tag small-rknn__en --out results/asr/
"""
import argparse
import json
import platform
import subprocess
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

CHUNK_MS = 100
SR = 16000


def board_info() -> dict:
    try:
        model = Path("/proc/device-tree/model").read_text().rstrip("\x00")
    except OSError:
        model = platform.machine()
    try:
        kernel = subprocess.check_output(["uname", "-r"], text=True).strip()
    except Exception:
        kernel = "?"
    return {"board": model, "kernel": kernel, "machine": platform.machine()}


def make_recognizer(args):
    import sherpa_onnx

    return sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        tokens=args.tokens,
        num_threads=args.num_threads,
        sample_rate=SR,
        feature_dim=80,
        provider=args.provider,
        enable_endpoint_detection=False,
        decoding_method="greedy_search",
    )


def pass_throughput(rec, samples: np.ndarray):
    """整段喂入。返回 (text, wall_s)。"""
    stream = rec.create_stream()
    t0 = time.monotonic()
    stream.accept_waveform(SR, samples)
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    wall = time.monotonic() - t0
    return rec.get_result(stream).strip(), wall


def pass_realtime(rec, samples: np.ndarray):
    """100ms 帧实时节奏。返回 (text, first_token_s, endpoint_s, decode_s)。"""
    stream = rec.create_stream()
    chunk = SR * CHUNK_MS // 1000
    t0 = time.monotonic()
    prev_partial = ""
    first_token = None
    decode_time = 0.0

    for off in range(0, len(samples), chunk):
        stream.accept_waveform(SR, samples[off:off + chunk])
        td = time.monotonic()
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        decode_time += time.monotonic() - td
        partial = rec.get_result(stream).strip()
        if partial and partial == prev_partial and first_token is None:
            first_token = time.monotonic() - t0  # 连续两次相同 → stable
        prev_partial = partial
        # 按真实节奏 pacing：本帧不足 100ms 则补足
        lag = t0 + (off + chunk) / SR - time.monotonic()
        if lag > 0:
            time.sleep(lag)

    t_audio_end = time.monotonic()
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    text = rec.get_result(stream).strip()
    endpoint = time.monotonic() - t_audio_end
    return text, first_token, endpoint, decode_time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["rknn", "cpu"])
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--joiner", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tag", required=True, help="结果文件名 results/asr/<tag>.json")
    ap.add_argument("--out", default=str(ROOT / "results/asr"))
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用）")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[run] {args.tag}: {len(rows)} 条, provider={args.provider}, threads={args.num_threads}")

    rec = make_recognizer(args)
    mon = Sysmon()
    mon.start()

    utts, n_text_diff = [], 0
    for i, row in enumerate(rows):
        samples, sr = sf.read(str(ROOT / row["wav"]), dtype="float32")
        assert sr == SR, f"{row['wav']}: {sr} != {SR}"
        dur = len(samples) / SR

        hyp_a, wall = pass_throughput(rec, samples)
        hyp_b, ft, ep, dec = pass_realtime(rec, samples)
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
              f"rtf={utts[-1]['rtf']} ft={utts[-1]['first_token_s']} ep={utts[-1]['endpoint_s']}")

    series = mon.stop()

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
            "tag": args.tag, "provider": args.provider,
            "encoder": args.encoder, "decoder": args.decoder,
            "joiner": args.joiner, "tokens": args.tokens,
            "num_threads": args.num_threads, "chunk_ms": CHUNK_MS,
            "decoding_method": "greedy_search", "manifest": args.manifest,
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
