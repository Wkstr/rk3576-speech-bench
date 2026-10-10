#!/usr/bin/env python3
"""TTS benchmark harness：合成时延/RTF + 样本落盘 + ASR 回译可懂度。

sherpa-onnx OfflineTts 为一次性合成，无流式首包概念——记录整段生成时间与 RTF；
"首包延迟"仅在有流式引擎时才有意义，本矩阵全离线引擎，报告中以 RTF 为主。

用法（driver 见 bench/run_p3.sh）：
  .venv/bin/python bench/run_tts.py --type vits --tag piper-zh__zh --langs zh \
    --model M/zh_CN-huayan-medium.onnx --tokens M/tokens.txt --data-dir M/espeak-ng-data
  回译（可选，默认开）：用 zipformer small cpu96 离线转写合成音频算 CER/WER。

结果 JSON 与 run_asr.py 同构：config/sysmon/aggregate/items。
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from metrics import edit_counts, percentiles, wilson  # noqa: E402
from run_asr import board_info, make_recognizer, pass_throughput  # noqa: E402
from sysmon import Sysmon  # noqa: E402


def build_tts(args):
    import sherpa_onnx

    if args.type == "matcha-rknn":
        # 全 NPU 拆分路径（对照用）：--model 指向 rknn 产物目录，--data-dir 指向前端目录（cpu/）
        from matcha_rknn import MatchaRKNN
        engine = MatchaRKNN(rknn_dir=args.model, frontend_dir=args.data_dir,
                            ode_steps=int(os.environ.get("MATCHA_ODE_STEPS", "1")))
        engine.load()
        return engine
    if args.type == "matcha-hybrid":
        # 声学 CPU ONNX + vocos NPU：--model=声学 onnx，--vocoder=vocos rknn，--data-dir=前端目录
        from matcha_rknn import MatchaHybrid
        engine = MatchaHybrid(acoustic_onnx=args.model, vocos_rknn=args.vocoder,
                              frontend_dir=args.data_dir)
        engine.load()
        return engine

    common = dict(
        num_threads=args.num_threads,
        provider=args.provider,
        debug=False,
    )
    if args.type == "vits":
        model_cfg = sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=args.model, lexicon=args.lexicon or "",
                tokens=args.tokens, data_dir=args.data_dir or "",
                dict_dir=args.dict_dir or "",
            ), **common)
    elif args.type == "matcha":
        model_cfg = sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=args.model, vocoder=args.vocoder,
                lexicon=args.lexicon or "", tokens=args.tokens,
                data_dir=args.data_dir or "", dict_dir=args.dict_dir or "",
            ), **common)
    elif args.type == "kokoro":
        model_cfg = sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=args.model, voices=args.voices,
                lexicon=args.lexicon or "", tokens=args.tokens,
                data_dir=args.data_dir or "", dict_dir=args.dict_dir or "",
            ), **common)
    else:
        raise SystemExit(f"未知 type: {args.type}")

    cfg = sherpa_onnx.OfflineTtsConfig(model=model_cfg, max_num_sentences=1)
    if not cfg.validate():
        raise SystemExit("TTS config 校验失败，检查文件路径")
    return sherpa_onnx.OfflineTts(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True,
                    choices=["vits", "matcha", "matcha-rknn", "matcha-hybrid", "kokoro"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--langs", nargs="+", required=True, help="跑哪些语言的文本")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", default="")  # matcha-rknn 不需要（前端目录自带）
    ap.add_argument("--vocoder", default="")
    ap.add_argument("--voices", default="")
    ap.add_argument("--lexicon", default="")
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--dict-dir", default="")
    ap.add_argument("--provider", default="cpu")
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument("--texts", default=str(ROOT / "bench/data/tts_texts.json"))
    ap.add_argument("--out", default=str(ROOT / "results/tts"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-roundtrip", action="store_true", help="跳过 ASR 回译")
    args = ap.parse_args()
    if not args.type.startswith("matcha-") and not args.tokens:
        raise SystemExit("--tokens 必填（matcha-rknn/matcha-hybrid 除外）")

    texts = json.loads(Path(args.texts).read_text())["texts"]
    texts = [t for t in texts if t["lang"] in args.langs]
    if args.limit:
        texts = texts[: args.limit]
    print(f"[run] {args.tag}: {len(texts)} 条文本, type={args.type}")

    tts = build_tts(args)

    # 回译 ASR：zipformer small cpu96（与 ASR 基线同口径）
    asr = None
    if not args.no_roundtrip:
        class AsrArgs:
            provider = "cpu"
            num_threads = 4
            encoder = str(ROOT / "models/zipformer-small-bilingual/cpu/96/encoder-epoch-99-avg-1.onnx")
            decoder = str(ROOT / "models/zipformer-small-bilingual/cpu/96/decoder-epoch-99-avg-1.onnx")
            joiner = str(ROOT / "models/zipformer-small-bilingual/cpu/96/joiner-epoch-99-avg-1.onnx")
            tokens = str(ROOT / "models/zipformer-small-bilingual/cpu/96/tokens.txt")
        asr = make_recognizer(AsrArgs)

    samples_dir = Path(args.out) / "samples" / args.tag
    samples_dir.mkdir(parents=True, exist_ok=True)

    mon = Sysmon()
    mon.start()
    items = []
    for i, t in enumerate(texts):
        t0 = time.monotonic()
        audio = tts.generate(t["text"], sid=0, speed=1.0)
        gen = time.monotonic() - t0
        wav = np.array(audio.samples, dtype="float32")
        dur = len(wav) / audio.sample_rate
        sf.write(str(samples_dir / f"{t['id']}.wav"), wav, audio.sample_rate, subtype="PCM_16")

        # 回译：合成音频重采样到 16k 再过 ASR
        rt = None
        if asr is not None and dur > 0.3:
            w16 = _resample(wav, audio.sample_rate, 16000)
            # 尾部补 1s 静音：流式 zipformer 需要尾静音冲刷最后 chunk 的右上下文，
            # TTS 音频尾音戛然而止，不补会被截尾（zh_01 “今天天气怎么样” → “今天天”）。
            w16 = np.concatenate([w16, np.zeros(16000, dtype=np.float32)])
            hyp, _ = pass_throughput(asr, w16)
            errors, units = edit_counts(t["text"], hyp, t["lang"])
            rt = dict(hyp=hyp, errors=errors, ref_units=units,
                      error_rate=round(errors / units, 4) if units else None)
        items.append(dict(
            id=t["id"], lang=t["lang"], text=t["text"],
            duration=round(dur, 2), gen_time_s=round(gen, 3),
            rtf=round(gen / dur, 4) if dur > 0 else None,
            roundtrip=rt,
        ))
        print(f"  [{i + 1}/{len(texts)}] {t['id']}: dur={dur:.1f}s gen={gen:.2f}s "
              f"rtf={items[-1]['rtf']} rt_er={rt['error_rate'] if rt else '-'}")

    series = mon.stop()

    rt_items = [it for it in items if it["roundtrip"]]
    total_err = sum(it["roundtrip"]["errors"] for it in rt_items)
    total_units = sum(it["roundtrip"]["ref_units"] for it in rt_items)
    center, lo, hi = wilson(total_err, total_units)
    total_audio = sum(it["duration"] for it in items)
    total_gen = sum(it["gen_time_s"] for it in items)
    aggregate = {
        "rtf_mean": round(total_gen / total_audio, 4) if total_audio else None,
        "rtfx": round(total_audio / total_gen, 2) if total_gen else None,
        "gen_time_s": percentiles([it["gen_time_s"] for it in items]),
        "roundtrip_error_rate": {"center": round(center, 4), "wilson95": [round(lo, 4), round(hi, 4)],
                                 "errors": total_err, "units": total_units} if rt_items else None,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "config": {"tag": args.tag, "type": args.type, "provider": args.provider,
                   "num_threads": args.num_threads, "model": args.model,
                   "langs": args.langs, "roundtrip_asr": "zipformer-small cpu96" if asr else None,
                   **board_info(),
                   "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "sysmon": mon.summary(),
        "aggregate": aggregate,
        "items": items,
    }
    (out / f"{args.tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    with (out / f"{args.tag}.sysmon.jsonl").open("w") as f:
        for s in series:
            f.write(json.dumps(s) + "\n")
    print(f"[done] {out / (args.tag + '.json')}")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


def _resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return wav
    n = int(len(wav) * sr_out / sr_in)
    x = np.arange(len(wav)) / sr_in
    xi = np.arange(n) / sr_out
    return np.interp(xi, x, wav).astype("float32")


if __name__ == "__main__":
    main()
