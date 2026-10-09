#!/usr/bin/env python3
"""数据采样：从原始数据集抽固定 N 条，落盘 16k wav + manifest.jsonl。

用法：
  .venv/bin/python scripts/prepare_data.py en     # LibriSpeech test-clean（读 data/_cache/test-clean.tar.gz）
  .venv/bin/python scripts/prepare_data.py zh     # FLEURS 中文 test（读 data/_cache/fleurs-zh/，需用户先放好）

口径：随机 N 条，seed=42（datasets.yaml），时长限 2–15s，16kHz 原样落盘（不重采样）。
en 抽完默认删除 346MB 缓存 tarball（磁盘紧张；--keep-cache 保留）。
zh 源（用户手动下载 2 个文件到 data/_cache/fleurs-zh/，URL 见 datasets.yaml）：
  test.tar.gz (525MB, 音频) + test.tsv (0.5MB, 转写)
"""
import argparse
import io
import json
import random
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
N = 30
MIN_DUR, MAX_DUR = 2.0, 15.0
SR = 16000

sys.path.insert(0, str(ROOT / "bench"))
from textnorm import normalize  # noqa: E402


def write_wav(path: Path, samples, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sr, subtype="PCM_16")


def check_sr(sr: int, name: str) -> None:
    if sr != SR:
        raise SystemExit(f"{name}: 采样率 {sr} != {SR}，不重采样（口径外），请检查数据源")


def pick(candidates: list, n: int) -> list:
    """candidates: [(key, duration, payload)]，按时长过滤后 seed=42 抽样。"""
    ok = [c for c in candidates if MIN_DUR <= c[1] <= MAX_DUR]
    if len(ok) < n:
        raise SystemExit(f"符合条件样本不足：{len(ok)} < {n}")
    return random.Random(SEED).sample(ok, n)


def prep_en(args) -> None:
    import soundfile as sf

    tarball = ROOT / "data/_cache/test-clean.tar.gz"
    out = ROOT / "data/librispeech-test-clean"
    if not tarball.exists():
        raise SystemExit(f"缺 {tarball}（用户手动下载，URL 见 datasets.yaml）")

    with tarfile.open(tarball) as tf:
        # 第一遍：收 transcript + 时长
        texts, durations = {}, {}
        for m in tf:
            if m.name.endswith(".trans.txt"):
                for line in tf.extractfile(m).read().decode().splitlines():
                    uid, txt = line.split(" ", 1)
                    texts[uid] = txt
            elif m.name.endswith(".flac"):
                f = tf.extractfile(m)
                info = sf.info(io.BytesIO(f.read()))
                uid = Path(m.name).stem
                durations[uid] = (info.duration, info.samplerate)

        candidates = [
            (uid, dur, None)
            for uid, (dur, sr) in durations.items()
            if uid in texts and sr == SR
        ]
        chosen = pick(candidates, N)
        chosen_ids = {uid for uid, _, _ in chosen}

        # 第二遍：抽中项落盘
        manifest = []
        for m in tf:
            uid = Path(m.name).stem
            if not (m.name.endswith(".flac") and uid in chosen_ids):
                continue
            data, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float32")
            check_sr(sr, uid)
            wav = out / f"en_{len(manifest):03d}.wav"
            write_wav(wav, data, sr)
            manifest.append(
                dict(id=wav.stem, lang="en", wav=str(wav.relative_to(ROOT)),
                     text=texts[uid], text_norm=normalize(texts[uid], "en"),
                     duration=round(len(data) / sr, 2), source_uid=uid)
            )

    write_manifest(out, manifest)
    if args.keep_cache:
        print(f"保留缓存 {tarball}")
    else:
        tarball.unlink()
        print(f"已删缓存 {tarball}（--keep-cache 可保留）")


def prep_zh(args) -> None:
    """FLEURS cmn_hans_cn test：test.tar.gz 内 *.wav + test.tsv（制表符分列）。"""
    import soundfile as sf

    base = ROOT / "data/_cache/fleurs-zh"
    tsv, tgz = base / "test.tsv", base / "test.tar.gz"
    if not (tsv.exists() and tgz.exists()):
        raise SystemExit(f"缺 {tsv} / {tgz}（用户手动下载，URL 见 datasets.yaml）")

    # tsv 无表头：一列是 xxx.wav 文件名，文本列取其余列中 CJK 最多的
    texts = {}
    for line in tsv.read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        wav_idx = next((i for i, c in enumerate(cols) if c.strip().endswith(".wav")), None)
        if wav_idx is None:
            continue
        others = [c for i, c in enumerate(cols) if i != wav_idx]
        text = max(others, key=lambda c: len(re.findall(r"[一-鿿]", c)), default="")
        if text.strip():
            texts[Path(cols[wav_idx].strip()).stem] = text.strip()
    print(f"tsv 转写: {len(texts)} 条")

    with tarfile.open(tgz) as tf:
        durations = {}
        for m in tf:
            if m.name.endswith(".wav"):
                info = sf.info(io.BytesIO(tf.extractfile(m).read()))
                durations[Path(m.name).stem] = (info.duration, info.samplerate)
        candidates = [
            (uid, dur, None)
            for uid, (dur, sr) in durations.items()
            if uid in texts and sr == SR
        ]
        chosen = pick(candidates, N)
        chosen_ids = {uid for uid, _, _ in chosen}

        out = ROOT / "data/fleurs-zh"
        manifest = []
        for m in tf:
            uid = Path(m.name).stem
            if not (m.name.endswith(".wav") and uid in chosen_ids):
                continue
            data, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float32")
            check_sr(sr, uid)
            wav = out / f"zh_{len(manifest):03d}.wav"
            write_wav(wav, data, sr)
            manifest.append(
                dict(id=wav.stem, lang="zh", wav=str(wav.relative_to(ROOT)),
                     text=texts[uid], text_norm=normalize(texts[uid], "zh"),
                     duration=round(len(data) / sr, 2), source_uid=uid)
            )
    write_manifest(out, manifest)
    if not args.keep_cache:
        tgz.unlink()
        print(f"已删缓存 {tgz}（--keep-cache 可保留）")


def write_manifest(out: Path, manifest: list) -> None:
    out.mkdir(parents=True, exist_ok=True)
    mp = out / "manifest.jsonl"
    with mp.open("w") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    dur = sum(r["duration"] for r in manifest)
    print(f"{mp}: {len(manifest)} 条，共 {dur:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("lang", choices=["en", "zh"])
    ap.add_argument("--keep-cache", action="store_true")
    args = ap.parse_args()
    {"en": prep_en, "zh": prep_zh}[args.lang](args)


if __name__ == "__main__":
    main()
