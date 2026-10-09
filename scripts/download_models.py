#!/usr/bin/env python3
"""按 models.yaml 下载并解压模型变体。幂等：目标目录非空则跳过。

用法:
  python scripts/download_models.py                    # 全部
  python scripts/download_models.py zipformer-small-bilingual:rknn sensevoice:cpu
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  缓存命中: {dest.name}")
        return
    print(f"  下载 {url}")
    subprocess.run(
        ["wget", "-q", "--show-progress", "-c", "-O", str(dest), url], check=True
    )


def install_variant(name: str, variant: str, url: str) -> None:
    out_dir = MODELS_DIR / name / variant
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[skip] {name}:{variant} 已存在")
        return
    tarball = MODELS_DIR / "_cache" / Path(url).name
    fetch(url, tarball)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  解压 → {out_dir}")
    subprocess.run(
        ["tar", "xf", str(tarball), "-C", str(out_dir), "--strip-components=1"],
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="*", help="name:variant，缺省为全部")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "models.yaml").read_text())
    wanted = set(args.specs)

    for task in ("asr", "tts"):
        for name, m in (cfg.get(task) or {}).items():
            for variant, v in m["variants"].items():
                if wanted and f"{name}:{variant}" not in wanted:
                    continue
                if "url" not in v:  # 自转变体无下载源，仅登记
                    print(f"[skip] {name}:{variant} — {v.get('note', 'self-converted')}")
                    continue
                print(f"[get] {name}:{variant} — {m['desc']}")
                install_variant(name, variant, v["url"])

    for name, a in (cfg.get("aux") or {}).items():
        if wanted and name not in wanted:
            continue
        dest = ROOT / a["dest"]
        print(f"[get] aux: {name}")
        fetch(a["url"], dest)

    print("完成。")


if __name__ == "__main__":
    sys.exit(main())
