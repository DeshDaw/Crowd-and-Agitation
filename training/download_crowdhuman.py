"""
Download CrowdHuman (Shao et al., 2018).

~11 GB total: train01/02/03 zips (~8.4 GB), val zip (~2.5 GB), two .odgt
annotation files. Primary source is the dataset author's Hugging Face
mirror (sshao0516/CrowdHuman) — fast, resumable, no download quota. Falls
back to the Google Drive links from https://www.crowdhuman.org/download.html,
which routinely hit Drive's public-download quota ("many accesses" error).

Usage:
    pip install huggingface_hub   # gdown only needed for the Drive fallback
    python download_crowdhuman.py --root path/to/crowdhuman [--val-only]

Extracts every zip's Images/ content into <root>/images/ (flat, as
convert_crowdhuman.py expects).
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

HF_REPO = "sshao0516/CrowdHuman"

# Google Drive file IDs from crowdhuman.org (fallback; verify there if stale)
FILES = {
    "CrowdHuman_train01.zip": "134QOvaatwKdy0iIeNqA_p-xkAhkV4F8Y",
    "CrowdHuman_train02.zip": "17evzPh7gc1JBNvnW1ENXLy5Kr4Q_Nnla",
    "CrowdHuman_train03.zip": "1tdp0UCgxrqy1B6p8LkR-Iy0aIJ8l4fJW",
    "CrowdHuman_val.zip": "18jFI789CoHTppQ7vmRSFEdnGaSQZ4YzO",
    "annotation_train.odgt": "1UUTea5mYqvlUObsC1Z8CFldHJAtLtMX3",
    "annotation_val.odgt": "10WIRwu8ju8GRLuCkZ_vT6hnNxs5ptwoL",
}
VAL_ONLY = {"CrowdHuman_val.zip", "annotation_val.odgt"}


def hf_download(name: str, root: Path) -> bool:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub not installed; skipping HF mirror")
        return False
    try:
        hf_hub_download(
            repo_id=HF_REPO, filename=name, repo_type="dataset", local_dir=root
        )
        return True
    except Exception as e:  # noqa: BLE001 — any failure falls through to Drive
        print(f"HF mirror failed for {name}: {e}")
        return False


def gdrive_download(name: str, file_id: str, dest: Path) -> bool:
    try:
        import gdown
    except ImportError:
        print("gdown not installed; skipping Google Drive fallback")
        return False
    try:
        gdown.download(id=file_id, output=str(dest), quiet=False)
    except Exception as e:  # noqa: BLE001
        print(f"gdown failed for {name}: {e}")
    return dest.exists()


def download(root: Path, val_only: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, file_id in FILES.items():
        if val_only and name not in VAL_ONLY:
            continue
        dest = root / name
        if dest.exists():
            print(f"skip (exists): {name}")
            continue
        print(f"downloading {name} ...")
        if hf_download(name, root) or gdrive_download(name, file_id, dest):
            continue
        sys.exit(
            f"Download failed for {name}: Hugging Face mirror ({HF_REPO}) "
            "and Google Drive both failed. Refresh the Drive file ID from "
            "https://www.crowdhuman.org/download.html"
        )


def extract(root: Path) -> None:
    images = root / "images"
    images.mkdir(exist_ok=True)
    for z in sorted(root.glob("CrowdHuman_*.zip")):
        print(f"extracting {z.name} ...")
        with zipfile.ZipFile(z) as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                # zips contain Images/<ID>.jpg — flatten into images/
                target = images / Path(member).name
                if target.exists():
                    continue
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
    print(f"images: {sum(1 for _ in images.glob('*.jpg'))}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--val-only", action="store_true",
                    help="Only val split (~2.5 GB) — enough for evaluation")
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()

    download(args.root, args.val_only)
    if not args.no_extract:
        extract(args.root)


if __name__ == "__main__":
    main()
