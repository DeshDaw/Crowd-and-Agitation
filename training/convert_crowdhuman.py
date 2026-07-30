"""
Convert CrowdHuman .odgt annotations to YOLO format.

CrowdHuman (Shao et al., 2018) annotates each person with three boxes:
    fbox — full body (amodal, may extend past the image)
    vbox — visible body region
    hbox — head

Output: two YOLO classes
    0 = person (fbox by default, or vbox with --box vbox)
    1 = head   (hbox)

Boxes tagged "mask" (crowd/ignore regions) and boxes with extra.ignore == 1
are dropped — YOLO has no ignore-region concept; keeping them as negatives
is the standard CrowdHuman->YOLO practice.

Usage (after downloading with download_crowdhuman.py):
    python convert_crowdhuman.py --root path/to/crowdhuman --split train
    python convert_crowdhuman.py --root path/to/crowdhuman --split val

Expected layout under --root:
    annotation_train.odgt, annotation_val.odgt
    images/<ID>.jpg   (all train01/02/03 + val zips extracted together)

Produces:
    <root>/yolo/images/{train,val}/<ID>.jpg      (hard links or copies)
    <root>/yolo/labels/{train,val}/<ID>.txt
    <root>/yolo/crowdhuman.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None  # fall back to cv2

CLASS_PERSON = 0
CLASS_HEAD = 1


def _image_size(path: Path) -> tuple[int, int]:
    """(width, height) without decoding the full image when PIL is present."""
    if Image is not None:
        with Image.open(path) as im:
            return im.size
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Unreadable image: {path}")
    h, w = img.shape[:2]
    return w, h


def _xywh_to_yolo(
    box: list[float], img_w: int, img_h: int,
) -> tuple[float, float, float, float] | None:
    """
    CrowdHuman [x, y, w, h] (top-left, may extend past image bounds) ->
    YOLO normalized (cx, cy, w, h), clipped to the image. None if the
    clipped box is degenerate.
    """
    if img_w <= 0 or img_h <= 0:
        return None
    x, y, w, h = box
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(img_w), x + w)
    y2 = min(float(img_h), y + h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    return cx, cy, (x2 - x1) / img_w, (y2 - y1) / img_h


def convert_record(
    rec: dict,
    img_path: Path,
    body_key: str,
) -> list[str]:
    """One .odgt record -> YOLO label lines."""
    img_w, img_h = _image_size(img_path)
    lines: list[str] = []
    for gt in rec.get("gtboxes", []):
        if gt.get("tag") != "person":
            continue  # "mask" = crowd/ignore region
        extra = gt.get("extra", {})
        if extra.get("ignore", 0) == 1:
            continue

        body = gt.get(body_key)
        if body:
            y = _xywh_to_yolo(body, img_w, img_h)
            if y:
                lines.append(f"{CLASS_PERSON} {y[0]:.6f} {y[1]:.6f} {y[2]:.6f} {y[3]:.6f}")

        head = gt.get("hbox")
        head_extra = gt.get("head_attr", {})
        if head and head_extra.get("ignore", 0) != 1:
            y = _xywh_to_yolo(head, img_w, img_h)
            if y:
                lines.append(f"{CLASS_HEAD} {y[0]:.6f} {y[1]:.6f} {y[2]:.6f} {y[3]:.6f}")
    return lines


def _place_image(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists():
        return
    if copy:
        shutil.copyfile(src, dst)
        return
    try:
        os.link(src, dst)  # hard link: no extra disk for 19 GB of images
    except OSError:
        shutil.copyfile(src, dst)


def convert_split(
    root: Path,
    split: str,
    body_key: str,
    copy_images: bool,
    limit: int | None = None,
) -> tuple[int, int]:
    odgt = root / f"annotation_{split}.odgt"
    if not odgt.is_file():
        sys.exit(f"Annotation file not found: {odgt}")

    images_dir = root / "images"
    out_img = root / "yolo" / "images" / split
    out_lbl = root / "yolo" / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    n_images = 0
    n_boxes = 0
    n_missing = 0
    with open(odgt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            img_id = rec["ID"]
            # IDs like "273271,1a0d6000b9e1f5b7" map to files with ',' kept
            src = images_dir / f"{img_id}.jpg"
            if not src.is_file():
                n_missing += 1
                continue

            lines = convert_record(rec, src, body_key)
            (out_lbl / f"{img_id}.txt").write_text("\n".join(lines), encoding="utf-8")
            _place_image(src, out_img / f"{img_id}.jpg", copy_images)

            n_images += 1
            n_boxes += len(lines)
            if n_images % 1000 == 0:
                print(f"  {split}: {n_images} images, {n_boxes} boxes")
            if limit and n_images >= limit:
                break

    if n_missing:
        print(f"  WARNING: {n_missing} annotated images missing from {images_dir}")
    print(f"{split}: {n_images} images, {n_boxes} boxes")
    return n_images, n_boxes


def write_dataset_yaml(root: Path) -> Path:
    yaml_path = root / "yolo" / "crowdhuman.yaml"
    yaml_path.write_text(
        f"""# CrowdHuman in YOLO format (generated by convert_crowdhuman.py)
path: {(root / 'yolo').resolve().as_posix()}
train: images/train
val: images/val

names:
  0: person
  1: head
""",
        encoding="utf-8",
    )
    print(f"Dataset config: {yaml_path}")
    return yaml_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="CrowdHuman root (contains annotation_*.odgt and images/)")
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    ap.add_argument("--box", choices=["fbox", "vbox"], default="fbox",
                    help="Which body box becomes class 'person' (default fbox)")
    ap.add_argument("--copy-images", action="store_true",
                    help="Copy images instead of hard-linking")
    ap.add_argument("--limit", type=int, default=None,
                    help="Convert at most N images per split (debugging)")
    args = ap.parse_args()

    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        convert_split(args.root, split, args.box, args.copy_images, args.limit)
    write_dataset_yaml(args.root)


if __name__ == "__main__":
    main()
