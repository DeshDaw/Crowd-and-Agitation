"""
Fine-tune YOLO11 on CrowdHuman (person + head, 2 classes).

Designed to run on a Colab/Kaggle GPU; works on CPU only for tiny smoke
tests. Trains a plain DETECTION model — CrowdHuman has no keypoint labels,
so pose weights are NOT the right starting point here. The fine-tuned
detector complements (does not replace) yolo11*-pose in the pipeline:
counting/density from this model, pose-based motion from the pose model.

Usage:
    python train_crowdhuman.py --data path/to/crowdhuman/yolo/crowdhuman.yaml \
        --model yolo11s.pt --epochs 30 --imgsz 640 --batch 16

Output: runs/crowdhuman/<name>/weights/best.pt — copy it into
crowd_project/models/ and select it via YOLO_WEIGHTS / the dashboard
yolo_weights field.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True,
                    help="crowdhuman.yaml from convert_crowdhuman.py")
    ap.add_argument("--model", default="yolo11s.pt",
                    help="Base weights (yolo11n.pt for quick runs, 11s recommended)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None,
                    help="0 for first GPU, 'cpu' for CPU (default: auto)")
    ap.add_argument("--name", default="crowdhuman-ft")
    ap.add_argument("--project", default="runs/crowdhuman")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project=args.project,
        resume=args.resume,
        # CrowdHuman-appropriate settings: dense small objects
        mosaic=1.0,
        close_mosaic=5,
        patience=10,
    )
    print("best weights:", results.save_dir / "weights" / "best.pt")


if __name__ == "__main__":
    main()
