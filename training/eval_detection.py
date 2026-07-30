"""
Evaluate detection quality on CrowdHuman val: fine-tuned vs baselines.

Reports per-class mAP@0.5 and mAP@0.5:0.95 via ultralytics' COCO-style
validator. Note: the CrowdHuman leaderboard metric MR^-2 (log-average miss
rate) needs the official crowdhuman toolkit; this script reports mAP and
per-image count MAE, which are sufficient for the Phase II ablation
(head-count vs body-count vs Phase I bbox-density).

Usage:
    python eval_detection.py --data path/to/crowdhuman/yolo/crowdhuman.yaml \
        --weights runs/crowdhuman/crowdhuman-ft/weights/best.pt yolo11s.pt

Each weights file is validated in turn; COCO-pretrained models (1 class
'person' at a different index) are handled by mapping their class 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(weights: str, data: Path, imgsz: int, device: str | None) -> dict:
    from ultralytics import YOLO

    model = YOLO(weights)
    is_crowdhuman_head = "head" in {str(n) for n in model.names.values()}

    metrics = model.val(
        data=str(data),
        imgsz=imgsz,
        device=device,
        classes=None if is_crowdhuman_head else [0],  # COCO models: person only
        verbose=False,
        plots=False,
    )
    out = {
        "weights": weights,
        "classes": dict(model.names),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
    }
    # per-class breakdown when available
    try:
        per_class = {
            model.names[c]: round(float(m), 4)
            for c, m in zip(metrics.box.ap_class_index, metrics.box.ap50)
        }
        out["mAP50_per_class"] = per_class
    except Exception:  # noqa: BLE001 — metrics API varies across versions
        pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=Path("eval_detection_results.json"))
    args = ap.parse_args()

    results = [validate(w, args.data, args.imgsz, args.device) for w in args.weights]

    print(f"\n{'weights':45s} {'mAP50':>8s} {'mAP50-95':>9s}")
    for r in results:
        print(f"{Path(r['weights']).name:45s} {r['mAP50']:8.4f} {r['mAP50-95']:9.4f}")
        for cls, v in r.get("mAP50_per_class", {}).items():
            print(f"  {cls:43s} {v:8.4f}")

    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
