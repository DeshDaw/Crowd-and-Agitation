"""
End-to-end smoke test for the Stage 2 chain — no dataset download, CPU-only.

Builds a synthetic 4-image CrowdHuman-style .odgt dataset, runs the
converter, trains 1 epoch of yolo11n at 320 px, then loads the resulting
weights in YoloEngine and asserts the head-class path activates.

~2 minutes on CPU. Validates converter output format, dataset yaml,
training wiring, and engine head_count integration before spending any
Colab time on the real 19 GB run.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.convert_crowdhuman import convert_split, write_dataset_yaml  # noqa: E402


def build_synthetic_dataset(root: Path, n_images: int = 4) -> None:
    """Images with rectangle 'persons' + matching odgt annotations."""
    images = root / "images"
    images.mkdir(parents=True)
    rng = np.random.default_rng(0)

    for split in ("train", "val"):
        records = []
        for i in range(n_images):
            img_id = f"{split}_{i:03d}"
            img = rng.integers(0, 80, (480, 640, 3), dtype=np.uint8)
            gtboxes = []
            for _ in range(3):
                x = int(rng.integers(20, 480))
                y = int(rng.integers(40, 300))
                w, h = 60, 140
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 200), -1)
                head = [x + 15, y, 30, 30]
                cv2.rectangle(img, (head[0], head[1]),
                              (head[0] + 30, head[1] + 30), (200, 100, 0), -1)
                gtboxes.append({
                    "tag": "person",
                    "fbox": [x, y, w, h],
                    "vbox": [x, y, w, h - 20],
                    "hbox": head,
                    "extra": {},
                    "head_attr": {},
                })
            # one ignore region + one mask per image — converter must drop both
            gtboxes.append({"tag": "person", "fbox": [0, 0, 30, 30],
                            "hbox": [0, 0, 10, 10],
                            "extra": {"ignore": 1}, "head_attr": {}})
            gtboxes.append({"tag": "mask", "fbox": [600, 440, 40, 40], "extra": {}})

            cv2.imwrite(str(images / f"{img_id}.jpg"), img)
            records.append({"ID": img_id, "gtboxes": gtboxes})

        with open(root / f"annotation_{split}.odgt", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="crowdhuman_smoke_"))
    print("workspace:", tmp)
    try:
        build_synthetic_dataset(tmp)

        # converter
        for split in ("train", "val"):
            n_img, n_box = convert_split(tmp, split, "fbox", copy_images=True)
            assert n_img == 4, n_img
            # 4 images × (3 person boxes + 3 head boxes) = 24 total per split
            assert n_box == 4 * 3 * 2, f"expected 24 boxes, got {n_box}"
        yaml_path = write_dataset_yaml(tmp)

        label = (tmp / "yolo" / "labels" / "train" / "train_000.txt").read_text().splitlines()
        classes = sorted({line.split()[0] for line in label})
        assert classes == ["0", "1"], classes
        print("converter OK: person + head classes, ignore/mask dropped")

        # 1-epoch training
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")
        results = model.train(
            data=str(yaml_path), epochs=1, imgsz=320, batch=2, device="cpu",
            project=str(tmp / "runs"), name="smoke", verbose=False,
            workers=0, plots=False, val=False,
        )
        best = Path(results.save_dir) / "weights" / "best.pt"
        assert best.is_file(), best
        print("training OK:", best)

        # engine integration: head class must be auto-detected
        from crowd_project.yolo_engine import YoloEngine
        engine = YoloEngine(weights=str(best), device="cpu", conf=0.01, imgsz=320)
        assert engine._head_class is not None, "head class not detected in fine-tuned weights"
        img = cv2.imread(str(tmp / "yolo" / "images" / "val" / "val_000.jpg"))
        detections, tracks, dt, head_count = engine.process(img, 0)
        assert isinstance(head_count, int), type(head_count)
        print(f"engine OK: {len(detections)} persons, head_count={head_count}, "
              f"{dt*1000:.0f} ms")

        print("SMOKE-TEST-PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
