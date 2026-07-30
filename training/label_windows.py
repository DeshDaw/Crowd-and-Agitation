"""
Keyboard labeling tool for crowd-state training data.

Steps through the annotated frames of a processed run, window by window
(~1 s each), and records a Calm/Restless/Agitated label per window.

Keys:
    1 = Calm   2 = Restless   3 = Agitated
    SPACE = repeat heuristic pre-label   b = back one window   q = quit+save

Usage:
    python label_windows.py --run-output path/to/output --fps 5 \
        --out labels.csv

The run output must contain metrics.json and annotated/ (any backend).
Labels CSV columns: window_start_frame,label — the format
train_state_classifier.py consumes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crowd_project.state_classifier import HeuristicStateClassifier, STATES  # noqa: E402

KEYMAP = {ord("1"): "Calm", ord("2"): "Restless", ord("3"): "Agitated"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-output", type=Path, required=True,
                    help="Run output dir containing metrics.json and annotated/")
    ap.add_argument("--fps", type=float, default=5.0,
                    help="Frames per window second (window size)")
    ap.add_argument("--out", type=Path, default=Path("labels.csv"))
    args = ap.parse_args()

    metrics_path = args.run_output / "metrics.json"
    annotated_dir = args.run_output / "annotated"
    if not metrics_path.is_file():
        sys.exit(f"metrics.json not found in {args.run_output}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    window = max(1, round(args.fps))
    windows = [records[i:i + window] for i in range(0, len(records), window)]
    heur = HeuristicStateClassifier()

    labels: list[tuple[int, str]] = []
    i = 0
    win_name = "label: 1=Calm 2=Restless 3=Agitated  SPACE=accept  b=back  q=quit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while i < len(windows):
        chunk = windows[i]
        pre = heur.classify_frame(chunk[len(chunk) // 2])
        # show the middle frame of the window
        frame_rec = chunk[len(chunk) // 2]
        img_path = annotated_dir / frame_rec["frame_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            img = 255 * __import__("numpy").ones((360, 640, 3), dtype="uint8")
        cv2.putText(img, f"window {i + 1}/{len(windows)}  pre-label: {pre}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(win_name, img)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q") or key == 27:
            break
        if key == ord("b") and labels:
            labels.pop()
            i -= 1
            continue
        if key in KEYMAP:
            label = KEYMAP[key]
        elif key == ord(" "):
            label = pre
        else:
            continue
        labels.append((int(chunk[0].get("frame_index", i * window)), label))
        i += 1

    cv2.destroyAllWindows()

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["window_start_frame", "label"])
        w.writerows(labels)
    counts = {s: sum(1 for _, l in labels if l == s) for s in STATES}
    print(f"saved {len(labels)} labels to {args.out}  {counts}")


if __name__ == "__main__":
    main()
