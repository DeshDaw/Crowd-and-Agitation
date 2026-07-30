"""
Train the Calm/Restless/Agitated MLP from labeled runs.

Inputs are pairs of (metrics.json, labels.csv) — one pair per labeled clip.
labels.csv comes from label_windows.py (window_start_frame,label). Features
are the windowed mean+max vectors defined in crowd_project.state_classifier,
so pipeline inference and training can never drift apart.

CPU-friendly: the model is tiny (≈1.3k params); minutes, not hours.

Usage:
    python train_state_classifier.py \
        --data clipA_output/metrics.json clipA_labels.csv \
        --data clipB_output/metrics.json clipB_labels.csv \
        --fps 5 --epochs 200 \
        --out ../crowd_project/models/state_classifier.pt

Evaluation: with >=2 clips, leave-one-clip-out cross-validation is reported
(the honest protocol — random window splits leak temporal correlation);
the final model trains on everything.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crowd_project.state_classifier import (  # noqa: E402
    STATES,
    build_mlp,
    save_checkpoint,
    window_features,
)


def load_clip(metrics_path: Path, labels_path: Path, window: int):
    with open(metrics_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    feats, starts = window_features(records, window)

    labels_by_start: dict[int, str] = {}
    with open(labels_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            labels_by_start[int(row["window_start_frame"])] = row["label"]

    x_rows, y_rows = [], []
    for feat, start in zip(feats, starts):
        label = labels_by_start.get(start)
        if label in STATES:
            x_rows.append(feat)
            y_rows.append(STATES.index(label))
    if not x_rows:
        raise SystemExit(f"No usable labels for {metrics_path} (start-frame mismatch?)")
    return np.stack(x_rows), np.array(y_rows, dtype=np.int64)


def train_once(x_train, y_train, epochs: int, seed: int = 0):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    xn = torch.from_numpy(((x_train - mean) / std).astype(np.float32))
    y = torch.from_numpy(y_train)

    net = build_mlp(x_train.shape[1])
    # class weights for the inevitable Calm-heavy imbalance
    counts = np.bincount(y_train, minlength=len(STATES)).astype(np.float32)
    weights = torch.from_numpy(counts.sum() / np.maximum(counts, 1) / len(STATES))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(xn), y)
        loss.backward()
        opt.step()
    net.eval()
    return net, mean, std


def evaluate(net, mean, std, x, y) -> tuple[float, np.ndarray]:
    import torch

    xn = torch.from_numpy(((x - mean) / std).astype(np.float32))
    with torch.no_grad():
        pred = net(xn).argmax(dim=1).numpy()
    acc = float((pred == y).mean())
    cm = np.zeros((len(STATES), len(STATES)), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return acc, cm


def macro_f1(cm: np.ndarray) -> float:
    f1s = []
    for c in range(len(cm)):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", nargs=2, action="append", required=True,
                    metavar=("METRICS_JSON", "LABELS_CSV"),
                    help="Repeatable: one labeled clip")
    ap.add_argument("--fps", type=float, default=5.0,
                    help="Frames per window (must match label_windows.py --fps)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "crowd_project" / "models" / "state_classifier.pt")
    args = ap.parse_args()

    window = max(1, round(args.fps))
    clips = [load_clip(Path(m), Path(l), window) for m, l in args.data]

    # Leave-one-clip-out CV (only honest with >=2 clips)
    loov: list[float] = []
    if len(clips) >= 2:
        for held in range(len(clips)):
            x_tr = np.concatenate([c[0] for i, c in enumerate(clips) if i != held])
            y_tr = np.concatenate([c[1] for i, c in enumerate(clips) if i != held])
            net, mean, std = train_once(x_tr, y_tr, args.epochs)
            acc, cm = evaluate(net, mean, std, *clips[held])
            f1 = macro_f1(cm)
            loov.append(f1)
            print(f"LOOV fold {held}: acc={acc:.3f} macroF1={f1:.3f}")
        print(f"LOOV macro-F1 mean: {np.mean(loov):.3f}")
    else:
        print("Single clip — skipping cross-validation (metrics would leak)")

    # Final model on everything
    x_all = np.concatenate([c[0] for c in clips])
    y_all = np.concatenate([c[1] for c in clips])
    net, mean, std = train_once(x_all, y_all, args.epochs)
    acc, cm = evaluate(net, mean, std, x_all, y_all)
    print(f"train-set acc={acc:.3f} (optimistic; trust LOOV)  confusion:\n{cm}")

    save_checkpoint(
        args.out, net, mean, std, window,
        extra={
            "train_windows": int(len(y_all)),
            "loov_macro_f1": float(np.mean(loov)) if loov else None,
            "class_counts": {s: int((y_all == i).sum()) for i, s in enumerate(STATES)},
        },
    )
    print(f"saved: {args.out} (+ .json sidecar). The pipeline auto-loads it.")


if __name__ == "__main__":
    main()
