"""
Agitation-onset evaluation — which signal detects abnormality first?

For a clip with a known abnormal-behavior onset (UMN clips label this; for
own footage note the frame by eye), compares detection signals:

- Phase I agitation_index (pose-motion weighted sum)
- crowd_pressure (Helbing ρ·Var[v])
- accel_event_rate
- fused crowd_state (Agitated == positive)

Reports per signal: ROC-AUC over frames (abnormal = frame >= onset) and
onset latency (first frame the signal crosses the threshold that yields a
5% false-positive rate on the normal prefix, minus the true onset).

Usage:
    python eval_agitation.py --metrics path/to/output/metrics.json \
        --onset-frame 240 [--out results.json]

This is the novelty-#2 protocol: micro (pose) vs macro (physics) vs fused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SIGNALS = ["agitation_index", "crowd_pressure", "accel_event_rate"]


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), no sklearn dependency."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[len(neg):].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def onset_latency(
    scores: np.ndarray, onset: int, fp_rate: float = 0.05,
) -> tuple[int | None, float]:
    """
    Threshold = (1 - fp_rate) quantile of the normal prefix; latency = first
    crossing at/after onset minus onset (frames). None if never crossed.
    """
    normal = scores[:onset]
    if len(normal) == 0:
        return None, float("nan")
    thresh = float(np.quantile(normal, 1.0 - fp_rate))
    after = np.nonzero(scores[onset:] > thresh)[0]
    return (int(after[0]) if len(after) else None), thresh


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--onset-frame", type=int, required=True,
                    help="First frame index of abnormal behavior (ground truth)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    with open(args.metrics, "r", encoding="utf-8") as f:
        records = json.load(f)

    n = len(records)
    onset = args.onset_frame
    if not 0 < onset < n:
        sys.exit(f"onset {onset} outside clip (0..{n - 1})")
    labels = np.array([1 if i >= onset else 0 for i in range(n)])

    results: dict[str, dict] = {}
    for sig in SIGNALS:
        scores = np.array(
            [float(r[sig]) if r.get(sig) is not None else 0.0 for r in records]
        )
        lat, thresh = onset_latency(scores, onset)
        results[sig] = {
            "auc": round(roc_auc(scores, labels), 4),
            "onset_latency_frames": lat,
            "threshold_at_5pct_fpr": round(thresh, 6),
        }

    # Fused discrete state
    states = np.array(
        [1.0 if r.get("crowd_state") == "Agitated" else 0.0 for r in records]
    )
    lat, _ = onset_latency(states, onset, fp_rate=0.0)
    results["crowd_state(fused)"] = {
        "auc": round(roc_auc(states, labels), 4),
        "onset_latency_frames": lat,
    }

    print(f"{'signal':24s} {'AUC':>8s} {'latency(frames)':>16s}")
    for sig, r in results.items():
        lat_s = str(r.get("onset_latency_frames"))
        print(f"{sig:24s} {r['auc']:8.4f} {lat_s:>16s}")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
