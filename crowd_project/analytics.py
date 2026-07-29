"""
Analytics — batch summary, JSON persistence, and trend plots.

Generates:
    summary.json            — aggregate statistics
    crowd_density_trend.png — density ratio over frames
    agitation_index_trend.png — agitation index over frames
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config

logger = logging.getLogger(__name__)


# =====================================================================
# Moving average utility
# =====================================================================

def moving_average(values: list[float], window: int) -> list[float]:
    """Centred simple moving average with boundary padding."""
    if not values:
        return []
    n = len(values)
    out: list[float] = []
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2 + 1)
        out.append(float(np.mean(values[lo:hi])))
    return out


# =====================================================================
# Summary construction
# =====================================================================

def build_summary(
    frame_records: list[dict[str, Any]],
    events_count: int,
    window: int = config.MOVING_AVERAGE_WINDOW,
) -> dict[str, Any]:
    """
    Compute batch summary from per-frame records.

    Each record is expected to have at least:
        frame_name, people_count, density_ratio, agitation_index, classification
    """
    if not frame_records:
        # Full shape, zero-filled — consumers (summary cards, JSON schema)
        # rely on every key being present even for an empty batch.
        return {
            "total_frames": 0,
            "mean_density": 0.0,
            "peak_density_frame": "",
            "peak_density_value": 0.0,
            "mean_agitation": 0.0,
            "highest_agitation_frame": "",
            "highest_agitation_value": 0.0,
            "total_escalation_events": events_count,
            "crowd_classification_distribution": {},
            "average_crowd_count": 0.0,
            "std_crowd_count": 0.0,
            "density_moving_average": [],
        }

    counts = [r["people_count"] for r in frame_records]
    densities = [r["density_ratio"] for r in frame_records]
    agitations = [r["agitation_index"] for r in frame_records]
    names = [r["frame_name"] for r in frame_records]
    classifications = [r["classification"] for r in frame_records]

    peak_density_idx = int(np.argmax(densities))
    peak_agitation_idx = int(np.argmax(agitations))

    class_dist: dict[str, int] = {}
    for c in classifications:
        class_dist[c] = class_dist.get(c, 0) + 1

    return {
        "total_frames": len(frame_records),
        "mean_density": round(float(np.mean(densities)), 6),
        "peak_density_frame": names[peak_density_idx],
        "peak_density_value": round(densities[peak_density_idx], 6),
        "mean_agitation": round(float(np.mean(agitations)), 6),
        "highest_agitation_frame": names[peak_agitation_idx],
        "highest_agitation_value": round(agitations[peak_agitation_idx], 6),
        "total_escalation_events": events_count,
        "crowd_classification_distribution": class_dist,
        "average_crowd_count": round(float(np.mean(counts)), 2),
        "std_crowd_count": round(float(np.std(counts)), 2) if len(counts) > 1 else 0.0,
        "density_moving_average": [round(v, 4) for v in moving_average(densities, window)],
    }


# =====================================================================
# JSON persistence
# =====================================================================

def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved JSON -> %s", path)


# =====================================================================
# Plotting
# =====================================================================

def plot_density_trend(
    frame_names: list[str],
    densities: list[float],
    classifications: list[str],
    output_path: Path,
    window: int = config.MOVING_AVERAGE_WINDOW,
) -> None:
    """Save crowd_density_trend.png."""
    n = len(densities)
    if n == 0:
        return
    x = np.arange(n)
    ma = moving_average(densities, window)

    color_map = {
        "Low Crowd": "green",
        "Moderate Crowd": "orange",
        "High Crowd": "red",
    }
    colors = [color_map.get(c, "gray") for c in classifications]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, densities, color=colors, alpha=0.5, label="Density ratio")
    ax.plot(x, ma, "-", color="navy", linewidth=2, label=f"Moving avg (k={window})")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Density ratio (bbox_area / image_area)")
    ax.set_title("Crowd Density Trend")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _set_xticks(ax, x, frame_names)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved density trend -> %s", output_path)


def plot_agitation_trend(
    frame_names: list[str],
    agitations: list[float],
    threshold: float,
    output_path: Path,
    window: int = config.MOVING_AVERAGE_WINDOW,
) -> None:
    """Save agitation_index_trend.png."""
    n = len(agitations)
    if n == 0:
        return
    x = np.arange(n)
    ma = moving_average(agitations, window)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, agitations, "o-", color="steelblue", alpha=0.6, label="Agitation index", markersize=4)
    ax.plot(x, ma, "-", color="darkorange", linewidth=2, label=f"Moving avg (k={window})")
    if threshold > 0:
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold:.4f})")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Agitation Index")
    ax.set_title("Crowd Agitation Index Trend (Abnormal Motion Detection)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _set_xticks(ax, x, frame_names)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved agitation trend -> %s", output_path)


def _set_xticks(ax, x: np.ndarray, names: list[str]) -> None:
    n = len(x)
    if n > 20:
        step = max(1, n // 15)
        ticks = list(range(0, n, step))
        if n - 1 not in ticks:
            ticks.append(n - 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([names[i] for i in ticks], rotation=45, ha="right")
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
