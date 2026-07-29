"""
Density Analyzer — per-frame crowd density metrics and dynamic classification.

Computes density_ratio = total_bbox_area / image_area for each frame, then
classifies frames as Low / Moderate / High Crowd after the full batch using
mean ± σ thresholds combined with absolute floors (configurable in config.py).

The absolute floors keep the labels honest: a purely batch-relative z-score
guarantees "High Crowd" outliers in a skewed calm batch and can never fire in
a uniformly packed one, so "High Crowd" additionally requires an absolute
minimum density and near-empty frames are always "Low Crowd".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import config
from .detection_engine import Detection

logger = logging.getLogger(__name__)


@dataclass
class DensityMetrics:
    """Per-frame density measurements."""
    people_count: int
    image_area: float
    total_bbox_area: float
    mean_bbox_area: float
    density_ratio: float
    classification: str = ""          # filled after batch analysis


class DensityAnalyzer:
    """
    Computes per-frame density and classifies the entire batch
    using dynamically computed thresholds with absolute floors.
    """

    def analyze_frame(
        self,
        detections: Sequence[Detection],
        image_shape: tuple[int, int],           # (H, W)
    ) -> DensityMetrics:
        """
        Compute density metrics for a single frame.

        Args:
            detections: Person detections for this frame.
            image_shape: (height, width) of the original image.

        Returns:
            DensityMetrics (classification is empty until batch classification).
        """
        h, w = image_shape
        image_area = float(h * w)

        if not detections:
            return DensityMetrics(
                people_count=0,
                image_area=image_area,
                total_bbox_area=0.0,
                mean_bbox_area=0.0,
                density_ratio=0.0,
            )

        areas = np.array([d.bbox_area for d in detections], dtype=np.float64)
        total_area = float(areas.sum())
        mean_area = float(areas.mean())
        density_ratio = total_area / image_area if image_area > 0 else 0.0

        return DensityMetrics(
            people_count=len(detections),
            image_area=image_area,
            total_bbox_area=total_area,
            mean_bbox_area=mean_area,
            density_ratio=density_ratio,
        )

    def classify_batch(
        self,
        metrics_list: Sequence[DensityMetrics],
        low_sigma: float | None = None,
        high_sigma: float | None = None,
        high_min_ratio: float | None = None,
        low_max_ratio: float | None = None,
    ) -> None:
        """
        In-place: set ``classification`` on every DensityMetrics in the batch.

        Thresholds::

            High Crowd     : density > mean + high_sigma * std
                             AND density > high_min_ratio (absolute floor)
            Low Crowd      : density < mean − low_sigma * std
                             OR  density < low_max_ratio (absolute ceiling)
            Moderate Crowd : otherwise

        Args:
            metrics_list: All per-frame DensityMetrics (mutated in place).
            low_sigma: σ multiplier for low threshold.
            high_sigma: σ multiplier for high threshold.
            high_min_ratio: absolute density floor for "High Crowd".
            low_max_ratio: absolute density ceiling that forces "Low Crowd".
        """
        if not metrics_list:
            return

        low_sigma = low_sigma if low_sigma is not None else config.DENSITY_LOW_SIGMA
        high_sigma = high_sigma if high_sigma is not None else config.DENSITY_HIGH_SIGMA
        high_min = (
            high_min_ratio if high_min_ratio is not None
            else config.DENSITY_HIGH_MIN_RATIO
        )
        low_max = (
            low_max_ratio if low_max_ratio is not None
            else config.DENSITY_LOW_MAX_RATIO
        )

        ratios = np.array([m.density_ratio for m in metrics_list])
        mu = float(ratios.mean())
        sigma = float(ratios.std()) if len(ratios) > 1 else 0.0

        low_thresh = mu - low_sigma * sigma
        high_thresh = mu + high_sigma * sigma

        for m in metrics_list:
            if m.density_ratio > high_thresh and m.density_ratio > high_min:
                m.classification = "High Crowd"
            elif m.density_ratio < low_thresh or m.density_ratio < low_max:
                m.classification = "Low Crowd"
            else:
                m.classification = "Moderate Crowd"

        logger.info(
            "Density classification -- mean=%.4f std=%.4f low<%.4f high>%.4f "
            "(abs floor %.2f, abs low %.2f)",
            mu, sigma, low_thresh, high_thresh, high_min, low_max,
        )
