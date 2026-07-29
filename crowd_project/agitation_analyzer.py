"""
Agitation Analyzer — Abnormal Crowd Motion Detection.

Combines per-frame mean motion, motion variance, directional variance, and
density-change rate into a single scalar Agitation Index.  After the full
batch, dynamic thresholds (mean + k·σ) flag frames with abnormal motion.

Weights default from config.py and can be overridden per run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import config
from .motion_analyzer import PersonMotion

logger = logging.getLogger(__name__)


@dataclass
class AgitationMetrics:
    """Per-frame agitation measurements."""
    mean_motion: float
    motion_variance: float
    directional_variance: float
    density_change_rate: float
    agitation_index: float


class AgitationAnalyzer:
    """
    Computes per-frame Agitation Index and batch-level thresholds.
    """

    def __init__(
        self,
        w_mean: float | None = None,
        w_var: float | None = None,
        w_dir: float | None = None,
        w_density: float | None = None,
        threshold_sigma: float | None = None,
    ) -> None:
        self._w = (
            w_mean if w_mean is not None else config.AGITATION_W_MEAN_MOTION,
            w_var if w_var is not None else config.AGITATION_W_MOTION_VARIANCE,
            w_dir if w_dir is not None else config.AGITATION_W_DIRECTIONAL_VARIANCE,
            w_density if w_density is not None else config.AGITATION_W_DENSITY_CHANGE,
        )
        self._threshold_sigma = (
            threshold_sigma if threshold_sigma is not None
            else config.AGITATION_THRESHOLD_SIGMA
        )

    def compute_frame(
        self,
        motions: Sequence[PersonMotion],
        current_density: float,
        previous_density: float | None,
    ) -> AgitationMetrics:
        """
        Compute Agitation Index for a single frame.

        Args:
            motions: PersonMotion list for this frame (may be empty).
            current_density: density_ratio for this frame.
            previous_density: density_ratio for the preceding frame, or None
                on the first frame (the density-change term is zeroed rather
                than scored against a fictitious empty scene).

        Returns:
            AgitationMetrics.
        """
        dcr = (
            abs(current_density - previous_density)
            if previous_density is not None else 0.0
        )

        if not motions:
            return AgitationMetrics(
                mean_motion=0.0,
                motion_variance=0.0,
                directional_variance=0.0,
                density_change_rate=dcr,
                agitation_index=self._w[3] * dcr,
            )

        norms = np.array([m.normalized_motion for m in motions])
        mean_motion = float(norms.mean())
        motion_var = float(norms.var()) if len(norms) > 1 else 0.0

        # Directional variance: angular variance of velocity vectors
        dir_var = self._directional_variance(motions)

        ai = (
            self._w[0] * mean_motion
            + self._w[1] * motion_var
            + self._w[2] * dir_var
            + self._w[3] * dcr
        )

        return AgitationMetrics(
            mean_motion=mean_motion,
            motion_variance=motion_var,
            directional_variance=dir_var,
            density_change_rate=dcr,
            agitation_index=ai,
        )

    def compute_batch_threshold(
        self,
        agitation_list: Sequence[AgitationMetrics],
        sigma_mult: float | None = None,
    ) -> tuple[float, float, float]:
        """
        After full batch, compute mean, std, and threshold for agitation.

        Note: this threshold is batch-relative (self-referential) — it flags
        the batch's own statistical outliers, not absolute agitation. Suitable
        for offline review; a live deployment needs a rolling/absolute
        threshold instead.

        Returns:
            (mean_agitation, std_agitation, threshold)
        """
        if not agitation_list:
            return 0.0, 0.0, 0.0
        if sigma_mult is None:
            sigma_mult = self._threshold_sigma
        vals = np.array([a.agitation_index for a in agitation_list])
        mu = float(vals.mean())
        sigma = float(vals.std()) if len(vals) > 1 else 0.0
        threshold = mu + sigma_mult * sigma
        logger.info(
            "Agitation threshold -- mean=%.4f std=%.4f threshold=%.4f",
            mu, sigma, threshold,
        )
        return mu, sigma, threshold

    @staticmethod
    def _directional_variance(motions: Sequence[PersonMotion]) -> float:
        """
        Compute angular variance of velocity vectors.

        Uses circular variance: 1 − |mean(unit_vectors)|.
        Returns 0 when all move the same direction, ~1 when chaotic.
        """
        vecs = np.array([m.velocity_vector for m in motions])
        magnitudes = np.linalg.norm(vecs, axis=1, keepdims=True)
        # avoid divide-by-zero for stationary persons
        safe_mag = np.where(magnitudes > 1e-6, magnitudes, 1.0)
        unit = vecs / safe_mag
        mean_unit = unit.mean(axis=0)
        return float(1.0 - np.linalg.norm(mean_unit))
