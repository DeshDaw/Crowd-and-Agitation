"""
Agitation Analyzer — Abnormal Crowd Motion Detection.

Combines per-frame mean motion, motion variance, directional variance, and
density-change rate into a single scalar Agitation Index.  After the full
batch, dynamic thresholds (mean + k·σ) flag frames with abnormal motion.

Weights are configurable in config.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

import config
from motion_analyzer import PersonMotion

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
        w_mean: float = config.AGITATION_W_MEAN_MOTION,
        w_var: float = config.AGITATION_W_MOTION_VARIANCE,
        w_dir: float = config.AGITATION_W_DIRECTIONAL_VARIANCE,
        w_density: float = config.AGITATION_W_DENSITY_CHANGE,
    ) -> None:
        self._w = (w_mean, w_var, w_dir, w_density)

    def compute_frame(
        self,
        motions: Sequence[PersonMotion],
        current_density: float,
        previous_density: float,
    ) -> AgitationMetrics:
        """
        Compute Agitation Index for a single frame.

        Args:
            motions: PersonMotion list for this frame (may be empty).
            current_density: density_ratio for this frame.
            previous_density: density_ratio for the preceding frame (0 for first).

        Returns:
            AgitationMetrics.
        """
        if not motions:
            dcr = abs(current_density - previous_density)
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

        dcr = abs(current_density - previous_density)

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
        sigma_mult: float = config.AGITATION_THRESHOLD_SIGMA,
    ) -> tuple[float, float, float]:
        """
        After full batch, compute mean, std, and threshold for agitation.

        Returns:
            (mean_agitation, std_agitation, threshold)
        """
        if not agitation_list:
            return 0.0, 0.0, 0.0
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
