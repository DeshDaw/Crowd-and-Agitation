"""
Motion Analyzer — pose-based per-person motion scoring.

For each tracked person with ≥2 keypoint snapshots, computes the sum of
Euclidean keypoint displacements between the last two frames, normalized
by the torso length (shoulder-midpoint → hip-midpoint distance).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

import config
from tracker import Track

logger = logging.getLogger(__name__)


@dataclass
class PersonMotion:
    """Motion data for a single tracked person in the current frame."""
    track_id: int
    raw_motion: float               # sum of keypoint displacements (px)
    torso_length: float             # shoulder-mid → hip-mid (px)
    normalized_motion: float        # raw / torso  (dimensionless)
    velocity_vector: np.ndarray     # (2,) mean displacement vector


class MotionAnalyzer:
    """
    Computes per-person motion from keypoint history stored in Track objects.
    """

    def __init__(
        self,
        vis_thresh: float = config.KEYPOINT_VISIBILITY_THRESH,
    ) -> None:
        self._vis_thresh = vis_thresh

    def compute_person_motions(
        self,
        tracks: list[Track],
    ) -> list[PersonMotion]:
        """
        Compute motion for every track that has ≥2 keypoint frames.

        Args:
            tracks: Current active tracks with keypoint_history.

        Returns:
            List of PersonMotion for persons with enough history.
        """
        motions: list[PersonMotion] = []
        for trk in tracks:
            if len(trk.keypoint_history) < 2:
                continue
            kps_prev = trk.keypoint_history[-2]   # (17, 2)
            kps_curr = trk.keypoint_history[-1]

            displacements = kps_curr - kps_prev   # (17, 2)
            dists = np.linalg.norm(displacements, axis=1)  # (17,)
            raw_motion = float(dists.sum())

            torso = self._torso_length(kps_curr)
            norm_motion = raw_motion / torso if torso > 1.0 else raw_motion

            velocity = displacements.mean(axis=0)   # (2,) mean direction

            pm = PersonMotion(
                track_id=trk.track_id,
                raw_motion=raw_motion,
                torso_length=torso,
                normalized_motion=norm_motion,
                velocity_vector=velocity.astype(np.float32),
            )
            motions.append(pm)

            # store scalar in track's motion_history for trend analysis
            trk.motion_history.append(norm_motion)

        return motions

    @staticmethod
    def _torso_length(kps: np.ndarray) -> float:
        """
        Euclidean distance from shoulder midpoint to hip midpoint.

        Uses COCO indices from config.  Returns >0 or falls back to 1.0.
        """
        ls = kps[config.KP_LEFT_SHOULDER]
        rs = kps[config.KP_RIGHT_SHOULDER]
        lh = kps[config.KP_LEFT_HIP]
        rh = kps[config.KP_RIGHT_HIP]

        shoulder_mid = (ls + rs) / 2.0
        hip_mid = (lh + rh) / 2.0
        length = float(np.linalg.norm(shoulder_mid - hip_mid))
        return max(length, 1.0)
