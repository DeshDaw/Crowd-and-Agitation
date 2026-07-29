"""
Motion Analyzer — pose-based per-person motion scoring.

For each tracked person with ≥2 keypoint snapshots, computes the mean
Euclidean displacement of *visible* keypoints between the last two
observations, normalized by torso length (shoulder-mid → hip-mid) and by
the frame gap between the observations.

Keypoints whose visibility logit falls below the threshold in either frame
are excluded — Keypoint R-CNN emits all 17 joints even when occluded, and
those hallucinated joints jitter frame to frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from . import config
from .tracker import Track

logger = logging.getLogger(__name__)

_TORSO_INDICES = (
    config.KP_LEFT_SHOULDER,
    config.KP_RIGHT_SHOULDER,
    config.KP_LEFT_HIP,
    config.KP_RIGHT_HIP,
)


@dataclass
class PersonMotion:
    """Motion data for a single tracked person in the current frame."""
    track_id: int
    raw_motion: float               # mean displacement of visible keypoints (px/frame)
    torso_length: float             # shoulder-mid → hip-mid (px)
    normalized_motion: float        # raw / torso  (dimensionless, per frame)
    velocity_vector: np.ndarray     # (2,) mean displacement vector (px/frame)


class MotionAnalyzer:
    """
    Computes per-person motion from keypoint history stored in Track objects.
    """

    def __init__(
        self,
        vis_thresh: float | None = None,
        min_valid_keypoints: int | None = None,
    ) -> None:
        self._vis_thresh = (
            vis_thresh if vis_thresh is not None
            else config.KEYPOINT_VISIBILITY_THRESH
        )
        self._min_valid = (
            min_valid_keypoints if min_valid_keypoints is not None
            else config.MIN_VALID_KEYPOINTS
        )

    def compute_person_motions(
        self,
        tracks: list[Track],
        frame_index: int,
    ) -> list[PersonMotion]:
        """
        Compute motion for every track with a fresh pose this frame and at
        least one prior pose.

        A track whose latest snapshot is not from *frame_index* (pose missed
        this frame) emits nothing — re-scoring a stale displacement would
        inflate the frame's agitation, and a displacement measured across a
        gap is normalized by the gap length instead of being scored as
        single-frame motion.

        Args:
            tracks: Tracks matched in the current frame.
            frame_index: Current frame index.

        Returns:
            List of PersonMotion for persons with enough visible history.
        """
        motions: list[PersonMotion] = []
        for trk in tracks:
            if len(trk.keypoint_history) < 2:
                continue
            snap_prev = trk.keypoint_history[-2]
            snap_curr = trk.keypoint_history[-1]

            if snap_curr.frame_index != frame_index:
                continue                     # no fresh pose this frame
            gap = snap_curr.frame_index - snap_prev.frame_index
            if gap <= 0:
                continue

            valid = (
                (snap_curr.scores >= self._vis_thresh)
                & (snap_prev.scores >= self._vis_thresh)
            )
            if int(valid.sum()) < self._min_valid:
                continue

            displacements = (
                snap_curr.keypoints[valid] - snap_prev.keypoints[valid]
            ) / float(gap)
            dists = np.linalg.norm(displacements, axis=1)
            raw_motion = float(dists.mean())

            torso = self._torso_length(snap_curr.keypoints, snap_curr.scores)
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

    def _torso_length(self, kps: np.ndarray, scores: np.ndarray) -> float:
        """
        Euclidean distance from shoulder midpoint to hip midpoint.

        Requires all four torso keypoints to be visible; otherwise returns
        the 1.0 fallback so the motion stays un-normalized rather than being
        divided by a hallucinated torso.
        """
        idx = list(_TORSO_INDICES)
        if not bool((scores[idx] >= self._vis_thresh).all()):
            return 1.0

        ls, rs, lh, rh = (kps[i] for i in idx)
        shoulder_mid = (ls + rs) / 2.0
        hip_mid = (lh + rh) / 2.0
        length = float(np.linalg.norm(shoulder_mid - hip_mid))
        return max(length, 1.0)
