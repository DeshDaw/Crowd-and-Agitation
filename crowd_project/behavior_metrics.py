"""
Behavior Metrics — crowd-physics instability signals from ground velocities.

Implements the Phase II physics layer on top of the Stage 3 calibration:

- **Crowd pressure** P(t) = ρ(t) · Var[v(t)] — local density times velocity
  variance. Helbing, Johansson & Al-Abideen (Phys. Rev. E 75, 046109, 2007)
  showed this quantity spikes before crowd crushes ("crowd turbulence").
- **Directional entropy** — Shannon entropy of the heading distribution,
  normalized to [0, 1]. Ordered flow → 0, chaotic motion → 1.
- **Acceleration events** — fraction of tracks whose ground velocity changed
  faster than a threshold between observations (sudden starts/stops/turns).

All quantities use ground-plane velocity *vectors* (m/s when the source
frame rate is known, m/frame otherwise), maintained per track with EMA
smoothing. Requires a camera calibration — without metric units, "pressure"
is not comparable across scenes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import config
from .calibration import CameraCalibration
from .tracker import Track

logger = logging.getLogger(__name__)


@dataclass
class BehaviorMetrics:
    """Per-frame crowd-physics measurements."""
    crowd_pressure: float           # ρ · Var[v]   (persons · m² / s² · m⁻⁴ … reported as-is)
    velocity_variance: float        # Var[v] = E[|v − v̄|²]  ((m/s)² or (m/frame)²)
    directional_entropy: float      # [0, 1], normalized Shannon entropy of headings
    accel_event_rate: float         # fraction of tracked persons with |Δv| > threshold
    tracked_persons: int            # tracks contributing velocity samples


class BehaviorAnalyzer:
    """
    Maintains per-track ground velocity vectors and derives per-frame
    physics signals. State is keyed by track_id and pruned with the same
    max-lost horizon as the tracker, so it works with both backends.
    """

    def __init__(
        self,
        calibration: CameraCalibration,
        source_fps: float | None = None,
        ema_alpha: float | None = None,
        max_lost: int | None = None,
        heading_bins: int | None = None,
        min_heading_speed: float | None = None,
        accel_threshold: float | None = None,
    ) -> None:
        self._calib = calibration
        self._fps = source_fps if source_fps and source_fps > 0 else None
        self._alpha = ema_alpha if ema_alpha is not None else config.GROUND_SPEED_EMA_ALPHA
        self._max_lost = max_lost if max_lost is not None else config.TRACKER_MAX_LOST
        self._bins = heading_bins if heading_bins is not None else config.HEADING_ENTROPY_BINS
        self._min_heading_speed = (
            min_heading_speed if min_heading_speed is not None
            else config.HEADING_MIN_SPEED
        )
        self._accel_thresh = (
            accel_threshold if accel_threshold is not None
            else config.ACCEL_EVENT_THRESHOLD
        )
        # track_id -> (last_frame, last_ground_pos, ema_velocity_vector)
        self._state: dict[int, tuple[int, np.ndarray, np.ndarray | None]] = {}

    def analyze_frame(
        self,
        tracks: Sequence[Track],
        frame_index: int,
        persons_per_m2: float,
    ) -> BehaviorMetrics:
        """
        Update velocity state with this frame's tracks and compute signals.

        Args:
            tracks: Tracks matched in the current frame.
            frame_index: Current frame index.
            persons_per_m2: Metric density from the ground-metrics stage.
        """
        scale = self._fps if self._fps else 1.0  # per-frame -> per-second

        velocities: list[np.ndarray] = []
        accel_events = 0
        accel_samples = 0

        if tracks:
            feet = np.array(
                [[(t.bbox[0] + t.bbox[2]) / 2.0, t.bbox[3]] for t in tracks],
                dtype=np.float32,
            )
            ground = self._calib.image_to_ground(feet)
        else:
            ground = np.empty((0, 2), dtype=np.float32)

        for t, g in zip(tracks, ground):
            prev = self._state.get(t.track_id)
            ema_v: np.ndarray | None = None
            if prev is not None:
                prev_frame, prev_pos, prev_v = prev
                gap = frame_index - prev_frame
                if 0 < gap <= self._max_lost:
                    inst_v = (g - prev_pos) / gap * scale  # velocity vector
                    ema_v = (
                        inst_v if prev_v is None
                        else self._alpha * inst_v + (1.0 - self._alpha) * prev_v
                    )
                    velocities.append(ema_v)
                    if prev_v is not None:
                        accel_samples += 1
                        # |Δv| per second: velocity change across the gap
                        dv = float(np.linalg.norm(inst_v - prev_v)) / gap * scale
                        if dv > self._accel_thresh:
                            accel_events += 1
            self._state[t.track_id] = (frame_index, g.copy(), ema_v)

        self._prune(frame_index)

        n = len(velocities)
        if n:
            v = np.array(velocities)                      # (n, 2)
            mean_v = v.mean(axis=0)
            var_v = float(np.mean(np.sum((v - mean_v) ** 2, axis=1)))
        else:
            var_v = 0.0

        pressure = float(persons_per_m2) * var_v
        entropy = self._directional_entropy(velocities)
        accel_rate = accel_events / accel_samples if accel_samples else 0.0

        return BehaviorMetrics(
            crowd_pressure=round(pressure, 6),
            velocity_variance=round(var_v, 6),
            directional_entropy=round(entropy, 4),
            accel_event_rate=round(accel_rate, 4),
            tracked_persons=n,
        )

    # ------------------------------------------------------------------

    def _directional_entropy(self, velocities: Sequence[np.ndarray]) -> float:
        """
        Normalized Shannon entropy of movement headings.

        Near-stationary tracks are excluded — their headings are jitter, not
        motion, and would saturate the entropy of a perfectly calm scene.
        """
        headings = [
            float(np.arctan2(v[1], v[0]))
            for v in velocities
            if float(np.linalg.norm(v)) >= self._min_heading_speed
        ]
        if len(headings) < 2:
            return 0.0
        hist, _ = np.histogram(headings, bins=self._bins, range=(-np.pi, np.pi))
        p = hist / hist.sum()
        p = p[p > 0]
        entropy = float(-(p * np.log(p)).sum())
        return min(1.0, entropy / float(np.log(self._bins)))

    def _prune(self, frame_index: int) -> None:
        stale = [
            tid for tid, (seen, _, _) in self._state.items()
            if frame_index - seen > self._max_lost
        ]
        for tid in stale:
            del self._state[tid]
