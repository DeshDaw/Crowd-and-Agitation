"""
Ground Metrics — metric crowd measurements via camera calibration.

With a :class:`CameraCalibration`, bbox bottom-centres project onto the
ground plane, giving per-frame:

- persons/m² inside the calibrated region and its Fruin Level of Service
  class (A–F, space-per-person thresholds from config),
- per-track ground speed (EMA-smoothed), aggregated to mean/std per frame —
  in m/s when the source frame rate is known, otherwise m/frame.

This complements (does not replace) the pixel-space bbox-area density ratio,
which stays as the Phase I baseline arm for the ablation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import config
from .calibration import CameraCalibration
from .detection_engine import Detection
from .tracker import Track

logger = logging.getLogger(__name__)


@dataclass
class GroundMetrics:
    """Per-frame metric measurements (None-fields when not computable)."""
    persons_in_region: int
    persons_per_m2: float
    space_per_person: float | None      # m² per person; None when region empty
    los_class: str
    mean_speed: float | None            # m/s (or m/frame when fps unknown)
    speed_std: float | None
    speed_unit: str                     # "m/s" | "m/frame"


def classify_fruin(space_per_person: float | None) -> str:
    """Fruin LOS from space per person (m²/p). Empty region = free flow (A)."""
    if space_per_person is None:
        return "A"
    for label, min_space in config.FRUIN_LOS_M2_PER_PERSON:
        if space_per_person >= min_space:
            return label
    return "F"


class GroundMetricsAnalyzer:
    """
    Projects tracked persons to the ground plane and accumulates per-track
    position history for speed estimation.

    Keeps its own bounded history keyed by track_id, so it works identically
    with the IoU tracker and the YOLO/ByteTrack backend without touching the
    shared Track dataclass.
    """

    def __init__(
        self,
        calibration: CameraCalibration,
        source_fps: float | None = None,
        ema_alpha: float | None = None,
        max_lost: int | None = None,
    ) -> None:
        self._calib = calibration
        self._fps = source_fps if source_fps and source_fps > 0 else None
        self._alpha = ema_alpha if ema_alpha is not None else config.GROUND_SPEED_EMA_ALPHA
        self._max_lost = max_lost if max_lost is not None else config.TRACKER_MAX_LOST
        # track_id -> (last_frame, last_ground_xy, ema_speed_per_frame)
        self._state: dict[int, tuple[int, np.ndarray, float | None]] = {}

    @property
    def speed_unit(self) -> str:
        return "m/s" if self._fps else "m/frame"

    def analyze_frame(
        self,
        detections: Sequence[Detection],
        tracks: Sequence[Track],
        frame_index: int,
    ) -> GroundMetrics:
        """
        Compute metric density and speeds for one frame.

        Density comes from raw detections — it needs positions only, and
        trackers issue no IDs during warm-up (or ever, for disjoint still
        images). Speeds need identity, so they come from tracks.

        Args:
            detections: Person detections for the current frame.
            tracks: Tracks matched in the current frame (any backend).
            frame_index: Current frame index.
        """
        # Ground positions from bbox bottom-centres (the feet, which touch
        # the ground plane — centroids would float with person height).
        if detections:
            feet = np.array(
                [[(d.bbox[0] + d.bbox[2]) / 2.0, d.bbox[3]] for d in detections],
                dtype=np.float32,
            )
            ground = self._calib.image_to_ground(feet)
            inside = self._calib.in_region(ground)
        else:
            ground = np.empty((0, 2), dtype=np.float32)
            inside = np.zeros((0,), dtype=bool)

        persons_in_region = int(inside.sum())
        area = self._calib.area_m2
        persons_per_m2 = persons_in_region / area if area > 0 else 0.0
        space = (area / persons_in_region) if persons_in_region > 0 else None
        los = classify_fruin(space)

        # -- speeds (track-identity based) -------------------------------
        if tracks:
            track_feet = np.array(
                [[(t.bbox[0] + t.bbox[2]) / 2.0, t.bbox[3]] for t in tracks],
                dtype=np.float32,
            )
            track_ground = self._calib.image_to_ground(track_feet)
        else:
            track_ground = np.empty((0, 2), dtype=np.float32)

        speeds_per_frame: list[float] = []
        for t, g in zip(tracks, track_ground):
            prev = self._state.get(t.track_id)
            ema: float | None = None
            if prev is not None:
                prev_frame, prev_g, prev_ema = prev
                gap = frame_index - prev_frame
                if 0 < gap <= self._max_lost:
                    inst = float(np.linalg.norm(g - prev_g)) / gap
                    ema = (
                        inst if prev_ema is None
                        else self._alpha * inst + (1.0 - self._alpha) * prev_ema
                    )
                    speeds_per_frame.append(ema)
            self._state[t.track_id] = (frame_index, g.copy(), ema)

        self._prune(frame_index)

        mean_speed = speed_std = None
        if speeds_per_frame:
            arr = np.array(speeds_per_frame)
            scale = self._fps if self._fps else 1.0
            mean_speed = float(arr.mean() * scale)
            speed_std = float(arr.std() * scale) if len(arr) > 1 else 0.0

        return GroundMetrics(
            persons_in_region=persons_in_region,
            persons_per_m2=round(persons_per_m2, 4),
            space_per_person=round(space, 4) if space is not None else None,
            los_class=los,
            mean_speed=round(mean_speed, 4) if mean_speed is not None else None,
            speed_std=round(speed_std, 4) if speed_std is not None else None,
            speed_unit=self.speed_unit,
        )

    def _prune(self, frame_index: int) -> None:
        stale = [
            tid for tid, (seen, _, _) in self._state.items()
            if frame_index - seen > self._max_lost
        ]
        for tid in stale:
            del self._state[tid]


def sanity_check_speed(mean_speed: float | None, unit: str) -> None:
    """Log a warning for physically implausible pedestrian speeds."""
    if mean_speed is None or unit != "m/s":
        return
    if mean_speed > 4.0 and math.isfinite(mean_speed):
        logger.warning(
            "Mean pedestrian speed %.2f m/s is implausible — check calibration "
            "dimensions and source fps", mean_speed,
        )
