"""
IoU-based Tracker with Hungarian (optimal) matching.

Assigns persistent ``track_id`` to each person across frames.
Designed so the matching back-end can later be swapped to DeepSORT
without changing the public API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

import config
from detection_engine import Detection
from pose_engine import PoseResult

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Persistent track for a single person across frames."""
    track_id: int
    bbox: np.ndarray                            # latest xyxy
    centroid: tuple[float, float]
    last_seen_frame: int
    motion_history: list[float] = field(default_factory=list)
    keypoint_history: list[np.ndarray] = field(default_factory=list)
    _lost_count: int = 0


def _iou_matrix(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:
    """
    Compute IoU between two sets of xyxy boxes.

    Args:
        boxes_a: (M, 4).
        boxes_b: (N, 4).

    Returns:
        (M, N) IoU matrix.
    """
    xa1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
    ya1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    xa2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    ya2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])

    inter = np.maximum(0.0, xa2 - xa1) * np.maximum(0.0, ya2 - ya1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


class IoUTracker:
    """
    Frame-by-frame IoU tracker using the Hungarian algorithm.

    Public API::

        tracker = IoUTracker()
        for frame_idx, detections, poses in ...:
            tracked = tracker.update(detections, poses, frame_idx)
    """

    def __init__(
        self,
        iou_threshold: float = config.TRACKER_IOU_THRESHOLD,
        max_lost: int = config.TRACKER_MAX_LOST,
    ) -> None:
        self._iou_thresh = iou_threshold
        self._max_lost = max_lost
        self._tracks: dict[int, Track] = {}
        self._next_id: int = 1

    @property
    def active_tracks(self) -> dict[int, Track]:
        return dict(self._tracks)

    def update(
        self,
        detections: Sequence[Detection],
        poses: Sequence[PoseResult],
        frame_index: int,
    ) -> list[Track]:
        """
        Match current detections to existing tracks and update state.

        Also associates the closest pose with each track using bbox IoU.

        Args:
            detections: Person detections for the current frame.
            poses: Pose results for the current frame.
            frame_index: Sequential frame counter.

        Returns:
            List of active tracks after update.
        """
        if not detections:
            self._age_all(frame_index)
            return list(self._tracks.values())

        det_boxes = np.array([d.bbox for d in detections], dtype=np.float32)
        det_centroids = [d.centroid for d in detections]

        # ---- match detections to existing tracks --------------------------
        matched_det: set[int] = set()
        matched_trk: set[int] = set()

        if self._tracks:
            trk_ids = list(self._tracks.keys())
            trk_boxes = np.array(
                [self._tracks[tid].bbox for tid in trk_ids], dtype=np.float32,
            )
            iou = _iou_matrix(det_boxes, trk_boxes)

            # Hungarian on *cost* (1 - IoU)
            cost = 1.0 - iou
            row_idx, col_idx = linear_sum_assignment(cost)

            for r, c in zip(row_idx, col_idx):
                if iou[r, c] >= self._iou_thresh:
                    tid = trk_ids[c]
                    self._tracks[tid].bbox = det_boxes[r]
                    self._tracks[tid].centroid = det_centroids[r]
                    self._tracks[tid].last_seen_frame = frame_index
                    self._tracks[tid]._lost_count = 0
                    matched_det.add(r)
                    matched_trk.add(tid)

        # ---- create tracks for unmatched detections -----------------------
        for i in range(len(detections)):
            if i not in matched_det:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = Track(
                    track_id=tid,
                    bbox=det_boxes[i],
                    centroid=det_centroids[i],
                    last_seen_frame=frame_index,
                )
                matched_trk.add(tid)

        # ---- associate poses with tracks ----------------------------------
        self._associate_poses(poses)

        # ---- age unmatched tracks; drop stale ones ------------------------
        self._age_all(frame_index, skip=matched_trk)

        return list(self._tracks.values())

    # ------------------------------------------------------------------
    def _associate_poses(self, poses: Sequence[PoseResult]) -> None:
        """Match each pose to the best overlapping track by bbox IoU."""
        if not poses or not self._tracks:
            return

        pose_boxes = np.array([p.bbox for p in poses], dtype=np.float32)
        trk_ids = list(self._tracks.keys())
        trk_boxes = np.array(
            [self._tracks[tid].bbox for tid in trk_ids], dtype=np.float32,
        )
        iou = _iou_matrix(pose_boxes, trk_boxes)

        row_idx, col_idx = linear_sum_assignment(1.0 - iou)
        for r, c in zip(row_idx, col_idx):
            if iou[r, c] >= self._iou_thresh:
                tid = trk_ids[c]
                self._tracks[tid].keypoint_history.append(poses[r].keypoints)

    def _age_all(
        self, frame_index: int, skip: set[int] | None = None,
    ) -> None:
        """Increment lost counter for tracks not in *skip*; prune stale."""
        skip = skip or set()
        to_remove: list[int] = []
        for tid, trk in self._tracks.items():
            if tid in skip:
                continue
            trk._lost_count += 1
            if trk._lost_count > self._max_lost:
                to_remove.append(tid)
        for tid in to_remove:
            del self._tracks[tid]
