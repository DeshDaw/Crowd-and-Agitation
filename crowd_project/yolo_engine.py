"""
YOLO Engine — single-pass detection + tracking (+ pose) via Ultralytics.

Phase II backend: one YOLO11-pose inference per frame yields person boxes,
persistent ByteTrack/BoT-SORT track IDs, and 17 COCO keypoints — replacing
the Detectron2 path's two R-CNN passes plus IoU re-association.

Emits the exact same ``Detection`` and ``Track`` objects the rest of the
pipeline consumes (density, motion, agitation, heatmap, DB), so switching
backends changes nothing downstream.

Keypoint confidence note: YOLO pose confidences are probabilities in [0, 1],
unlike Keypoint R-CNN's logits — use ``YOLO_KEYPOINT_CONF`` (not
``KEYPOINT_VISIBILITY_THRESH``) as the motion-gating threshold.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .detection_engine import Detection
from .tracker import KeypointSnapshot, Track

logger = logging.getLogger(__name__)


def yolo_available() -> bool:
    """True if the ultralytics package is importable."""
    try:
        import ultralytics  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_weights(name: str) -> str:
    """
    Resolve a weights reference to a stable location.

    Bare model names (e.g. "yolo11n-pose.pt") are cached under
    ``crowd_project/models/`` so downloads do not land in whatever the
    process working directory happens to be.
    """
    p = Path(name)
    if p.is_file():
        return str(p)

    target = config.PROJECT_ROOT / "models" / p.name
    if target.is_file():
        return str(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(str(target))
        if target.is_file():
            return str(target)
    except Exception:  # noqa: BLE001 — fall through to ultralytics' own download
        logger.warning("Could not pre-download %s; deferring to ultralytics", name)
    return name


class YoloEngine:
    """
    Wraps ``model.track()`` and maintains Track objects with timestamped
    keypoint history, mirroring the IoUTracker contract:

    - ``process`` returns only tracks matched in the current frame,
    - internal tracks are pruned after ``max_lost`` unseen frames,
    - keypoint history entries are ``KeypointSnapshot`` with frame index
      and per-joint confidences.
    """

    def __init__(
        self,
        weights: str | None = None,
        device: str = "cpu",
        conf: float | None = None,
        imgsz: int | None = None,
        tracker_type: str | None = None,
        max_lost: int | None = None,
    ) -> None:
        from ultralytics import YOLO  # heavy import kept lazy

        self._weights = _resolve_weights(weights or config.YOLO_WEIGHTS)
        self._device = device
        self._conf = conf if conf is not None else config.YOLO_CONF
        self._imgsz = imgsz if imgsz is not None else config.YOLO_IMGSZ
        tracker = tracker_type or config.YOLO_TRACKER
        if not tracker.endswith(".yaml"):
            tracker = f"{tracker}.yaml"
        self._tracker_cfg = tracker
        self._max_lost = max_lost if max_lost is not None else config.TRACKER_MAX_LOST

        logger.info(
            "Loading YOLO weights=%s device=%s imgsz=%d tracker=%s",
            self._weights, device, self._imgsz, self._tracker_cfg,
        )
        self._model = YOLO(self._weights)

        # CrowdHuman-fine-tuned weights carry a "head" class: heads survive
        # occlusion that hides bodies, so head count is the robust count
        # signal in dense scenes. Auto-detected from the model's own names.
        names: dict[int, str] = {}
        for k, v in self._model.names.items():
            try:
                names[int(k)] = str(v)
            except (TypeError, ValueError):
                continue  # tolerate exotic custom-model name keys
        self._person_class = next(
            (i for i, n in names.items() if n == "person"), 0,
        )
        self._head_class = next(
            (i for i, n in names.items() if n == "head"), None,
        )
        self._classes = (
            [self._person_class]
            if self._head_class is None
            else [self._person_class, self._head_class]
        )
        if self._head_class is not None:
            logger.info("Head class detected (id=%d) — head_count enabled",
                        self._head_class)

        self._tracks: dict[int, Track] = {}
        self._last_seen: dict[int, int] = {}

    @property
    def active_tracks(self) -> dict[int, Track]:
        return dict(self._tracks)

    def process(
        self,
        image: np.ndarray,
        frame_index: int,
    ) -> tuple[list[Detection], list[Track], float, int | None]:
        """
        Run one tracked inference pass on a BGR frame.

        Returns:
            (person_detections, tracks_matched_this_frame,
             inference_time_seconds, head_count_or_None)

        head_count is only present with CrowdHuman-fine-tuned weights;
        person boxes drive detections/tracks/density, head boxes only the
        count.
        """
        t0 = time.perf_counter()
        results = self._model.track(
            image,
            persist=True,
            conf=self._conf,
            imgsz=self._imgsz,
            tracker=self._tracker_cfg,
            classes=self._classes,
            device=self._device,
            verbose=False,
        )
        inference_time = time.perf_counter() - t0

        r = results[0]
        detections: list[Detection] = []
        matched: list[Track] = []
        head_count = 0 if self._head_class is not None else None

        boxes = r.boxes
        n = len(boxes) if boxes is not None else 0
        if n:
            xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            ids = (
                boxes.id.cpu().numpy().astype(int)
                if boxes.id is not None else [None] * n
            )

            kps_xy = kps_conf = None
            if r.keypoints is not None and r.keypoints.xy is not None and len(r.keypoints) == n:
                kps_xy = r.keypoints.xy.cpu().numpy().astype(np.float32)
                conf_t = r.keypoints.conf
                kps_conf = (
                    conf_t.cpu().numpy().astype(np.float32)
                    if conf_t is not None
                    else np.ones((n, kps_xy.shape[1]), dtype=np.float32)
                )

            for i in range(n):
                if self._head_class is not None and clss[i] == self._head_class:
                    head_count += 1
                    continue
                if clss[i] != self._person_class:
                    continue

                b = xyxy[i]
                cx = float((b[0] + b[2]) / 2)
                cy = float((b[1] + b[3]) / 2)
                area = float((b[2] - b[0]) * (b[3] - b[1]))
                detections.append(Detection(
                    bbox=b,
                    confidence=float(confs[i]),
                    bbox_area=area,
                    centroid=(cx, cy),
                ))

                tid = ids[i]
                if tid is None:
                    continue  # tracker warm-up frames have no IDs yet

                trk = self._tracks.get(tid)
                if trk is None:
                    trk = Track(
                        track_id=int(tid),
                        bbox=b,
                        centroid=(cx, cy),
                        last_seen_frame=frame_index,
                    )
                    self._tracks[tid] = trk
                else:
                    trk.bbox = b
                    trk.centroid = (cx, cy)
                    trk.last_seen_frame = frame_index

                if kps_xy is not None:
                    trk.keypoint_history.append(KeypointSnapshot(
                        frame_index=frame_index,
                        keypoints=kps_xy[i],
                        scores=kps_conf[i],
                    ))

                self._last_seen[tid] = frame_index
                matched.append(trk)

        self._prune(frame_index)
        return detections, matched, inference_time, head_count

    def _prune(self, frame_index: int) -> None:
        stale = [
            tid for tid, seen in self._last_seen.items()
            if frame_index - seen > self._max_lost
        ]
        for tid in stale:
            self._tracks.pop(tid, None)
            self._last_seen.pop(tid, None)

    def info(self) -> dict[str, Any]:
        return {
            "weights": str(self._weights),
            "device": self._device,
            "imgsz": self._imgsz,
            "tracker": self._tracker_cfg,
            "conf": self._conf,
        }
