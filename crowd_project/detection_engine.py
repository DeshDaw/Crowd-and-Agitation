"""
Detection Engine — person-only detection via Faster R-CNN (COCO).

Returns structured detections with bbox, confidence, area, and centroid.
All coordinates are in the *original* image space.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from . import config
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single person detection in original-image coordinates."""
    bbox: np.ndarray          # (4,) float32 xyxy
    confidence: float
    bbox_area: float
    centroid: tuple[float, float]


def _resize_for_inference(
    image: np.ndarray,
    max_width: int,
) -> tuple[np.ndarray, float]:
    """
    Down-scale *image* so width <= *max_width*.

    Returns:
        (resized_image, scale_factor)  where scale_factor converts
        resized coordinates → original coordinates (multiply).
    """
    h, w = image.shape[:2]
    if w <= max_width:
        return image, 1.0
    scale = max_width / w
    new_w = max_width
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, 1.0 / scale          # inverse: resized→original


class DetectionEngine:
    """
    Runs Faster R-CNN person detection and returns structured results.

    The engine fetches its predictor from :class:`ModelRegistry`
    so it is fully decoupled from model-loading logic.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        max_inference_width: int | None = None,
    ) -> None:
        self._predictor = registry.get("detector")
        self._max_width = (
            max_inference_width if max_inference_width is not None
            else config.MAX_INFERENCE_WIDTH
        )

    def detect(
        self,
        image: np.ndarray,
        resize: bool = True,
    ) -> tuple[list[Detection], float]:
        """
        Run person detection on a single BGR image.

        Args:
            image: BGR image (H, W, 3).
            resize: Down-scale to the configured max inference width first.

        Returns:
            (detections, inference_time_seconds)
        """
        if resize:
            img_in, inv_scale = _resize_for_inference(image, self._max_width)
        else:
            img_in, inv_scale = image, 1.0

        t0 = time.perf_counter()
        outputs = self._predictor(img_in)
        inference_time = time.perf_counter() - t0

        instances = outputs["instances"]
        detections = self._parse(instances, inv_scale)
        return detections, inference_time

    @staticmethod
    def draw_boxes(
        image: np.ndarray,
        detections: Sequence[Detection],
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Return a copy of *image* with bounding boxes drawn."""
        out = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            label = f"{det.confidence:.2f}"
            cv2.putText(
                out, label, (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _parse(instances, inv_scale: float) -> list[Detection]:
        """Filter to person class and build Detection list."""
        if len(instances) == 0:
            return []

        classes = instances.pred_classes.cpu().numpy()
        mask = classes == config.COCO_PERSON_CLASS_ID
        if not mask.any():
            return []

        boxes = instances.pred_boxes.tensor.cpu().numpy()[mask]
        scores = instances.scores.cpu().numpy()[mask]

        # scale back to original image coordinates
        boxes *= inv_scale

        detections: list[Detection] = []
        for i in range(len(boxes)):
            b = boxes[i].astype(np.float32)
            cx = float((b[0] + b[2]) / 2)
            cy = float((b[1] + b[3]) / 2)
            area = float((b[2] - b[0]) * (b[3] - b[1]))
            detections.append(Detection(
                bbox=b,
                confidence=float(scores[i]),
                bbox_area=area,
                centroid=(cx, cy),
            ))
        return detections
