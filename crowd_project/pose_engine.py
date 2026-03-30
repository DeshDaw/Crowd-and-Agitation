"""
Pose Engine — per-person 17-keypoint extraction via Keypoint R-CNN (COCO).

Runs independently of the Detection Engine (separate model, separate inference).
Returns structured pose results in *original* image coordinates.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

import config
from model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class PoseResult:
    """Pose extraction result for a single person."""
    bbox: np.ndarray              # (4,) float32 xyxy (original coords)
    keypoints: np.ndarray         # (17, 2) float32 x,y (original coords)
    keypoint_scores: np.ndarray   # (17,) float32  (logits from model)


def _resize_for_inference(
    image: np.ndarray,
    max_width: int = config.MAX_INFERENCE_WIDTH,
) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    if w <= max_width:
        return image, 1.0
    scale = max_width / w
    new_w = max_width
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, 1.0 / scale


class PoseEngine:
    """
    Extracts 17 COCO keypoints per detected person using Keypoint R-CNN.

    The engine is decoupled from model loading via :class:`ModelRegistry`.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._predictor = registry.get("pose")

    def extract(
        self,
        image: np.ndarray,
        resize: bool = True,
    ) -> tuple[list[PoseResult], float]:
        """
        Run keypoint estimation on a single BGR image.

        Args:
            image: BGR image (H, W, 3).
            resize: Down-scale to MAX_INFERENCE_WIDTH before inference.

        Returns:
            (pose_results, inference_time_seconds)
        """
        if resize:
            img_in, inv_scale = _resize_for_inference(image)
        else:
            img_in, inv_scale = image, 1.0

        t0 = time.perf_counter()
        outputs = self._predictor(img_in)
        inference_time = time.perf_counter() - t0

        instances = outputs["instances"]
        results = self._parse(instances, inv_scale)
        return results, inference_time

    @staticmethod
    def _parse(instances, inv_scale: float) -> list[PoseResult]:
        """Convert Detectron2 instances to list[PoseResult]."""
        if len(instances) == 0:
            return []

        classes = instances.pred_classes.cpu().numpy()
        mask = classes == config.COCO_PERSON_CLASS_ID
        if not mask.any():
            return []

        boxes = instances.pred_boxes.tensor.cpu().numpy()[mask] * inv_scale
        # pred_keypoints: (N, 17, 3)  last dim = (x, y, logit_score)
        kps_all = instances.pred_keypoints.cpu().numpy()[mask]

        results: list[PoseResult] = []
        for i in range(len(boxes)):
            kps_xy = kps_all[i, :, :2] * inv_scale   # (17, 2)
            kps_score = kps_all[i, :, 2]              # (17,)
            results.append(PoseResult(
                bbox=boxes[i].astype(np.float32),
                keypoints=kps_xy.astype(np.float32),
                keypoint_scores=kps_score.astype(np.float32),
            ))
        return results
