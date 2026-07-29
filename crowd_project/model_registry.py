"""
Model Registry — lazy-loads and caches Detectron2 models.

Each inference engine requests its model from the registry by key.
Models are built on first access and cached for reuse.
Device abstraction is handled here so engines stay device-agnostic.

Score thresholds are resolved per registry instance (not at import time),
so per-run configuration overrides actually reach the models.
"""

import logging

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from . import config

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Singleton-style registry that lazily builds and caches Detectron2 predictors.

    Usage::

        registry = ModelRegistry(device="cpu")
        det_predictor = registry.get("detector")
        pose_predictor = registry.get("pose")
    """

    def __init__(
        self,
        device: str | None = None,
        confidence_threshold: float | None = None,
        pose_confidence_threshold: float | None = None,
    ) -> None:
        self._device: str = (device or config.DEVICE).lower()
        self._cache: dict[str, DefaultPredictor] = {}
        self._specs: dict[str, dict[str, object]] = {
            "detector": {
                "config_file": config.DETECTOR_CONFIG,
                "score_thresh": (
                    confidence_threshold
                    if confidence_threshold is not None
                    else config.CONFIDENCE_THRESHOLD
                ),
            },
            "pose": {
                "config_file": config.POSE_CONFIG,
                "score_thresh": (
                    pose_confidence_threshold
                    if pose_confidence_threshold is not None
                    else config.POSE_CONFIDENCE_THRESHOLD
                ),
            },
        }
        logger.info("ModelRegistry created (device=%s)", self._device)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def get(self, name: str) -> DefaultPredictor:
        """
        Return the cached predictor for *name*, building it on first call.

        Args:
            name: Key in the model specs ("detector", "pose", ...).

        Raises:
            KeyError: If *name* is not a registered model spec.
        """
        if name not in self._cache:
            self._cache[name] = self._build(name)
        return self._cache[name]

    @property
    def device(self) -> str:
        return self._device

    @property
    def loaded_models(self) -> list[str]:
        return list(self._cache.keys())

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------
    def _build(self, name: str) -> DefaultPredictor:
        if name not in self._specs:
            raise KeyError(
                f"Unknown model '{name}'. Registered: {list(self._specs)}"
            )
        spec = self._specs[name]
        logger.info("Building model '%s' ...", name)

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(spec["config_file"]))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(spec["config_file"])
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = spec["score_thresh"]
        cfg.MODEL.DEVICE = self._device

        predictor = DefaultPredictor(cfg)
        logger.info("Model '%s' ready.", name)
        return predictor
