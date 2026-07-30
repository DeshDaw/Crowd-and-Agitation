"""
Per-run pipeline settings.

Every tunable knob the pipeline consumes is carried in a single
:class:`PipelineSettings` instance built at run construction time.
``config.py`` remains the source of *defaults* only — no component reads
module globals after import, so callers (CLI or dashboard) can override any
value per run without mutating shared state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from . import config


@dataclass
class PipelineSettings:
    """All tunable parameters for one pipeline run."""

    # Device / models
    device: str = config.DEVICE
    detection_backend: str = config.DETECTION_BACKEND
    confidence_threshold: float = config.CONFIDENCE_THRESHOLD
    pose_confidence_threshold: float = config.POSE_CONFIDENCE_THRESHOLD
    max_inference_width: int = config.MAX_INFERENCE_WIDTH

    # YOLO backend
    yolo_weights: str = config.YOLO_WEIGHTS
    yolo_conf: float = config.YOLO_CONF
    yolo_imgsz: int = config.YOLO_IMGSZ
    yolo_tracker: str = config.YOLO_TRACKER
    yolo_keypoint_conf: float = config.YOLO_KEYPOINT_CONF

    # Tracking
    tracker_iou_threshold: float = config.TRACKER_IOU_THRESHOLD
    tracker_max_lost: int = config.TRACKER_MAX_LOST

    # Density
    density_low_sigma: float = config.DENSITY_LOW_SIGMA
    density_high_sigma: float = config.DENSITY_HIGH_SIGMA
    density_high_min_ratio: float = config.DENSITY_HIGH_MIN_RATIO
    density_low_max_ratio: float = config.DENSITY_LOW_MAX_RATIO

    # Motion / agitation
    keypoint_visibility_thresh: float = config.KEYPOINT_VISIBILITY_THRESH
    min_valid_keypoints: int = config.MIN_VALID_KEYPOINTS
    agitation_w_mean_motion: float = config.AGITATION_W_MEAN_MOTION
    agitation_w_motion_variance: float = config.AGITATION_W_MOTION_VARIANCE
    agitation_w_directional_variance: float = config.AGITATION_W_DIRECTIONAL_VARIANCE
    agitation_w_density_change: float = config.AGITATION_W_DENSITY_CHANGE
    agitation_threshold_sigma: float = config.AGITATION_THRESHOLD_SIGMA

    # Heatmap
    heatmap_downscale: int = config.HEATMAP_DOWNSCALE
    heatmap_sigma: float = config.HEATMAP_SIGMA
    heatmap_colormap: int = config.HEATMAP_COLORMAP
    heatmap_alpha: float = config.HEATMAP_ALPHA

    # Ground metrics (optional; enables persons/m², Fruin LOS, m/s speeds)
    calibration_file: Path | None = None
    source_fps: float | None = None      # frame rate of the source; speeds
                                         # are m/frame when unknown

    # Output
    output_dir: Path = field(default_factory=lambda: config.OUTPUT_DIR)
    save_annotated: bool = True
    save_heatmaps: bool = True

    @property
    def annotated_dir(self) -> Path:
        return self.output_dir / "annotated"

    @property
    def heatmaps_dir(self) -> Path:
        return self.output_dir / "heatmaps"

    @property
    def escalation_dir(self) -> Path:
        return self.output_dir / "escalation_frames"

    @classmethod
    def from_overrides(cls, overrides: dict[str, Any]) -> "PipelineSettings":
        """Build settings from a dict, ignoring unknown keys and None values."""
        known = {f.name for f in fields(cls)}
        kwargs = {
            k: v for k, v in overrides.items()
            if k in known and v is not None
        }
        if "output_dir" in kwargs:
            kwargs["output_dir"] = Path(kwargs["output_dir"])
        if "calibration_file" in kwargs:
            kwargs["calibration_file"] = Path(kwargs["calibration_file"])
        return cls(**kwargs)
