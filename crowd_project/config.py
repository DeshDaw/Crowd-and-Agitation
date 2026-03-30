"""
Central configuration for the Crowd Detection and Density Estimation System.

All paths, model parameters, thresholds, and weights are defined here.
No hardcoded values in any other module. Designed for CPU-first with
seamless switch to CUDA (future RTX 5060 / government-grade GPU clusters).
"""

import os
from pathlib import Path
from typing import Optional

# =============================================================================
# Project paths
# =============================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent
INPUT_IMAGES_DIR: Path = PROJECT_ROOT / "input" / "images"
INPUT_VIDEO_DIR: Path = PROJECT_ROOT / "input" / "video"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
OUTPUT_ANNOTATED_DIR: Path = OUTPUT_DIR / "annotated"
OUTPUT_HEATMAPS_DIR: Path = OUTPUT_DIR / "heatmaps"
OUTPUT_ESCALATION_DIR: Path = OUTPUT_DIR / "escalation_frames"

# Output filenames
METRICS_JSON: str = "metrics.json"
SUMMARY_JSON: str = "summary.json"
EVENTS_JSON: str = "event_timeline.json"
CROWD_DENSITY_TREND_PNG: str = "crowd_density_trend.png"
AGITATION_TREND_PNG: str = "agitation_index_trend.png"
DATABASE_FILE: str = "crowd_analysis.db"

# =============================================================================
# Device
# =============================================================================
DEVICE: str = os.environ.get("CROWD_DEVICE", "cpu")

# =============================================================================
# Models — Detectron2
# =============================================================================
DETECTOR_CONFIG: str = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
POSE_CONFIG: str = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"

COCO_PERSON_CLASS_ID: int = 0

CONFIDENCE_THRESHOLD: float = 0.5
POSE_CONFIDENCE_THRESHOLD: float = 0.5

# =============================================================================
# Inference / preprocessing
# =============================================================================
MAX_INFERENCE_WIDTH: int = 960
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# =============================================================================
# Tracking (IoU-based, Hungarian matching)
# =============================================================================
TRACKER_IOU_THRESHOLD: float = 0.3
TRACKER_MAX_LOST: int = 5  # frames before track is dropped

# =============================================================================
# Density analysis
# =============================================================================
DENSITY_LOW_SIGMA: float = 0.5
DENSITY_HIGH_SIGMA: float = 1.5

# =============================================================================
# Heatmap (retained for annotated density overlays)
# =============================================================================
HEATMAP_DOWNSCALE: int = 4
HEATMAP_SIGMA: float = 2.0
HEATMAP_COLORMAP: int = 2  # cv2.COLORMAP_JET
HEATMAP_ALPHA: float = 0.5

# =============================================================================
# Motion analysis
# =============================================================================
# COCO keypoint indices used for torso length
KP_LEFT_SHOULDER: int = 5
KP_RIGHT_SHOULDER: int = 6
KP_LEFT_HIP: int = 11
KP_RIGHT_HIP: int = 12

# Minimum keypoint visibility score (logit) to trust a keypoint
KEYPOINT_VISIBILITY_THRESH: float = 2.0

# =============================================================================
# Agitation index weights  (must sum to 1.0)
# =============================================================================
AGITATION_W_MEAN_MOTION: float = 0.4
AGITATION_W_MOTION_VARIANCE: float = 0.3
AGITATION_W_DIRECTIONAL_VARIANCE: float = 0.2
AGITATION_W_DENSITY_CHANGE: float = 0.1

# Dynamic threshold: mean + AGITATION_THRESHOLD_SIGMA * std
AGITATION_THRESHOLD_SIGMA: float = 2.0

# =============================================================================
# Video extraction
# =============================================================================
VIDEO_EXTRACT_FPS: Optional[float] = None

# =============================================================================
# Analytics
# =============================================================================
MOVING_AVERAGE_WINDOW: int = 5

# =============================================================================
# Logging
# =============================================================================
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
