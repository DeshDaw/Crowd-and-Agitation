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
# Detection backend
# =============================================================================
# "detectron2" — Phase I baseline: Faster R-CNN + Keypoint R-CNN + IoU tracker
# "yolo"       — Phase II: YOLO11-pose + ByteTrack/BoT-SORT, single pass
DETECTION_BACKEND: str = os.environ.get("CROWD_BACKEND", "detectron2")

# =============================================================================
# Models — Detectron2
# =============================================================================
DETECTOR_CONFIG: str = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
POSE_CONFIG: str = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"

# =============================================================================
# Models — YOLO (Ultralytics)
# =============================================================================
YOLO_WEIGHTS: str = os.environ.get("CROWD_YOLO_WEIGHTS", "yolo11n-pose.pt")
YOLO_CONF: float = 0.3
YOLO_IMGSZ: int = 640
YOLO_TRACKER: str = "bytetrack"  # "bytetrack" | "botsort"
# YOLO pose keypoint confidences are probabilities [0,1], unlike the
# R-CNN visibility logits gated by KEYPOINT_VISIBILITY_THRESH
YOLO_KEYPOINT_CONF: float = 0.5

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

# Absolute floors so classification is not purely batch-relative:
# "High Crowd" additionally requires density_ratio above this value, and any
# frame below the low ceiling is "Low Crowd" regardless of batch statistics.
DENSITY_HIGH_MIN_RATIO: float = 0.25
DENSITY_LOW_MAX_RATIO: float = 0.05

# =============================================================================
# Ground metrics (homography calibration)
# =============================================================================
# Fruin Level of Service thresholds, space per person in m²/p
# (Fruin, Pedestrian Planning & Design, 1971 — queueing/waiting areas).
# Class = first entry whose minimum space the frame meets; below all = F.
FRUIN_LOS_M2_PER_PERSON: list[tuple[str, float]] = [
    ("A", 1.2),
    ("B", 0.9),
    ("C", 0.7),
    ("D", 0.5),
    ("E", 0.2),
]

# EMA smoothing factor for per-track ground speed (1.0 = no smoothing)
GROUND_SPEED_EMA_ALPHA: float = 0.5

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

# Minimum number of visible keypoints (in both frames) required before a
# per-person motion score is emitted at all
MIN_VALID_KEYPOINTS: int = 6

# Bounded per-track history length (only the last two entries are ever read
# for motion; the rest exists for trend analysis)
TRACK_HISTORY_MAXLEN: int = 30

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
