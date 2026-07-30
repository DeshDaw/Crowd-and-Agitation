"""
Pydantic models for API request/response validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class RunState(str, Enum):
    """Processing states for a run."""

    CREATED = "created"
    UPLOADING = "uploading"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunConfig(BaseModel):
    """Configuration overrides for a processing run."""

    device: str = Field(default="cpu", description="Device for inference: cpu or cuda")
    detection_backend: str = Field(
        default="detectron2",
        pattern="^(detectron2|yolo)$",
        description="Detection backend: detectron2 (Phase I baseline) or yolo (Phase II)",
    )
    yolo_weights: str = Field(
        default="yolo11n-pose.pt",
        pattern=r"^[\w][\w.\-]*\.pt$",
        description="YOLO weights filename (resolved from crowd_project/models/); "
        "e.g. crowdhuman_yolo11s.pt for the fine-tuned head+body detector",
    )
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    pose_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_inference_width: int = Field(default=960, ge=320, le=2048)
    tracker_iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    tracker_max_lost: int = Field(default=5, ge=1, le=100)
    density_low_sigma: float = Field(default=0.5, ge=0.1, le=5.0)
    density_high_sigma: float = Field(default=1.5, ge=0.1, le=10.0)
    agitation_threshold_sigma: float = Field(default=2.0, ge=0.0, le=5.0)
    video_extract_fps: Optional[float] = Field(default=None, ge=1.0, le=60.0)
    save_annotated: bool = Field(default=True)
    save_heatmaps: bool = Field(default=True)
    generate_plots: bool = Field(default=True)
    save_database: bool = Field(default=True)
    video_file: Optional[str] = Field(
        default=None, description="Uploaded video filename, if this is a video run"
    )
    source_fps: Optional[float] = Field(
        default=None, ge=0.1, le=240.0,
        description="Frame rate of the source frames (for m/s speeds); "
        "auto-detected for video runs",
    )


class CalibrationData(BaseModel):
    """Homography calibration: 4 image points + real-world rectangle size."""

    image_points: list[list[float]] = Field(
        min_length=4, max_length=4,
        description="4 pixel points (TL, TR, BR, BL) of a known ground rectangle",
    )
    width_m: float = Field(gt=0, le=1000)
    height_m: float = Field(gt=0, le=1000)
    image_size: list[int] = Field(
        default_factory=lambda: [0, 0],
        description="[width, height] of the frame the points were clicked on",
    )


class RunCreateResponse(BaseModel):
    """Response after creating a run."""

    run_id: str
    status: RunState
    message: str


class ProgressInfo(BaseModel):
    """Current processing progress."""

    total_frames: int = 0
    processed_frames: int = 0
    current_frame: Optional[str] = None
    current_stage: Optional[str] = None
    message: Optional[str] = None
    eta_seconds: Optional[int] = None
    per_stage_timings: dict[str, float] = Field(default_factory=dict)


class RunStatus(BaseModel):
    """Full run status response."""

    run_id: str
    state: RunState
    progress: ProgressInfo
    config: RunConfig
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error_message: Optional[str] = None


class RunSummary(BaseModel):
    """Brief run info for listing."""

    run_id: str
    state: RunState
    total_frames: int = 0
    created_at: datetime
    finished_at: Optional[datetime] = None
    has_events: bool = False


class RunListResponse(BaseModel):
    """List of runs response."""

    runs: list[RunSummary]
    total: int


class FileInfo(BaseModel):
    """File metadata in a run."""

    name: str
    path: str
    type: str
    size_bytes: int
    modified_at: datetime


class FileListResponse(BaseModel):
    """List of files in a run output."""

    files: list[FileInfo]
    categories: dict[str, list[FileInfo]]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    device_available: str
    cuda_available: bool
    backends_available: list[str] = Field(default_factory=list)
    version: str = "1.0.0"


class EventTimelineResponse(BaseModel):
    """Escalation events timeline."""

    events: list[dict[str, Any]]
    total_events: int


class ProcessingStage(str, Enum):
    """Processing stages as written to progress.current_stage by the runner."""

    VIDEO_EXTRACTION = "video_extraction"
    INFERENCE = "inference"
    CLASSIFICATION = "classification"
    EVENT_DETECTION = "event_detection"
    SAVING_OUTPUTS = "saving_outputs"
