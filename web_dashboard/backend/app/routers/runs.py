"""
FastAPI routers for run management.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from ..models import (
    EventTimelineResponse,
    FileListResponse,
    ProgressInfo,
    RunConfig,
    RunCreateResponse,
    RunListResponse,
    RunState,
    RunStatus,
    RunSummary,
)
from ..services.runner import (
    cancel_run,
    cleanup_run,
    create_run,
    get_run_status,
    start_run,
)
from ..services.storage import (
    delete_run_workspace,
    discover_output_files,
    get_artifact_path,
    get_output_file_path,
    get_run_dir,
    get_run_output_dir,
    list_run_ids,
    load_status,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/runs", tags=["runs"])


def _serialize_status(status: dict) -> RunStatus:
    """Convert stored status dict to RunStatus model."""
    prog = status.get("progress", {})
    progress = ProgressInfo(
        total_frames=prog.get("total_frames", 0),
        processed_frames=prog.get("processed_frames", 0),
        current_frame=prog.get("current_frame"),
        current_stage=prog.get("current_stage"),
        eta_seconds=prog.get("eta_seconds"),
        per_stage_timings=prog.get("per_stage_timings", {}),
    )

    config = status.get("config", {})
    run_config = RunConfig(
        device=config.get("device", "cpu"),
        confidence_threshold=config.get("confidence_threshold", 0.5),
        pose_confidence_threshold=config.get("pose_confidence_threshold", 0.5),
        max_inference_width=config.get("max_inference_width", 960),
        tracker_iou_threshold=config.get("tracker_iou_threshold", 0.3),
        tracker_max_lost=config.get("tracker_max_lost", 5),
        density_low_sigma=config.get("density_low_sigma", 0.5),
        density_high_sigma=config.get("density_high_sigma", 1.5),
        agitation_threshold_sigma=config.get("agitation_threshold_sigma", 2.0),
        video_extract_fps=config.get("video_extract_fps"),
        save_annotated=config.get("save_annotated", True),
        save_heatmaps=config.get("save_heatmaps", True),
        generate_plots=config.get("generate_plots", True),
        save_database=config.get("save_database", True),
    )

    return RunStatus(
        run_id=status.get("run_id", ""),
        state=status.get("state", RunState.CREATED),
        progress=progress,
        config=run_config,
        created_at=datetime.fromisoformat(status.get("created_at", datetime.now().isoformat())),
        started_at=datetime.fromisoformat(status["started_at"]) if status.get("started_at") else None,
        finished_at=datetime.fromisoformat(status["finished_at"]) if status.get("finished_at") else None,
        error_message=status.get("error_message"),
    )


@router.post("", response_model=RunCreateResponse)
async def create_new_run(
    config: Annotated[str, Form()] = None,
) -> RunCreateResponse:
    """
    Create a new run. Configuration can be passed as JSON string.
    Upload files separately via /runs/{run_id}/upload.
    """
    run_id = str(uuid4())[:8]  # Short UUID for readability

    # Parse config if provided
    run_config = {}
    if config:
        try:
            run_config = json.loads(config)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON in config field",
            )

    # Create run
    create_run(run_id, run_config)

    return RunCreateResponse(
        run_id=run_id,
        status=RunState.CREATED,
        message=f"Run {run_id} created. Upload files to /api/runs/{run_id}/upload",
    )


@router.post("/{run_id}/upload")
async def upload_files(
    run_id: str,
    files: list[UploadFile] = File(None),
    video: UploadFile = File(None),
) -> dict:
    """
    Upload files to a run. Either multiple image files OR a single video.
    """
    # Check run exists
    run_dir = get_run_dir(run_id)
    if not run_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    uploaded = []

    # Handle multiple images
    if files:
        for upload_file in files:
            if upload_file.filename:
                file_path = input_dir / Path(upload_file.filename).name

                # Validate extension
                ext = file_path.suffix.lower()
                if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Unsupported image format: {ext}",
                    )

                with open(file_path, "wb") as f:
                    shutil.copyfileobj(upload_file.file, f)
                uploaded.append(upload_file.filename)

    # Handle single video
    if video and video.filename:
        file_path = input_dir / Path(video.filename).name

        # Validate extension
        ext = file_path.suffix.lower()
        if ext not in (".mp4", ".avi", ".mov", ".mkv"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported video format: {ext}",
            )

        with open(file_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        uploaded.append(video.filename)

        # Update config to indicate video processing
        from ..services.storage import update_status
        from ..services.runner import _active_runs
        
        current_status = load_status(run_id)
        if current_status:
            current_status["config"] = current_status.get("config", {})
            current_status["config"]["video_file"] = video.filename
            update_status(run_id, current_status)
            
            # Sync memory context so processing thread knows about the video
            if run_id in _active_runs:
                _active_runs[run_id].config["video_file"] = video.filename

    return {
        "run_id": run_id,
        "uploaded_files": uploaded,
        "total_files": len(uploaded),
        "message": f"Successfully uploaded {len(uploaded)} file(s)",
    }


@router.post("/{run_id}/start", response_model=dict)
async def start_processing_run(run_id: str) -> dict:
    """Start processing a run."""
    current_status = load_status(run_id)
    if not current_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    if current_status.get("state") not in ("created", "uploading"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Run is in state '{current_status.get('state')}', cannot start",
        )

    success = start_run(run_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start run",
        )

    return {
        "run_id": run_id,
        "status": "processing",
        "message": "Run started successfully",
    }


@router.get("/{run_id}/status", response_model=RunStatus)
async def get_status(run_id: str) -> RunStatus:
    """Get the current status of a run."""
    current_status = get_run_status(run_id)
    if not current_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    return _serialize_status(current_status)


@router.get("/{run_id}/summary")
async def get_summary(run_id: str) -> dict:
    """Get the summary.json for a completed run."""
    path = get_artifact_path(run_id, "summary")
    if not path or not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Summary not found. Run may not be completed.",
        )

    with open(path, "r") as f:
        return json.load(f)


@router.get("/{run_id}/metrics")
async def get_metrics(run_id: str) -> list[dict]:
    """Get the metrics.json for a completed run."""
    path = get_artifact_path(run_id, "metrics")
    if not path or not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics not found. Run may not be completed.",
        )

    with open(path, "r") as f:
        return json.load(f)


@router.get("/{run_id}/events", response_model=EventTimelineResponse)
async def get_events(run_id: str) -> EventTimelineResponse:
    """Get the event_timeline.json for a completed run."""
    path = get_artifact_path(run_id, "events")
    if not path or not path.exists():
        return EventTimelineResponse(events=[], total_events=0)

    with open(path, "r") as f:
        events = json.load(f)
        return EventTimelineResponse(events=events, total_events=len(events))


@router.get("/{run_id}/files", response_model=FileListResponse)
async def list_files(run_id: str) -> FileListResponse:
    """List all output files for a run."""
    if not get_run_dir(run_id).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    categories = discover_output_files(run_id)

    # Flatten for files list
    all_files = []
    for cat_files in categories.values():
        all_files.extend(cat_files)

    return FileListResponse(files=all_files, categories=categories)


@router.get("", response_model=RunListResponse)
async def list_runs() -> RunListResponse:
    """List all runs with summary info."""
    run_ids = list_run_ids()
    runs = []

    for run_id in run_ids:
        current_status = load_status(run_id)
        if current_status:
            progress = current_status.get("progress", {})

            # Check for events
            has_events = False
            events_path = get_artifact_path(run_id, "events")
            if events_path and events_path.exists():
                try:
                    with open(events_path) as f:
                        has_events = len(json.load(f)) > 0
                except:
                    pass

            runs.append(
                RunSummary(
                    run_id=run_id,
                    state=current_status.get("state", RunState.CREATED),
                    total_frames=progress.get("total_frames", 0),
                    created_at=datetime.fromisoformat(current_status.get("created_at", datetime.now().isoformat())),
                    finished_at=datetime.fromisoformat(current_status["finished_at"]) if current_status.get("finished_at") else None,
                    has_events=has_events,
                )
            )

    # Sort by created_at descending
    runs.sort(key=lambda r: r.created_at, reverse=True)

    return RunListResponse(runs=runs, total=len(runs))


@router.post("/{run_id}/cancel")
async def cancel_processing_run(run_id: str) -> dict:
    """Cancel a running or queued run."""
    success = cancel_run(run_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found or already completed",
        )

    return {"run_id": run_id, "status": "cancelled", "message": "Run cancellation requested"}


@router.delete("/{run_id}")
async def delete_run(run_id: str) -> dict:
    """Delete a run and all its data."""
    if not get_run_dir(run_id).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    cleanup_run(run_id)
    return {"run_id": run_id, "deleted": True, "message": "Run deleted successfully"}
