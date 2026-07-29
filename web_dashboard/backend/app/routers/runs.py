"""
FastAPI routers for run management.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional
from uuid import uuid4

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile, status
from fastapi import Path as PathParam
from pydantic import ValidationError

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
    RUN_ID_PATTERN,
    discover_output_files,
    get_artifact_path,
    get_run_dir,
    list_run_ids,
    load_status,
    update_status,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/runs", tags=["runs"])

# Path-parameter validation: run ids are exactly 8 hex chars; anything else
# (including traversal attempts) is rejected with 422 before touching disk.
RunID = Annotated[str, PathParam(pattern=RUN_ID_PATTERN)]


def _tolerant_run_config(config: dict) -> RunConfig:
    """Build a RunConfig from stored data, falling back to defaults on bad values."""
    try:
        return RunConfig.model_validate({k: v for k, v in config.items() if v is not None})
    except ValidationError:
        logger.warning("Stored run config invalid, serving defaults: %s", config)
        return RunConfig()


def _serialize_status(status_data: dict) -> RunStatus:
    """Convert stored status dict to RunStatus model."""
    prog = status_data.get("progress", {})
    progress = ProgressInfo(
        total_frames=prog.get("total_frames", 0),
        processed_frames=prog.get("processed_frames", 0),
        current_frame=prog.get("current_frame"),
        current_stage=prog.get("current_stage"),
        message=prog.get("message"),
        eta_seconds=prog.get("eta_seconds"),
        per_stage_timings=prog.get("per_stage_timings", {}),
    )

    return RunStatus(
        run_id=status_data.get("run_id", ""),
        state=status_data.get("state", RunState.CREATED),
        progress=progress,
        config=_tolerant_run_config(status_data.get("config", {})),
        created_at=datetime.fromisoformat(
            status_data.get("created_at", datetime.now().isoformat())
        ),
        started_at=datetime.fromisoformat(status_data["started_at"])
        if status_data.get("started_at") else None,
        finished_at=datetime.fromisoformat(status_data["finished_at"])
        if status_data.get("finished_at") else None,
        error_message=status_data.get("error_message"),
    )


@router.post("", response_model=RunCreateResponse)
async def create_new_run(
    config: Annotated[Optional[str], Form()] = None,
) -> RunCreateResponse:
    """
    Create a new run. Configuration can be passed as JSON string.
    Upload files separately via /runs/{run_id}/upload.
    """
    run_id = str(uuid4())[:8]  # Short UUID for readability

    # Parse and validate config if provided — invalid values fail here with
    # 422 instead of poisoning the stored status and 500ing /status later.
    run_config: dict = {}
    if config:
        try:
            raw = json.loads(config)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON in config field",
            )
        try:
            run_config = RunConfig.model_validate(
                {k: v for k, v in raw.items() if v is not None}
            ).model_dump()
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid run config: {e.errors()}",
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
    run_id: RunID,
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
                uploaded.append(file_path.name)

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
        uploaded.append(file_path.name)

        # Update config to indicate video processing
        from ..services.runner import _active_runs

        current_status = load_status(run_id)
        if current_status:
            config = current_status.get("config", {})
            config["video_file"] = file_path.name
            update_status(run_id, {"config": config})

            # Sync memory context so processing thread knows about the video
            if run_id in _active_runs:
                _active_runs[run_id].config["video_file"] = file_path.name

    return {
        "run_id": run_id,
        "uploaded_files": uploaded,
        "total_files": len(uploaded),
        "message": f"Successfully uploaded {len(uploaded)} file(s)",
    }


@router.post("/{run_id}/start", response_model=dict)
async def start_processing_run(
    run_id: RunID,
    config: Optional[RunConfig] = Body(default=None),
) -> dict:
    """
    Start processing a run.

    An optional config body overrides the run's stored configuration — the
    frontend sends the user's final Configure-step values here, so edits made
    after upload are honored.
    """
    current_status = load_status(run_id)
    if not current_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    if current_status.get("state") not in ("created", "uploading", "queued"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run is in state '{current_status.get('state')}', cannot start",
        )

    overrides = config.model_dump(exclude_unset=True) if config else None
    success = start_run(run_id, config_overrides=overrides)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Run could not be queued (already started or missing)",
        )

    return {
        "run_id": run_id,
        "status": "queued",
        "message": "Run queued for processing",
    }


@router.get("/{run_id}/status", response_model=RunStatus)
async def get_status(run_id: RunID) -> RunStatus:
    """Get the current status of a run."""
    current_status = get_run_status(run_id)
    if not current_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    return _serialize_status(current_status)


@router.get("/{run_id}/summary")
async def get_summary(run_id: RunID) -> dict:
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
async def get_metrics(run_id: RunID) -> list[dict]:
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
async def get_events(run_id: RunID) -> EventTimelineResponse:
    """Get the event_timeline.json for a completed run."""
    path = get_artifact_path(run_id, "events")
    if not path or not path.exists():
        return EventTimelineResponse(events=[], total_events=0)

    with open(path, "r") as f:
        events = json.load(f)
        return EventTimelineResponse(events=events, total_events=len(events))


@router.get("/{run_id}/files", response_model=FileListResponse)
async def list_files(run_id: RunID) -> FileListResponse:
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
                except (OSError, json.JSONDecodeError):
                    pass

            runs.append(
                RunSummary(
                    run_id=run_id,
                    state=current_status.get("state", RunState.CREATED),
                    total_frames=progress.get("total_frames", 0),
                    created_at=datetime.fromisoformat(
                        current_status.get("created_at", datetime.now().isoformat())
                    ),
                    finished_at=datetime.fromisoformat(current_status["finished_at"])
                    if current_status.get("finished_at") else None,
                    has_events=has_events,
                )
            )

    # Sort by created_at descending
    runs.sort(key=lambda r: r.created_at, reverse=True)

    return RunListResponse(runs=runs, total=len(runs))


@router.post("/{run_id}/cancel")
async def cancel_processing_run(run_id: RunID) -> dict:
    """Cancel a running or queued run."""
    success = cancel_run(run_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found or already completed",
        )

    return {"run_id": run_id, "status": "cancelled", "message": "Run cancellation requested"}


@router.delete("/{run_id}")
async def delete_run(run_id: RunID) -> dict:
    """Delete a run and all its data."""
    if not get_run_dir(run_id).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    cleanup_run(run_id)
    return {"run_id": run_id, "deleted": True, "message": "Run deleted successfully"}
