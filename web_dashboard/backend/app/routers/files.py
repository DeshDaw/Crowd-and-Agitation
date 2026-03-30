"""
FastAPI routers for file downloads and artifacts.
"""

import mimetypes
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse

from ..services.storage import (
    get_artifact_path,
    get_output_file_path,
    get_run_dir,
    get_run_output_dir,
)

router = APIRouter(prefix="/runs", tags=["files"])


def _guess_mime_type(path: Path) -> str:
    """Guess MIME type from file extension."""
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"


@router.get("/{run_id}/download")
async def download_file(run_id: str, path: str) -> FileResponse:
    """
    Download a file from a run's output directory.
    Path should be relative to the output directory.
    """
    if not get_run_dir(run_id).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    file_path = get_output_file_path(run_id, path)
    if not file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {path}",
        )

    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type=_guess_mime_type(file_path),
    )


@router.get("/{run_id}/artifacts/summary.json")
async def get_summary_artifact(run_id: str) -> FileResponse:
    """Download summary.json."""
    path = get_artifact_path(run_id, "summary")
    if not path or not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Summary not found",
        )
    return FileResponse(path, media_type="application/json")


@router.get("/{run_id}/artifacts/metrics.json")
async def get_metrics_artifact(run_id: str) -> FileResponse:
    """Download metrics.json."""
    path = get_artifact_path(run_id, "metrics")
    if not path or not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics not found",
        )
    return FileResponse(path, media_type="application/json")


@router.get("/{run_id}/artifacts/events.json")
async def get_events_artifact(run_id: str) -> FileResponse:
    """Download event_timeline.json."""
    path = get_artifact_path(run_id, "events")
    if not path or not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Events not found",
        )
    return FileResponse(path, media_type="application/json")


@router.get("/{run_id}/artifacts/database.db")
async def get_database_artifact(run_id: str) -> FileResponse:
    """Download crowd_analysis.db."""
    path = get_artifact_path(run_id, "database")
    if not path or not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Database not found",
        )
    return FileResponse(
        path,
        media_type="application/vnd.sqlite3",
        filename="crowd_analysis.db",
    )


@router.get("/{run_id}/artifacts/density_plot.png")
async def get_density_plot(run_id: str) -> FileResponse:
    """Download crowd_density_trend.png."""
    path = get_artifact_path(run_id, "density_plot")
    if not path or not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Density plot not found",
        )
    return FileResponse(path, media_type="image/png")


@router.get("/{run_id}/artifacts/agitation_plot.png")
async def get_agitation_plot(run_id: str) -> FileResponse:
    """Download agitation_index_trend.png."""
    path = get_artifact_path(run_id, "agitation_plot")
    if not path or not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agitation plot not found",
        )
    return FileResponse(path, media_type="image/png")


@router.get("/{run_id}/artifacts/annotated/{filename}")
async def get_annotated_frame(run_id: str, filename: str) -> FileResponse:
    """Download an annotated frame image."""
    path = get_artifact_path(run_id, "annotated", filename)
    if not path or not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Annotated frame not found: {filename}",
        )
    return FileResponse(path, media_type="image/jpeg")


@router.get("/{run_id}/artifacts/heatmaps/{filename}")
async def get_heatmap_frame(run_id: str, filename: str) -> FileResponse:
    """Download a heatmap overlay image."""
    path = get_artifact_path(run_id, "heatmaps", filename)
    if not path or not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Heatmap not found: {filename}",
        )
    return FileResponse(path, media_type="image/jpeg")


@router.get("/{run_id}/artifacts/escalation/{filename}")
async def get_escalation_frame(run_id: str, filename: str) -> FileResponse:
    """Download an escalation event frame."""
    path = get_artifact_path(run_id, "escalation", filename)
    if not path or not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Escalation frame not found: {filename}",
        )
    return FileResponse(path, media_type="image/jpeg")


@router.get("/{run_id}/artifacts/annotated")
async def list_annotated_frames(run_id: str) -> list[str]:
    """List all annotated frame filenames."""
    dir_path = get_artifact_path(run_id, "annotated")
    if not dir_path or not dir_path.exists():
        return []

    return sorted([f.name for f in dir_path.iterdir() if f.is_file()])


@router.get("/{run_id}/artifacts/heatmaps")
async def list_heatmap_frames(run_id: str) -> list[str]:
    """List all heatmap filenames."""
    dir_path = get_artifact_path(run_id, "heatmaps")
    if not dir_path or not dir_path.exists():
        return []

    return sorted([f.name for f in dir_path.iterdir() if f.is_file()])


@router.get("/{run_id}/artifacts/escalation")
async def list_escalation_frames(run_id: str) -> list[dict]:
    """List all escalation frames with metadata from events.json."""
    import json

    dir_path = get_artifact_path(run_id, "escalation")
    events_path = get_artifact_path(run_id, "events")

    # Load events data
    events_map = {}
    if events_path and events_path.exists():
        try:
            with open(events_path) as f:
                for event in json.load(f):
                    events_map[event.get("frame_name", "")] = event
        except:
            pass

    if not dir_path or not dir_path.exists():
        return []

    frames = []
    for f in sorted(dir_path.iterdir()):
        if f.is_file():
            event_data = events_map.get(f.name, {})
            frames.append(
                {
                    "filename": f.name,
                    "frame_index": event_data.get("frame_index", 0),
                    "agitation_score": event_data.get("agitation_score", 0),
                    "density_ratio": event_data.get("density_ratio", 0),
                }
            )

    return frames
