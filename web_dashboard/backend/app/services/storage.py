"""
File storage utilities for run isolation.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles

# Base directory for all runs
RUNS_BASE_DIR = Path(__file__).parent.parent.parent / "runs"
RUNS_BASE_DIR.mkdir(parents=True, exist_ok=True)


def get_run_dir(run_id: str) -> Path:
    """Get the base directory for a specific run."""
    return RUNS_BASE_DIR / run_id


def get_run_input_dir(run_id: str) -> Path:
    """Get input directory for a run."""
    return get_run_dir(run_id) / "input"


def get_run_output_dir(run_id: str) -> Path:
    """Get output directory for a run."""
    return get_run_dir(run_id) / "output"


def create_run_workspace(run_id: str) -> Path:
    """Create isolated workspace for a run."""
    run_dir = get_run_dir(run_id)
    input_dir = run_dir / "input"
    output_dir = run_dir / "output"
    escalation_dir = output_dir / "escalation_frames"
    annotated_dir = output_dir / "annotated"
    heatmaps_dir = output_dir / "heatmaps"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    escalation_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def delete_run_workspace(run_id: str) -> bool:
    """Delete a run's workspace. Returns True if deleted."""
    run_dir = get_run_dir(run_id)
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
        return True
    return False


def list_run_ids() -> list[str]:
    """List all run IDs."""
    if not RUNS_BASE_DIR.exists():
        return []
    return [
        d.name
        for d in RUNS_BASE_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]


# Status file operations

STATUS_FILENAME = "status.json"


def load_status(run_id: str) -> dict[str, Any] | None:
    """Load status JSON for a run."""
    status_path = get_run_dir(run_id) / STATUS_FILENAME
    if status_path.exists():
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def save_status(run_id: str, status: dict[str, Any]) -> None:
    """Save status JSON for a run."""
    status_path = get_run_dir(run_id) / STATUS_FILENAME
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, default=str)


def update_status(run_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Update status with partial data. Returns merged status."""
    status = load_status(run_id) or {}
    status.update(updates)
    save_status(run_id, status)
    return status


# Async file operations for uploads

async def save_uploaded_file(run_id: str, filename: str, content: bytes) -> Path:
    """Save an uploaded file to the run's input directory."""
    input_dir = get_run_input_dir(run_id)
    input_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    safe_filename = Path(filename).name
    file_path = input_dir / safe_filename

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    return file_path


# Output file discovery

def discover_output_files(run_id: str) -> dict[str, list[dict[str, Any]]]:
    """Discover all output files for a run, categorized."""
    output_dir = get_run_output_dir(run_id)
    if not output_dir.exists():
        return {}

    categories = {
        "json": [],
        "images": [],
        "plots": [],
        "database": [],
        "escalation": [],
        "annotated": [],
        "heatmaps": [],
        "other": [],
    }

    for root, _, files in os.walk(output_dir):
        root_path = Path(root)
        for filename in files:
            file_path = root_path / filename
            rel_path = file_path.relative_to(output_dir)

            file_info = {
                "name": filename,
                "path": str(rel_path).replace("\\", "/"),
                "full_path": str(file_path),
                "size_bytes": file_path.stat().st_size,
                "modified_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            }

            # Categorize
            ext = file_path.suffix.lower()
            rel_str = str(rel_path).lower()

            if ext == ".json":
                categories["json"].append(file_info)
            elif ext in (".db", ".sqlite", ".sqlite3"):
                categories["database"].append(file_info)
            elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                if "escalation" in rel_str:
                    categories["escalation"].append(file_info)
                elif "annotated" in rel_str or "annotated" in str(root_path).lower():
                    categories["annotated"].append(file_info)
                elif "heatmap" in rel_str or "heatmap" in str(root_path).lower():
                    categories["heatmaps"].append(file_info)
                elif "trend" in rel_str or "plot" in rel_str:
                    categories["plots"].append(file_info)
                else:
                    categories["images"].append(file_info)
            else:
                categories["other"].append(file_info)

    return categories


def get_output_file_path(run_id: str, relative_path: str) -> Path | None:
    """Get absolute path for an output file, validating it's within the run."""
    output_dir = get_run_output_dir(run_id)

    # Prevent directory traversal
    requested = (output_dir / relative_path).resolve()

    # Ensure the resolved path is still within output_dir
    try:
        requested.relative_to(output_dir.resolve())
    except ValueError:
        return None

    if requested.exists():
        return requested
    return None


def get_artifact_path(run_id: str, artifact_type: str, filename: str | None = None) -> Path | None:
    """Get path for known artifacts by type."""
    output_dir = get_run_output_dir(run_id)

    artifact_patterns = {
        "summary": "summary.json",
        "metrics": "metrics.json",
        "events": "event_timeline.json",
        "database": "crowd_analysis.db",
        "density_plot": "crowd_density_trend.png",
        "agitation_plot": "agitation_index_trend.png",
    }

    if artifact_type in artifact_patterns:
        path = output_dir / artifact_patterns[artifact_type]
        if path.exists():
            return path

    # For subdirs like annotated, heatmaps, escalation
    if artifact_type in ("annotated", "heatmaps", "escalation"):
        if filename:
            path = output_dir / f"{artifact_type}_frames" / filename
            if path.exists():
                return path
        # Return directory
        path = output_dir / f"{artifact_type}_frames"
        if path.exists():
            return path

    return None
