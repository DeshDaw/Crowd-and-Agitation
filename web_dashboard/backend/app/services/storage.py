"""
File storage utilities for run isolation.
"""

import json
import logging
import os
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles

logger = logging.getLogger(__name__)

# Base directory for all runs
RUNS_BASE_DIR = Path(__file__).parent.parent.parent / "runs"
RUNS_BASE_DIR.mkdir(parents=True, exist_ok=True)

# Run IDs are produced by str(uuid4())[:8] — enforce that shape everywhere so
# path parameters can never traverse out of RUNS_BASE_DIR.
RUN_ID_PATTERN = r"^[0-9a-f]{8}$"
_RUN_ID_RE = re.compile(RUN_ID_PATTERN)


def validate_run_id(run_id: str) -> str:
    """Return run_id if it matches the expected shape, else raise ValueError."""
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"Invalid run id: {run_id!r}")
    return run_id


def get_run_dir(run_id: str) -> Path:
    """Get the base directory for a specific run."""
    validate_run_id(run_id)
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
    resolved = run_dir.resolve()
    base = RUNS_BASE_DIR.resolve()
    if resolved == base or base not in resolved.parents:
        logger.error("Refusing to delete path outside runs dir: %s", resolved)
        return False
    if run_dir.exists():
        try:
            shutil.rmtree(run_dir)
        except OSError:
            logger.exception("Failed to fully delete run workspace %s", run_dir)
            return False
        return True
    return False


def list_run_ids() -> list[str]:
    """List all run IDs."""
    if not RUNS_BASE_DIR.exists():
        return []
    return [
        d.name
        for d in RUNS_BASE_DIR.iterdir()
        if d.is_dir() and _RUN_ID_RE.fullmatch(d.name)
    ]


# Status file operations
#
# status.json is written by the worker thread (per-frame progress), the event
# loop (upload metadata) and read by pollers. Writes are atomic
# (tmp + os.replace) and each run's read-modify-write cycle is serialized by a
# per-run lock so concurrent writers cannot drop keys or expose torn files.

STATUS_FILENAME = "status.json"

_status_locks: dict[str, threading.Lock] = {}
_status_locks_guard = threading.Lock()


def _status_lock(run_id: str) -> threading.Lock:
    with _status_locks_guard:
        if run_id not in _status_locks:
            _status_locks[run_id] = threading.Lock()
        return _status_locks[run_id]


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
    """Save status JSON for a run atomically."""
    status_path = get_run_dir(run_id) / STATUS_FILENAME
    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = status_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, default=str)
    os.replace(tmp_path, status_path)


def update_status(run_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Update status with partial data. Returns merged status."""
    with _status_lock(run_id):
        status = load_status(run_id) or {}
        status.update(updates)
        save_status(run_id, status)
        return status


def update_status_progress(run_id: str, progress_updates: dict[str, Any]) -> dict[str, Any]:
    """Merge updates into the nested progress dict under the run's lock."""
    with _status_lock(run_id):
        status = load_status(run_id) or {}
        progress = status.get("progress", {})
        progress.update(progress_updates)
        status["progress"] = progress
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

            # Categorize
            ext = file_path.suffix.lower()
            rel_str = str(rel_path).lower()

            if ext == ".json":
                category = "json"
            elif ext in (".db", ".sqlite", ".sqlite3"):
                category = "database"
            elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                if "escalation" in rel_str:
                    category = "escalation"
                elif "annotated" in rel_str:
                    category = "annotated"
                elif "heatmap" in rel_str:
                    category = "heatmaps"
                elif "trend" in rel_str or "plot" in rel_str:
                    category = "plots"
                else:
                    category = "images"
            else:
                category = "other"

            categories[category].append({
                "name": filename,
                "path": str(rel_path).replace("\\", "/"),
                "type": category,
                "size_bytes": file_path.stat().st_size,
                "modified_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            })

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


# Maps artifact types to the directory names the pipeline actually writes
# (create_run_workspace / runner.py). Only escalation uses a "_frames" suffix.
_ARTIFACT_DIRS = {
    "annotated": "annotated",
    "heatmaps": "heatmaps",
    "escalation": "escalation_frames",
}


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

    if artifact_type in _ARTIFACT_DIRS:
        dir_path = output_dir / _ARTIFACT_DIRS[artifact_type]
        if filename:
            # Strip any path components so {filename} cannot traverse
            path = dir_path / Path(filename).name
            try:
                path.resolve().relative_to(output_dir.resolve())
            except ValueError:
                return None
            if path.exists():
                return path
            return None
        if dir_path.exists():
            return dir_path

    return None
