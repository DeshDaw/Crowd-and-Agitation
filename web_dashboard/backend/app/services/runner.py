"""
Background processing service that adapts FrameProcessor for API usage.
"""

import json
import logging
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Callable

import cv2
import numpy as np

# Add crowd_project paths so both `import crowd_project.xxx` (used here)
# and bare `import config` (used inside crowd_project modules) resolve.
_RECAM_ROOT = Path(__file__).parent.parent.parent.parent.parent
_CROWD_DIR = _RECAM_ROOT / "crowd_project"
for _p in (str(_RECAM_ROOT), str(_CROWD_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import crowd_project modules
import crowd_project.config as crowd_config
from crowd_project.database import CrowdDatabase
from crowd_project.main import FrameProcessor
from crowd_project.video_utils import extract_frames_from_video, get_image_paths, load_image

from .storage import (
    create_run_workspace,
    delete_run_workspace,
    get_run_dir,
    get_run_input_dir,
    get_run_output_dir,
    save_status,
    update_status,
    load_status,
)

logger = logging.getLogger(__name__)

# Global executor for background processing
_executor = ThreadPoolExecutor(max_workers=2)
_active_runs: dict[str, "RunContext"] = {}
_lock = threading.Lock()


@dataclass
class RunContext:
    """Context for an active run."""

    run_id: str
    config: dict[str, Any]
    state: str = "created"
    progress: dict = field(default_factory=dict)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    error_message: str | None = None


def _update_progress(run_id: str, updates: dict[str, Any]) -> None:
    """Update progress in status file."""
    with _lock:
        if run_id in _active_runs:
            _active_runs[run_id].progress.update(updates)
    update_status(
        run_id, {"progress": {**load_status(run_id).get("progress", {}), **updates}}
    )


def _set_state(run_id: str, state: str, error: str | None = None) -> None:
    """Update run state."""
    with _lock:
        if run_id in _active_runs:
            _active_runs[run_id].state = state
            if error:
                _active_runs[run_id].error_message = error

    updates = {"state": state}
    if state == "processing":
        updates["started_at"] = datetime.now().isoformat()
    elif state in ("completed", "failed", "cancelled"):
        updates["finished_at"] = datetime.now().isoformat()
    if error:
        updates["error_message"] = error

    update_status(run_id, updates)


def _copy_config_to_crowd_project(run_config: dict[str, Any], output_dir: Path) -> None:
    """Temporarily override crowd_project config values."""
    # Store original values
    run_config["_original"] = {}

    # Override config values
    overrides = {
        "DEVICE": run_config.get("device", "cpu"),
        "CONFIDENCE_THRESHOLD": run_config.get("confidence_threshold", 0.5),
        "POSE_CONFIDENCE_THRESHOLD": run_config.get("pose_confidence_threshold", 0.5),
        "MAX_INFERENCE_WIDTH": run_config.get("max_inference_width", 960),
        "TRACKER_IOU_THRESHOLD": run_config.get("tracker_iou_threshold", 0.3),
        "TRACKER_MAX_LOST": run_config.get("tracker_max_lost", 5),
        "DENSITY_LOW_SIGMA": run_config.get("density_low_sigma", 0.5),
        "DENSITY_HIGH_SIGMA": run_config.get("density_high_sigma", 1.5),
        "AGITATION_THRESHOLD_SIGMA": run_config.get("agitation_threshold_sigma", 2.0),
        "VIDEO_EXTRACT_FPS": run_config.get("video_extract_fps"),
        "OUTPUT_DIR": output_dir,
        "OUTPUT_ANNOTATED_DIR": output_dir / "annotated",
        "OUTPUT_HEATMAPS_DIR": output_dir / "heatmaps",
        "OUTPUT_ESCALATION_DIR": output_dir / "escalation_frames",
    }

    for key, value in overrides.items():
        if hasattr(crowd_config, key):
            run_config["_original"][key] = getattr(crowd_config, key)
            setattr(crowd_config, key, value)

    # Ensure directories exist
    crowd_config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    crowd_config.OUTPUT_ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    crowd_config.OUTPUT_HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)
    crowd_config.OUTPUT_ESCALATION_DIR.mkdir(parents=True, exist_ok=True)


def _restore_config(run_config: dict[str, Any]) -> None:
    """Restore original crowd_project config values."""
    original = run_config.get("_original", {})
    for key, value in original.items():
        if hasattr(crowd_config, key):
            setattr(crowd_config, key, value)


def _process_run(context: RunContext) -> None:
    """Main processing function for a run."""
    run_id = context.run_id
    run_config = context.config

    try:
        _set_state(run_id, "processing")

        # Setup workspace
        create_run_workspace(run_id)
        input_dir = get_run_input_dir(run_id)
        output_dir = get_run_output_dir(run_id)

        # Override crowd_project config
        _copy_config_to_crowd_project(run_config, output_dir)

        # Handle video extraction if needed
        video_file = run_config.get("video_file")
        if video_file:
            _update_progress(run_id, {"current_stage": "video_extraction", "message": f"Extracting frames from {video_file}"})
            from crowd_project.video_utils import extract_frames_from_video

            video_path = input_dir / video_file
            fps = run_config.get("video_extract_fps")
            frames_dir = input_dir / f"extracted_{video_path.stem}"
            extract_frames_from_video(video_path, frames_dir, fps=fps)
            input_dir = frames_dir

        # Get image paths
        image_paths = get_image_paths(input_dir)
        if not image_paths:
            raise ValueError("No images found to process")

        total_frames = len(image_paths)
        _update_progress(
            run_id, {"total_frames": total_frames, "current_stage": "inference"}
        )

        # Initialize processor
        processor = FrameProcessor(device=run_config.get("device", "cpu"))

        # Create database if needed
        db = None
        if run_config.get("save_database", True):
            db_path = output_dir / "crowd_analysis.db"
            db = CrowdDatabase(db_path)

        # Process frames
        from crowd_project.analytics import build_summary, save_json
        from crowd_project.heatmap import generate_heatmap_visualization
        from crowd_project.tracker import Track

        timings = {"detection": [], "pose": []}

        for idx, path in enumerate(image_paths):
            if context.cancel_event.is_set():
                _set_state(run_id, "cancelled")
                if db:
                    db.close()
                _restore_config(run_config)
                return

            image = load_image(path)
            if image is None:
                continue

            # Process frame
            record = processor.process_frame(image, path.name, idx)

            # Track timings
            timings["detection"].append(record.get("inference_time_det", 0))
            timings["pose"].append(record.get("inference_time_pose", 0))

            # Save outputs based on config
            if run_config.get("save_annotated", True):
                cv2.imwrite(
                    str(output_dir / "annotated" / path.name),
                    record["annotated_image"],
                )

            if run_config.get("save_heatmaps", True):
                hm_name = path.stem + "_heatmap" + path.suffix
                cv2.imwrite(str(output_dir / "heatmaps" / hm_name), record["heatmap_overlay"])

            # Update progress
            _update_progress(
                run_id,
                {
                    "processed_frames": idx + 1,
                    "current_frame": path.name,
                    "per_stage_timings": {
                        "avg_detection_ms": round(sum(timings["detection"]) / len(timings["detection"]) * 1000, 2),
                        "avg_pose_ms": round(sum(timings["pose"]) / len(timings["pose"]) * 1000, 2),
                    },
                },
            )

        # Finalize batch
        _update_progress(run_id, {"current_stage": "classification"})
        mu_ag, std_ag, ag_threshold = processor.finalize_batch()

        # Detect escalation events
        _update_progress(run_id, {"current_stage": "event_detection"})
        processor.detect_escalation_events(ag_threshold, db=db)

        # Persist to database
        if db:
            for rec in processor.frame_records:
                db.insert_frame(
                    frame_id=rec["frame_name"],
                    people_count=rec["people_count"],
                    density_ratio=rec["density_ratio"],
                    agitation_index=rec["agitation_index"],
                    classification=rec["classification"],
                )
                db.insert_persons(rec["frame_name"], rec.get("person_rows", []))
            db.commit()
            db.close()

        # Save JSON outputs
        _update_progress(run_id, {"current_stage": "saving_outputs"})

        records = processor.frame_records
        metrics_rows = [
            {
                k: v
                for k, v in r.items()
                if k
                not in ("person_rows", "annotated_image", "heatmap_overlay", "active_tracks")
            }
            for r in records
        ]
        save_json(metrics_rows, output_dir / "metrics.json")

        summary = build_summary(
            records, len(processor.event_manager.events)
        )
        save_json(summary, output_dir / "summary.json")

        processor.event_manager.save_events_json(output_dir / "event_timeline.json")

        # Generate plots
        if run_config.get("generate_plots", True):
            from crowd_project.analytics import (
                plot_agitation_trend,
                plot_density_trend,
            )

            names = [r["frame_name"] for r in records]
            density_vals = [r["density_ratio"] for r in records]
            agitation_vals = [r["agitation_index"] for r in records]
            class_vals = [r["classification"] for r in records]

            plot_density_trend(
                names, density_vals, class_vals, output_dir / "crowd_density_trend.png"
            )
            plot_agitation_trend(
                names, agitation_vals, ag_threshold, output_dir / "agitation_index_trend.png"
            )

        _set_state(run_id, "completed")

    except Exception as e:
        logger.exception("Run %s failed", run_id)
        _set_state(run_id, "failed", error=str(e))
    finally:
        _restore_config(run_config)
        with _lock:
            if run_id in _active_runs:
                del _active_runs[run_id]


def create_run(run_id: str, config: dict[str, Any]) -> RunContext:
    """Create a new run context."""
    context = RunContext(run_id=run_id, config=config)
    with _lock:
        _active_runs[run_id] = context

    # Initialize status file
    save_status(
        run_id,
        {
            "run_id": run_id,
            "state": "created",
            "config": config,
            "progress": {
                "total_frames": 0,
                "processed_frames": 0,
                "current_frame": None,
                "current_stage": None,
            },
            "created_at": datetime.now().isoformat(),
        },
    )

    return context


def start_run(run_id: str) -> bool:
    """Start processing a run. Returns True if started."""
    with _lock:
        if run_id not in _active_runs:
            return False
        context = _active_runs[run_id]

    if context.state not in ("created", "queued"):
        return False

    thread = threading.Thread(target=_process_run, args=(context,), daemon=True)
    context.thread = thread
    thread.start()
    return True


def cancel_run(run_id: str) -> bool:
    """Request cancellation of a run. Returns True if cancelled or already done."""
    with _lock:
        if run_id not in _active_runs:
            # Check if run exists but is already finished
            status = load_status(run_id)
            if status and status.get("state") in ("completed", "failed", "cancelled"):
                return True
            return False
        context = _active_runs[run_id]

    context.cancel_event.set()
    return True


def get_run_status(run_id: str) -> dict[str, Any] | None:
    """Get current status of a run."""
    # First check active runs for live updates
    with _lock:
        if run_id in _active_runs:
            context = _active_runs[run_id]
            return {
                "run_id": run_id,
                "state": context.state,
                "progress": context.progress,
                "config": context.config,
            }

    # Fall back to status file
    from ..services.storage import load_status

    return load_status(run_id)


def cleanup_run(run_id: str) -> bool:
    """Cancel and delete a run."""
    cancel_run(run_id)

    # Wait briefly for thread to finish
    with _lock:
        if run_id in _active_runs:
            context = _active_runs[run_id]
            if context.thread and context.thread.is_alive():
                context.thread.join(timeout=2.0)

    delete_run_workspace(run_id)
    return True


def list_active_runs() -> list[str]:
    """List all active run IDs."""
    with _lock:
        return list(_active_runs.keys())
