"""
Background processing service that adapts FrameProcessor for API usage.

Runs are executed by a single dedicated worker thread consuming a queue:
two concurrent runs would keep four Detectron2 models resident (and race on
GPU memory once CUDA is in play), so queued execution is both the safe and
the fast option — the expensive model build is also reused per process.

Per-run configuration is passed as an explicit PipelineSettings object; no
module-global config is mutated anywhere.
"""

import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Resolve detectron2 BEFORE the repo root goes on sys.path: the vendored
# detectron2/ source clone at the repo root would otherwise shadow the
# editable install as an empty namespace package.
try:
    import detectron2  # noqa: F401
except ImportError:
    pass

# Ensure the repository root is importable so `crowd_project` (a real
# package) resolves regardless of the uvicorn working directory.
_REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from crowd_project.analytics import (
    build_summary,
    plot_agitation_trend,
    plot_density_trend,
    save_json,
)
from crowd_project.database import CrowdDatabase
from crowd_project.main import FrameProcessor
from crowd_project.settings import PipelineSettings
from crowd_project.video_utils import (
    extract_frames_from_video,
    get_image_paths,
    get_video_fps,
    load_image,
)

from .storage import (
    CALIBRATION_FILENAME,
    create_run_workspace,
    delete_run_workspace,
    get_run_dir,
    get_run_input_dir,
    get_run_output_dir,
    load_status,
    save_status,
    update_status,
    update_status_progress,
)

logger = logging.getLogger(__name__)

_active_runs: dict[str, "RunContext"] = {}
_lock = threading.Lock()

_run_queue: "queue.Queue[str]" = queue.Queue()
_worker_started = False

# Progress writes are throttled to roughly this interval; the dashboard
# polls every 2 s, so rewriting status.json per frame is pure churn.
_PROGRESS_WRITE_INTERVAL = 0.5


@dataclass
class RunContext:
    """Context for an active (queued or processing) run."""

    run_id: str
    config: dict[str, Any]
    state: str = "created"
    progress: dict = field(default_factory=dict)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    error_message: str | None = None


def _ensure_worker() -> None:
    global _worker_started
    with _lock:
        if _worker_started:
            return
        _worker_started = True
    thread = threading.Thread(target=_worker_loop, name="run-worker", daemon=True)
    thread.start()


def _worker_loop() -> None:
    while True:
        run_id = _run_queue.get()
        with _lock:
            context = _active_runs.get(run_id)
        if context is None:
            continue
        if context.cancel_event.is_set():
            _set_state(run_id, "cancelled")
            with _lock:
                _active_runs.pop(run_id, None)
            continue
        _process_run(context)


def _update_progress(run_id: str, updates: dict[str, Any]) -> None:
    """Update progress in memory and the status file."""
    with _lock:
        if run_id in _active_runs:
            _active_runs[run_id].progress.update(updates)
    update_status_progress(run_id, updates)


def _set_state(run_id: str, state: str, error: str | None = None) -> None:
    """Update run state."""
    with _lock:
        if run_id in _active_runs:
            _active_runs[run_id].state = state
            if error:
                _active_runs[run_id].error_message = error

    updates: dict[str, Any] = {"state": state}
    if state == "processing":
        updates["started_at"] = datetime.now().isoformat()
    elif state in ("completed", "failed", "cancelled"):
        updates["finished_at"] = datetime.now().isoformat()
    if error:
        updates["error_message"] = error

    update_status(run_id, updates)


def _settings_for_run(run_config: dict[str, Any], output_dir: Path) -> PipelineSettings:
    """Build per-run pipeline settings from the stored run config."""
    settings = PipelineSettings.from_overrides(run_config)
    settings.output_dir = output_dir
    settings.save_annotated = bool(run_config.get("save_annotated", True))
    settings.save_heatmaps = bool(run_config.get("save_heatmaps", True))
    return settings


def _process_run(context: RunContext) -> None:
    """Main processing function for a run (executes on the worker thread)."""
    run_id = context.run_id
    run_config = context.config

    try:
        _set_state(run_id, "processing")

        # Setup workspace
        create_run_workspace(run_id)
        input_dir = get_run_input_dir(run_id)
        output_dir = get_run_output_dir(run_id)

        # Handle video extraction if needed
        source_fps: float | None = run_config.get("source_fps")
        video_file = run_config.get("video_file")
        if video_file:
            _update_progress(run_id, {
                "current_stage": "video_extraction",
                "message": f"Extracting frames from {video_file}",
            })
            video_path = input_dir / Path(video_file).name
            fps = run_config.get("video_extract_fps")
            frames_dir = input_dir / f"extracted_{video_path.stem}"
            extract_frames_from_video(video_path, frames_dir, fps=fps)
            input_dir = frames_dir
            # Frames flow at the extraction rate (or the video's native rate)
            if source_fps is None:
                source_fps = fps or get_video_fps(video_path)

        # Get image paths
        image_paths = get_image_paths(input_dir)
        if not image_paths:
            raise ValueError("No images found to process")

        total_frames = len(image_paths)
        _update_progress(
            run_id,
            {"total_frames": total_frames, "current_stage": "inference", "message": None},
        )

        settings = _settings_for_run(run_config, output_dir)
        settings.source_fps = source_fps

        # Pick up a calibration saved via POST /runs/{id}/calibration
        calib_path = get_run_dir(run_id) / CALIBRATION_FILENAME
        if calib_path.is_file():
            settings.calibration_file = calib_path
            _update_progress(run_id, {"message": "Calibration active — metric density enabled"})

        processor = FrameProcessor(settings=settings)

        # Create database if needed
        db = None
        if run_config.get("save_database", True):
            db = CrowdDatabase(output_dir / "crowd_analysis.db")

        timings = {"detection": 0.0, "pose": 0.0, "frames": 0}
        started = time.monotonic()
        last_write = 0.0

        for idx, path in enumerate(image_paths):
            if context.cancel_event.is_set():
                _set_state(run_id, "cancelled")
                if db:
                    db.close()
                return

            image = load_image(path)
            if image is None:
                continue

            # Process frame (annotated/heatmap images are written inside)
            record = processor.process_frame(image, path.name, idx)

            timings["detection"] += record.get("inference_time_det", 0.0)
            timings["pose"] += record.get("inference_time_pose", 0.0)
            timings["frames"] += 1

            now = time.monotonic()
            if now - last_write >= _PROGRESS_WRITE_INTERVAL or idx + 1 == total_frames:
                last_write = now
                n = max(timings["frames"], 1)
                per_frame = (now - started) / max(idx + 1, 1)
                _update_progress(
                    run_id,
                    {
                        "processed_frames": idx + 1,
                        "current_frame": path.name,
                        "eta_seconds": int(per_frame * (total_frames - idx - 1)),
                        "per_stage_timings": {
                            "avg_detection_ms": round(timings["detection"] / n * 1000, 2),
                            "avg_pose_ms": round(timings["pose"] / n * 1000, 2),
                        },
                    },
                )

        # Finalize batch
        _update_progress(run_id, {"current_stage": "classification"})
        _mu_ag, _std_ag, ag_threshold = processor.finalize_batch()

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
                    persons_per_m2=rec.get("persons_per_m2"),
                    los_class=rec.get("los_class"),
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
                if k not in (
                    "person_rows", "keypoint_snapshots",
                    "annotated_path", "heatmap_path",
                )
            }
            for r in records
        ]
        save_json(metrics_rows, output_dir / "metrics.json")

        summary = build_summary(
            records, len(processor.event_manager.events),
            extra={"state_classifier": getattr(processor, "state_classifier_name", None)},
        )
        save_json(summary, output_dir / "summary.json")

        processor.event_manager.save_events_json(output_dir / "event_timeline.json")

        # Generate plots
        if run_config.get("generate_plots", True):
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
        with _lock:
            _active_runs.pop(run_id, None)


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


def _rehydrate_run(run_id: str) -> RunContext | None:
    """Rebuild an in-memory context from status.json (e.g. after restart)."""
    status = load_status(run_id)
    if not status:
        return None
    if status.get("state") not in ("created", "queued", "uploading"):
        return None
    context = RunContext(
        run_id=run_id,
        config=status.get("config", {}),
        state=status.get("state", "created"),
        progress=status.get("progress", {}),
    )
    with _lock:
        _active_runs[run_id] = context
    return context


def start_run(run_id: str, config_overrides: dict[str, Any] | None = None) -> bool:
    """Queue a run for processing. Returns True if queued."""
    with _lock:
        context = _active_runs.get(run_id)

    if context is None:
        context = _rehydrate_run(run_id)
        if context is None:
            return False

    if context.state not in ("created", "queued", "uploading"):
        return False

    if config_overrides:
        context.config.update(config_overrides)
        update_status(run_id, {"config": context.config})

    _set_state(run_id, "queued")
    _ensure_worker()
    _run_queue.put(run_id)
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
    """
    Get current status of a run.

    The persisted status.json is the base (it carries created_at, config,
    started_at); live in-memory state/progress is merged over it so active
    runs report both fresh progress and their real timestamps.
    """
    status = load_status(run_id)

    with _lock:
        context = _active_runs.get(run_id)
        if context is not None:
            status = status or {"run_id": run_id, "config": context.config}
            status["state"] = context.state
            merged_progress = dict(status.get("progress", {}))
            merged_progress.update(context.progress)
            status["progress"] = merged_progress

    return status


def cleanup_run(run_id: str) -> bool:
    """Cancel and delete a run."""
    cancel_run(run_id)

    # Give a processing run a moment to observe the cancel flag before the
    # workspace disappears under it.
    for _ in range(20):
        with _lock:
            if run_id not in _active_runs:
                break
        time.sleep(0.1)

    with _lock:
        _active_runs.pop(run_id, None)

    delete_run_workspace(run_id)
    return True


def list_active_runs() -> list[str]:
    """List all active run IDs."""
    with _lock:
        return list(_active_runs.keys())
