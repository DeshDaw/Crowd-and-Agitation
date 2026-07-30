"""
Orchestrator for the Crowd Intelligence System.

Central class: :class:`FrameProcessor`, which owns every engine and analyzer
as a composed sub-component with no side effects outside its domain.

Pipeline per frame:
    Frame -> Detection -> Pose -> Tracking -> Density -> Motion -> Agitation

Batch post-processing:
    Classify density -> Compute agitation threshold -> Detect escalation events
    -> Persist (SQLite + JSON + plots) -> Console report

Usage:
    python main.py                             # images from input/images
    python main.py --video path/to/video.mp4   # extract frames first
    python main.py --input-dir path/to/frames
"""

from __future__ import annotations

# Allow running both as a module (python -m crowd_project.main) and as a
# script (python main.py) — the script path needs the parent directory on
# sys.path and an explicit package name before the relative imports below.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path

    # Resolve detectron2 BEFORE the repo root goes on sys.path: the vendored
    # detectron2/ clone at the repo root would otherwise shadow the editable
    # install as an empty namespace package (PEP 660 finder loses to
    # PathFinder's namespace fallback).
    try:
        import detectron2  # noqa: F401
    except ImportError:
        pass

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    __package__ = "crowd_project"

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from . import analytics, config
from .agitation_analyzer import AgitationAnalyzer, AgitationMetrics
from .database import CrowdDatabase
from .density_analyzer import DensityAnalyzer, DensityMetrics
from .detection_engine import Detection, DetectionEngine
from .event_manager import EventManager
from .heatmap import generate_heatmap_visualization
from .model_registry import ModelRegistry
from .motion_analyzer import MotionAnalyzer
from .pose_engine import PoseEngine
from .settings import PipelineSettings
from .tracker import IoUTracker
from .video_utils import extract_frames_from_video, get_image_paths, load_image

import cv2

# ── logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format=config.LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =====================================================================
# FrameProcessor — the central orchestration class
# =====================================================================

class FrameProcessor:
    """
    Multi-model frame-processing orchestrator.

    Owns all engines and analyzers as composed sub-components::

        FrameProcessor
        ├── DetectionEngine   (spatial)
        ├── PoseEngine        (structural)
        ├── IoUTracker        (temporal)
        ├── DensityAnalyzer   (statistical)
        ├── MotionAnalyzer    (kinematic)
        ├── AgitationAnalyzer (behavioral)
        └── EventManager      (escalation)

    All tunables come from a :class:`PipelineSettings` instance resolved at
    construction time, so per-run overrides reach every component and no
    module-global state is mutated.

    Annotated and heatmap images are written to disk as each frame is
    processed; frame records carry only their paths, keeping memory usage
    flat regardless of batch length.
    """

    def __init__(
        self,
        settings: PipelineSettings | None = None,
        device: str | None = None,
    ) -> None:
        s = settings or PipelineSettings()
        if device is not None:
            s.device = device.lower()
        self.settings = s
        self.backend = s.detection_backend.lower()
        logger.info(
            "Initializing FrameProcessor (backend=%s, device=%s)",
            self.backend, s.device,
        )

        if self.backend == "yolo":
            # Single-pass detection + tracking + pose. YOLO keypoint
            # confidences are probabilities, so motion gating uses the
            # YOLO-specific threshold instead of the R-CNN logit threshold.
            from .yolo_engine import YoloEngine
            self.yolo_engine = YoloEngine(
                weights=s.yolo_weights,
                device=s.device,
                conf=s.yolo_conf,
                imgsz=s.yolo_imgsz,
                tracker_type=s.yolo_tracker,
                max_lost=s.tracker_max_lost,
            )
            motion_vis_thresh = s.yolo_keypoint_conf
        elif self.backend == "detectron2":
            self._registry = ModelRegistry(
                device=s.device,
                confidence_threshold=s.confidence_threshold,
                pose_confidence_threshold=s.pose_confidence_threshold,
            )
            self.detection_engine = DetectionEngine(
                self._registry, max_inference_width=s.max_inference_width,
            )
            self.pose_engine = PoseEngine(
                self._registry, max_inference_width=s.max_inference_width,
            )
            self.tracker = IoUTracker(
                iou_threshold=s.tracker_iou_threshold,
                max_lost=s.tracker_max_lost,
            )
            motion_vis_thresh = s.keypoint_visibility_thresh
        else:
            raise ValueError(
                f"Unknown detection backend {s.detection_backend!r} "
                "(expected 'detectron2' or 'yolo')"
            )

        self.density_analyzer = DensityAnalyzer()
        self.motion_analyzer = MotionAnalyzer(
            vis_thresh=motion_vis_thresh,
            min_valid_keypoints=s.min_valid_keypoints,
        )
        self.agitation_analyzer = AgitationAnalyzer(
            w_mean=s.agitation_w_mean_motion,
            w_var=s.agitation_w_motion_variance,
            w_dir=s.agitation_w_directional_variance,
            w_density=s.agitation_w_density_change,
            threshold_sigma=s.agitation_threshold_sigma,
        )
        self.event_manager = EventManager(output_dir=s.output_dir)

        if s.save_annotated:
            s.annotated_dir.mkdir(parents=True, exist_ok=True)
        if s.save_heatmaps:
            s.heatmaps_dir.mkdir(parents=True, exist_ok=True)

        # batch accumulators (reset on each run)
        self._frame_records: list[dict[str, Any]] = []
        self._density_list: list[DensityMetrics] = []
        self._agitation_list: list[AgitationMetrics] = []
        self._prev_density: float | None = None

    # ------------------------------------------------------------------
    # Per-frame inference (Pass 1)
    # ------------------------------------------------------------------

    def process_frame(
        self,
        image: np.ndarray,
        frame_name: str,
        frame_index: int,
    ) -> dict[str, Any]:
        """
        Run the full per-frame pipeline and return a record dict.

        Stages (all independent-domain, merged here):
            1. Detection  (spatial)
            2. Pose       (structural)
            3. Tracking   (temporal)
            4. Density    (statistical)
            5. Motion     (kinematic)
            6. Agitation  (behavioral — computed, not yet classified)
        """
        h, w = image.shape[:2]
        s = self.settings

        if self.backend == "yolo":
            # 1-3. Detection + tracking + pose in a single pass
            detections, active_tracks, det_time, head_count = (
                self.yolo_engine.process(image, frame_index)
            )
            pose_time = 0.0
        else:
            head_count = None
            # 1. Detection
            detections, det_time = self.detection_engine.detect(image, resize=True)

            # 2. Pose
            poses, pose_time = self.pose_engine.extract(image, resize=True)

            # 3. Tracking (returns only tracks matched in this frame)
            active_tracks = self.tracker.update(detections, poses, frame_index)

        # 4. Density
        density = self.density_analyzer.analyze_frame(detections, (h, w))
        self._density_list.append(density)

        # 5. Motion
        motions = self.motion_analyzer.compute_person_motions(
            active_tracks, frame_index,
        )

        # 6. Agitation (frame-level)
        agitation = self.agitation_analyzer.compute_frame(
            motions, density.density_ratio, self._prev_density,
        )
        self._agitation_list.append(agitation)
        self._prev_density = density.density_ratio

        # -- Build per-person rows for DB ----------------------------------
        motion_map: dict[int, float] = {
            m.track_id: m.normalized_motion for m in motions
        }
        person_rows: list[dict[str, Any]] = []
        keypoint_snapshots: dict[int, np.ndarray] = {}
        for trk in active_tracks:
            person_rows.append({
                "track_id": trk.track_id,
                "motion_score": motion_map.get(trk.track_id, 0.0),
                "centroid_x": trk.centroid[0],
                "centroid_y": trk.centroid[1],
                "bbox_area": float(
                    (trk.bbox[2] - trk.bbox[0]) * (trk.bbox[3] - trk.bbox[1])
                ),
            })
            if (
                trk.keypoint_history
                and trk.keypoint_history[-1].frame_index == frame_index
            ):
                keypoint_snapshots[trk.track_id] = trk.keypoint_history[-1].keypoints

        # -- Annotated + heatmap images: write immediately, keep paths -----
        annotated_path: Path | None = None
        heatmap_path: Path | None = None

        if s.save_annotated:
            annotated = DetectionEngine.draw_boxes(image, detections)
            annotated_path = s.annotated_dir / frame_name
            cv2.imwrite(str(annotated_path), annotated)

        if s.save_heatmaps:
            _, heatmap_overlay = generate_heatmap_visualization(
                image,
                _det_boxes(detections),
                downscale=s.heatmap_downscale,
                sigma=s.heatmap_sigma,
                colormap=s.heatmap_colormap,
                alpha=s.heatmap_alpha,
            )
            p = Path(frame_name)
            heatmap_path = s.heatmaps_dir / f"{p.stem}_heatmap{p.suffix}"
            cv2.imwrite(str(heatmap_path), heatmap_overlay)

        avg_conf = (
            sum(d.confidence for d in detections) / len(detections)
            if detections else 0.0
        )

        record = {
            "frame_name": frame_name,
            "frame_index": frame_index,
            "people_count": len(detections),
            "head_count": head_count,
            "inference_time_det": round(det_time, 4),
            "inference_time_pose": round(pose_time, 4),
            "average_confidence": round(avg_conf, 4),
            "density_ratio": density.density_ratio,
            "agitation_index": agitation.agitation_index,
            "classification": "",               # set in batch post-processing
            "person_rows": person_rows,
            "keypoint_snapshots": keypoint_snapshots,
            "annotated_path": str(annotated_path) if annotated_path else None,
            "heatmap_path": str(heatmap_path) if heatmap_path else None,
        }
        self._frame_records.append(record)
        return record

    # ------------------------------------------------------------------
    # Batch post-processing (Pass 2 + 3)
    # ------------------------------------------------------------------

    def finalize_batch(self) -> tuple[float, float, float]:
        """
        Run batch-level classification and compute agitation thresholds.

        Returns:
            (mean_agitation, std_agitation, agitation_threshold)
        """
        s = self.settings

        # Density classification (modifies DensityMetrics in-place)
        self.density_analyzer.classify_batch(
            self._density_list,
            low_sigma=s.density_low_sigma,
            high_sigma=s.density_high_sigma,
            high_min_ratio=s.density_high_min_ratio,
            low_max_ratio=s.density_low_max_ratio,
        )

        # Agitation threshold
        mu_ag, std_ag, ag_threshold = (
            self.agitation_analyzer.compute_batch_threshold(self._agitation_list)
        )

        # Back-fill classification into frame records
        for i, rec in enumerate(self._frame_records):
            rec["classification"] = self._density_list[i].classification
            rec["agitation_index"] = self._agitation_list[i].agitation_index

        return mu_ag, std_ag, ag_threshold

    def detect_escalation_events(
        self,
        ag_threshold: float,
        db: CrowdDatabase | None = None,
    ) -> None:
        """
        Scan all frame records for escalation events (Pass 3).

        If *db* is provided and an event fires, raw keypoints for every
        tracked person in that frame are persisted to the ``keypoints`` table.
        """
        for rec in self._frame_records:
            event = self.event_manager.check(
                frame_name=rec["frame_name"],
                frame_index=rec["frame_index"],
                agitation_score=rec["agitation_index"],
                agitation_threshold=ag_threshold,
                density_classification=rec["classification"],
                density_ratio=rec["density_ratio"],
                annotated_path=rec.get("annotated_path"),
            )
            if event is not None and db is not None:
                self._store_escalation_keypoints(db, rec)

    @property
    def frame_records(self) -> list[dict[str, Any]]:
        return self._frame_records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _store_escalation_keypoints(
        db: CrowdDatabase,
        rec: dict[str, Any],
    ) -> None:
        """Persist raw keypoints for an escalation-event frame."""
        snapshots: dict[int, np.ndarray] = rec.get("keypoint_snapshots", {})
        pairs = [(tid, kps) for tid, kps in snapshots.items()]
        if pairs:
            db.insert_keypoints(rec["frame_name"], pairs)


# =====================================================================
# Utility
# =====================================================================

def _det_boxes(detections: list[Detection]) -> np.ndarray:
    """Stack detection bboxes into (N,4) ndarray for heatmap generator."""
    if not detections:
        return np.empty((0, 4), dtype=np.float32)
    return np.array([d.bbox for d in detections], dtype=np.float32)


# =====================================================================
# run_pipeline — end-to-end batch execution
# =====================================================================

def run_pipeline(
    input_dir: Path,
    settings: PipelineSettings | None = None,
) -> None:
    """Full batch: process all frames -> classify -> events -> persist -> report."""
    s = settings or PipelineSettings()

    image_paths = get_image_paths(input_dir)
    if not image_paths:
        logger.warning("No images in %s -- nothing to do.", input_dir)
        return

    # -- Build processor + storage -----------------------------------------
    processor = FrameProcessor(settings=s)
    db = CrowdDatabase(s.output_dir / config.DATABASE_FILE)

    # -- Pass 1: per-frame inference (images written inside process_frame) --
    for idx, path in enumerate(tqdm(image_paths, desc="Processing frames", unit="frame")):
        image = load_image(path)
        if image is None:
            continue
        processor.process_frame(image, path.name, idx)

    records = processor.frame_records
    if not records:
        logger.warning("No frames processed.")
        db.close()
        return

    # -- Pass 2: batch classification --------------------------------------
    _mu_ag, _std_ag, ag_threshold = processor.finalize_batch()

    # -- Pass 3: escalation events (with conditional keypoint storage) -----
    processor.detect_escalation_events(ag_threshold, db=db)

    # -- Persist to SQLite -------------------------------------------------
    for rec in records:
        db.insert_frame(
            frame_id=rec["frame_name"],
            people_count=rec["people_count"],
            density_ratio=rec["density_ratio"],
            agitation_index=rec["agitation_index"],
            classification=rec["classification"],
        )
        db.insert_persons(rec["frame_name"], rec["person_rows"])
    db.commit()
    db.close()

    # -- JSON outputs ------------------------------------------------------
    metrics_rows = [
        {k: v for k, v in r.items()
         if k not in ("person_rows", "keypoint_snapshots",
                      "annotated_path", "heatmap_path")}
        for r in records
    ]
    analytics.save_json(metrics_rows, s.output_dir / config.METRICS_JSON)

    summary = analytics.build_summary(
        records, len(processor.event_manager.events),
    )
    analytics.save_json(summary, s.output_dir / config.SUMMARY_JSON)

    processor.event_manager.save_events_json()

    # -- Plots -------------------------------------------------------------
    names = [r["frame_name"] for r in records]
    density_vals = [r["density_ratio"] for r in records]
    agitation_vals = [r["agitation_index"] for r in records]
    class_vals = [r["classification"] for r in records]

    analytics.plot_density_trend(
        names, density_vals, class_vals,
        s.output_dir / config.CROWD_DENSITY_TREND_PNG,
    )
    analytics.plot_agitation_trend(
        names, agitation_vals, ag_threshold,
        s.output_dir / config.AGITATION_TREND_PNG,
    )

    # -- Console report ----------------------------------------------------
    _print_report(summary, ag_threshold, s)


# =====================================================================
# Console report
# =====================================================================

def _print_report(
    summary: dict[str, Any],
    ag_threshold: float,
    s: PipelineSettings,
) -> None:
    w = 62
    print("\n" + "=" * w)
    print("  CROWD INTELLIGENCE SYSTEM - BATCH REPORT")
    print("=" * w)
    print(f"  Total frames processed       : {summary['total_frames']}")
    print(f"  Average crowd count          : {summary['average_crowd_count']}")
    print(f"  Std crowd count              : {summary['std_crowd_count']}")
    print(f"  Mean density ratio           : {summary['mean_density']}")
    print(f"  Peak density frame           : {summary['peak_density_frame']}")
    print(f"  Peak density value           : {summary['peak_density_value']}")
    print(f"  Mean agitation index         : {summary['mean_agitation']}")
    print(f"  Highest agitation frame      : {summary['highest_agitation_frame']}")
    print(f"  Highest agitation value      : {summary['highest_agitation_value']}")
    print(f"  Agitation threshold          : {ag_threshold:.6f}")
    print(f"  Total escalation events      : {summary['total_escalation_events']}")
    print("-" * w)
    print("  Crowd classification distribution:")
    for cls, cnt in summary.get("crowd_classification_distribution", {}).items():
        print(f"    {cls:20s} : {cnt}")
    print("-" * w)
    print("  Outputs:")
    print(f"    Annotated frames  : {s.annotated_dir}")
    print(f"    Heatmaps          : {s.heatmaps_dir}")
    print(f"    Escalation frames : {s.escalation_dir}")
    print(f"    {config.METRICS_JSON}")
    print(f"    {config.SUMMARY_JSON}")
    print(f"    {config.EVENTS_JSON}")
    print(f"    {config.DATABASE_FILE}")
    print(f"    {config.CROWD_DENSITY_TREND_PNG}")
    print(f"    {config.AGITATION_TREND_PNG}")
    print("=" * w + "\n")


# =====================================================================
# CLI
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crowd Intelligence System (Detectron2 multi-model pipeline).",
    )
    parser.add_argument(
        "--video", type=Path, default=None,
        help="Extract frames from video then process.",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Folder of images to process (default: config.INPUT_IMAGES_DIR).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Inference device (cpu/cuda). Default: config.DEVICE.",
    )
    parser.add_argument(
        "--backend", type=str, default=None, choices=["detectron2", "yolo"],
        help="Detection backend. Default: config.DETECTION_BACKEND (env CROWD_BACKEND).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: crowd_project/output).",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir or config.INPUT_IMAGES_DIR

    if args.video is not None:
        if not args.video.is_file():
            logger.error("Video not found: %s", args.video)
            return
        frames_dir = config.INPUT_IMAGES_DIR / f"extracted_{args.video.stem}"
        extract_frames_from_video(args.video, frames_dir, fps=config.VIDEO_EXTRACT_FPS)
        input_dir = frames_dir

    settings = PipelineSettings()
    if args.device:
        settings.device = args.device.lower()
    if args.backend:
        settings.detection_backend = args.backend.lower()
    if args.output_dir:
        settings.output_dir = args.output_dir

    logger.info("Backend: %s", settings.detection_backend)
    logger.info("Device : %s", settings.device)
    logger.info("Input  : %s", input_dir)

    run_pipeline(input_dir, settings=settings)


if __name__ == "__main__":
    main()
