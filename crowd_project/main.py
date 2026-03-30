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

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

import analytics
import config
from agitation_analyzer import AgitationAnalyzer, AgitationMetrics
from database import CrowdDatabase
from density_analyzer import DensityAnalyzer, DensityMetrics
from detection_engine import Detection, DetectionEngine
from event_manager import EventManager
from heatmap import generate_heatmap_visualization
from model_registry import ModelRegistry
from motion_analyzer import MotionAnalyzer, PersonMotion
from pose_engine import PoseEngine
from tracker import IoUTracker, Track
from video_utils import extract_frames_from_video, get_image_paths, load_image

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

    Each engine:
      - Receives a frame (or derived data).
      - Returns structured output.
      - Has no side effects outside its own domain.

    FrameProcessor merges outputs and manages batch-level post-processing.
    """

    def __init__(self, device: str | None = None) -> None:
        dev = (device or config.DEVICE).lower()
        logger.info("Initializing FrameProcessor (device=%s)", dev)

        self._registry = ModelRegistry(device=dev)
        self.detection_engine = DetectionEngine(self._registry)
        self.pose_engine = PoseEngine(self._registry)
        self.tracker = IoUTracker()
        self.density_analyzer = DensityAnalyzer()
        self.motion_analyzer = MotionAnalyzer()
        self.agitation_analyzer = AgitationAnalyzer()
        self.event_manager = EventManager()

        # batch accumulators (reset on each run)
        self._frame_records: list[dict[str, Any]] = []
        self._density_list: list[DensityMetrics] = []
        self._agitation_list: list[AgitationMetrics] = []
        self._prev_density: float = 0.0

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

        # 1. Detection
        detections, det_time = self.detection_engine.detect(image, resize=True)

        # 2. Pose
        poses, pose_time = self.pose_engine.extract(image, resize=True)

        # 3. Tracking
        active_tracks = self.tracker.update(detections, poses, frame_index)

        # 4. Density
        density = self.density_analyzer.analyze_frame(detections, (h, w))
        self._density_list.append(density)

        # 5. Motion
        motions = self.motion_analyzer.compute_person_motions(active_tracks)

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

        # -- Annotated + heatmap images ------------------------------------
        annotated = self.detection_engine.draw_boxes(image, detections)
        _, heatmap_overlay = generate_heatmap_visualization(
            image,
            _det_boxes(detections),
            downscale=config.HEATMAP_DOWNSCALE,
            sigma=config.HEATMAP_SIGMA,
            colormap=config.HEATMAP_COLORMAP,
            alpha=config.HEATMAP_ALPHA,
        )

        avg_conf = (
            sum(d.confidence for d in detections) / len(detections)
            if detections else 0.0
        )

        record = {
            "frame_name": frame_name,
            "frame_index": frame_index,
            "people_count": len(detections),
            "inference_time_det": round(det_time, 4),
            "inference_time_pose": round(pose_time, 4),
            "average_confidence": round(avg_conf, 4),
            "density_ratio": density.density_ratio,
            "agitation_index": agitation.agitation_index,
            "classification": "",               # set in batch post-processing
            "person_rows": person_rows,
            "active_tracks": active_tracks,
            "annotated_image": annotated,
            "heatmap_overlay": heatmap_overlay,
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
        # Density classification (modifies DensityMetrics in-place)
        self.density_analyzer.classify_batch(self._density_list)

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
                annotated_image=rec.get("annotated_image"),
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
        tracks: list[Track] = rec.get("active_tracks", [])
        pairs: list[tuple[int, np.ndarray]] = []
        for trk in tracks:
            if trk.keypoint_history:
                pairs.append((trk.track_id, trk.keypoint_history[-1]))
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

def run_pipeline(input_dir: Path) -> None:
    """Full batch: process all frames -> classify -> events -> persist -> report."""
    # Ensure output dirs
    config.OUTPUT_ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = get_image_paths(input_dir)
    if not image_paths:
        logger.warning("No images in %s -- nothing to do.", input_dir)
        return

    # -- Build processor + storage -----------------------------------------
    processor = FrameProcessor(device=config.DEVICE)
    db = CrowdDatabase()

    # -- Pass 1: per-frame inference ---------------------------------------
    for idx, path in enumerate(tqdm(image_paths, desc="Processing frames", unit="frame")):
        image = load_image(path)
        if image is None:
            continue

        rec = processor.process_frame(image, path.name, idx)

        # Save annotated + heatmap immediately (keeps memory bounded)
        cv2.imwrite(
            str(config.OUTPUT_ANNOTATED_DIR / rec["frame_name"]),
            rec["annotated_image"],
        )
        hm_name = path.stem + "_heatmap" + path.suffix
        cv2.imwrite(
            str(config.OUTPUT_HEATMAPS_DIR / hm_name),
            rec["heatmap_overlay"],
        )

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
         if k not in ("person_rows", "annotated_image", "heatmap_overlay", "active_tracks")}
        for r in records
    ]
    analytics.save_json(metrics_rows, config.OUTPUT_DIR / config.METRICS_JSON)

    summary = analytics.build_summary(
        records, len(processor.event_manager.events),
    )
    analytics.save_json(summary, config.OUTPUT_DIR / config.SUMMARY_JSON)

    processor.event_manager.save_events_json()

    # -- Plots -------------------------------------------------------------
    names = [r["frame_name"] for r in records]
    density_vals = [r["density_ratio"] for r in records]
    agitation_vals = [r["agitation_index"] for r in records]
    class_vals = [r["classification"] for r in records]

    analytics.plot_density_trend(
        names, density_vals, class_vals,
        config.OUTPUT_DIR / config.CROWD_DENSITY_TREND_PNG,
    )
    analytics.plot_agitation_trend(
        names, agitation_vals, ag_threshold,
        config.OUTPUT_DIR / config.AGITATION_TREND_PNG,
    )

    # -- Console report ----------------------------------------------------
    _print_report(summary, ag_threshold)


# =====================================================================
# Console report
# =====================================================================

def _print_report(summary: dict[str, Any], ag_threshold: float) -> None:
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
    print(f"    Annotated frames  : {config.OUTPUT_ANNOTATED_DIR}")
    print(f"    Heatmaps          : {config.OUTPUT_HEATMAPS_DIR}")
    print(f"    Escalation frames : {config.OUTPUT_ESCALATION_DIR}")
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
    args = parser.parse_args()

    input_dir: Path = args.input_dir or config.INPUT_IMAGES_DIR

    if args.video is not None:
        if not args.video.is_file():
            logger.error("Video not found: %s", args.video)
            return
        frames_dir = config.INPUT_IMAGES_DIR / f"extracted_{args.video.stem}"
        extract_frames_from_video(args.video, frames_dir, fps=config.VIDEO_EXTRACT_FPS)
        input_dir = frames_dir

    logger.info("Device : %s", config.DEVICE)
    logger.info("Input  : %s", input_dir)

    run_pipeline(input_dir)


if __name__ == "__main__":
    main()
