# Crowd Intelligence System — Detection, Density & Agitation Pipeline

Research-oriented prototype using **Detectron2** (Faster R-CNN + Keypoint R-CNN, COCO) for person detection, pose estimation, IoU/Hungarian tracking, per-frame density metrics, pose-based motion scoring, a weighted **Agitation Index**, and escalation-event detection. Batch analytics are persisted to JSON, SQLite, and trend plots. A FastAPI + React dashboard lives in `../web_dashboard/`.

Designed CPU-first with an easy switch to CUDA (`--device cuda` or `CROWD_DEVICE=cuda`).

## Environment

- **OS:** Windows 11
- **Python:** 3.10
- **PyTorch:** 2.1.2 (CPU-only by default)
- **Detectron2:** built from source (see `requirements.txt` for the install command)
- **NumPy:** 1.26.4 · **OpenCV:** 4.8.1.78 · **SciPy:** ≥1.10 (Hungarian matching)

## Project Structure

```
crowd_project/
├── __init__.py              # Package marker (import as crowd_project.*)
├── config.py                # All default parameters, thresholds, weights, paths
├── settings.py              # PipelineSettings: per-run override object
├── model_registry.py        # Lazy-load & cache Detectron2 models by key
├── detection_engine.py      # Faster R-CNN person detection -> bbox, confidence, area, centroid
├── pose_engine.py           # Keypoint R-CNN -> 17 keypoints + scores per person
├── tracker.py               # IoU + Hungarian matching -> persistent track_id per person
├── density_analyzer.py      # density_ratio + Low/Moderate/High classification (relative + absolute floors)
├── motion_analyzer.py       # Visible-keypoint displacement normalized by torso length & frame gap
├── agitation_analyzer.py    # Weighted agitation index (motion + variance + direction + density)
├── event_manager.py         # Escalation event detection + event_timeline.json + escalation frames
├── database.py              # SQLite storage (frames, persons, keypoints tables)
├── analytics.py             # Summary, JSON persistence, trend plots
├── heatmap.py               # Gaussian density heatmap overlays
├── video_utils.py           # Image loading, video frame extraction
├── main.py                  # Orchestrator: 3-pass pipeline + console report
├── requirements.txt
├── input/images/            # Put images here (or extracted frames)
├── input/video/             # Optional: place videos here
└── output/
    ├── annotated/           # Frames with bounding boxes
    ├── heatmaps/            # Heatmap overlays
    ├── escalation_frames/   # Copies of frames that triggered escalation events
    ├── metrics.json         # Per-frame metrics
    ├── summary.json         # Batch summary
    ├── event_timeline.json  # Escalation events
    ├── crowd_analysis.db    # SQLite: frames, persons, keypoints
    ├── crowd_density_trend.png
    └── agitation_index_trend.png
```

## Setup

1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install PyTorch 2.1.2 (CPU)**
   ```bash
   pip install torch==2.1.2 torchvision==0.16.2
   ```

3. **Build and install Detectron2 from source**
   Install Visual Studio Build Tools (C++), then:
   ```bash
   pip install "git+https://github.com/facebookresearch/detectron2.git@fd27788"
   ```

4. **Install remaining dependencies**
   ```bash
   pip install numpy==1.26.4 opencv-python==4.8.1.78 scipy matplotlib tqdm
   ```
   (or `pip install -r requirements.txt`)

## Detection backends

Two interchangeable backends, selected via `--backend`, `CROWD_BACKEND`, or the dashboard run form:

| | `detectron2` (Phase I baseline) | `yolo` (Phase II) |
|---|---|---|
| Models | Faster R-CNN + Keypoint R-CNN (two passes) | YOLO11-pose, single pass |
| Tracking | IoU + Hungarian (`tracker.py`) | ByteTrack / BoT-SORT (`model.track`) |
| CPU speed (sample images) | ~10.5 s/frame | ~0.17 s/frame (~62× faster) |
| Dense-crowd recall | higher (R-CNN, 960 px) | lower with nano weights — CrowdHuman fine-tune planned |
| Keypoint gating | `KEYPOINT_VISIBILITY_THRESH` (logits) | `YOLO_KEYPOINT_CONF` (probabilities) |

YOLO weights auto-download to `crowd_project/models/` on first use (`YOLO_WEIGHTS`, default `yolo11n-pose.pt`; use `yolo11s-pose.pt` for better dense-crowd counts at ~2-3× cost).

## Usage

- **Process a folder of images**
  ```bash
  python main.py
  ```

- **YOLO backend, separate output for comparison**
  ```bash
  python main.py --backend yolo --output-dir output_yolo
  ```

- **Process a video (extract frames then run pipeline)**
  ```bash
  python main.py --video path/to/video.mp4
  ```

- **Custom input folder / device**
  ```bash
  python main.py --input-dir path/to/frames --device cuda
  ```
  (`CROWD_DEVICE=cuda` also works.)

- **Web dashboard** — see `../web_dashboard/README.md` (FastAPI backend + React frontend; upload images/video, configure a run, monitor progress, browse results).

## Pipeline

1. **Pass 1 (per frame):** detection → pose → tracking → density → motion → agitation index. Annotated + heatmap images are written to disk immediately; memory stays flat regardless of batch length.
2. **Pass 2 (batch):** density classification (batch statistics + absolute floors) and the dynamic agitation threshold (mean + kσ — batch-relative, offline-review semantics).
3. **Pass 3 (batch):** escalation events (High Crowd AND agitation above threshold); event frames copied to `escalation_frames/`, keypoints persisted to SQLite for event frames only.

## Configuration

Defaults live in `config.py`; per-run overrides go through `settings.PipelineSettings` (the dashboard passes these from its run form). Key knobs: confidence thresholds, `MAX_INFERENCE_WIDTH`, tracker IoU/max-lost, density sigmas and absolute floors, agitation weights and threshold sigma, heatmap parameters, keypoint visibility threshold.

## Design Notes

- **Modular, OOP:** each engine/analyzer is independent; `FrameProcessor` composes them.
- **Per-run settings object** — no module-global mutation; safe for the dashboard's queued runs.
- **Empty detections:** zero persons never crashes; empty heatmap, count 0.
- **Keypoint gating:** motion only uses keypoints above the visibility threshold in both frames; coasting (undetected) tracks emit no motion.
- **Extensible:** structure allows adding YOLO/ByteTrack backends, homography calibration, or edge deployment later (see Phase II plan).
