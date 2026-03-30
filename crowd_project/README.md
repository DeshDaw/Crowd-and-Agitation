# Frame-Based Crowd Detection and Density Estimation System

Research-oriented prototype using **Detectron2** (Faster R-CNN R50-FPN, COCO) for **person-only** detection, with per-frame counting, bounding boxes, Gaussian density heatmaps, and batch analytics. Designed for CPU-first with an easy switch to CUDA (e.g. future RTX 5060).

## Environment

- **OS:** Windows 11  
- **Python:** 3.10  
- **PyTorch:** 2.1.2 (CPU-only by default)  
- **Detectron2:** built from source  
- **NumPy:** 1.26.4  
- **OpenCV:** 4.8.1.78  

## Project Structure

```
crowd_project/
├── config.py          # All paths and parameters (no hardcoded paths)
├── detector.py        # Detectron2 model loading and person inference
├── heatmap.py         # Gaussian density heatmap from bbox centers
├── analytics.py       # Per-frame metrics, moving average, summary, trend plot
├── video_utils.py     # Image loading, video frame extraction
├── main.py            # Entry point: process frames, save outputs
├── requirements.txt
├── README.md
├── input/
│   ├── images/        # Put images here (or extracted frames)
│   └── video/         # Optional: place videos here
└── output/
    ├── annotated/     # Frames with bounding boxes
    ├── heatmaps/      # Heatmap overlays
    ├── metrics.json   # Per-frame metrics
    ├── summary.json   # Batch summary (peak, mean, std, trend)
    └── crowd_trend.png
```

## Setup

1. **Create a virtual environment (recommended)**  
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install PyTorch 2.1.2 (CPU)**  
   ```bash
   pip install torch==2.1.2 torchvision==0.16.2
   ```

3. **Build and install Detectron2 from source**  
   - Install Visual Studio Build Tools (C++).  
   - From project root:  
   ```bash
   pip install "git+https://github.com/facebookresearch/detectron2.git"
   ```  
   Or clone the repo and `pip install -e .` (see [Detectron2 install](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)).

4. **Install remaining dependencies**  
   ```bash
   pip install numpy==1.26.4 opencv-python==4.8.1.78 matplotlib tqdm
   ```

## Usage

- **Process a folder of images**  
  Place images in `input/images/`, then:

  ```bash
  python main.py
  ```

- **Process a video (extract frames then run pipeline)**  
  ```bash
  python main.py --video path/to/video.mp4
  ```  
  Frames are extracted to `input/images/extracted_<videoname>/`, then processed.

- **Use a custom input folder**  
  ```bash
  python main.py --input-dir path/to/my/frames
  ```

- **Switch to GPU (when available)**  
  Set env before running:  
  ```bash
  set CROWD_DEVICE=cuda
  python main.py
  ```  
  Or change `DEVICE` in `config.py` to `"cuda"`.

## Outputs

- **annotated/** – Each frame with person bounding boxes and confidence.  
- **heatmaps/** – Gaussian density heatmap overlaid on each frame.  
- **metrics.json** – Per-frame: `frame_name`, `people_count`, `inference_time`, `average_confidence`.  
- **summary.json** – Peak frame, average/std crowd count, moving average trend.  
- **crowd_trend.png** – Plot of raw count and moving average over frames.  
- Console summary at the end of the run.

## Configuration

All parameters are in `config.py`:

- Paths: `INPUT_IMAGES_DIR`, `OUTPUT_*`, file names.  
- Device: `DEVICE` (`"cpu"` / `"cuda"`).  
- Model: `MODEL_CONFIG`, `MODEL_WEIGHTS`, `CONFIDENCE_THRESHOLD`.  
- Inference: `MAX_INFERENCE_WIDTH` (e.g. 960 for CPU).  
- Heatmap: `HEATMAP_DOWNSCALE`, `HEATMAP_SIGMA`, `HEATMAP_COLORMAP`, `HEATMAP_ALPHA`.  
- Analytics: `MOVING_AVERAGE_WINDOW`.  
- Video: `VIDEO_EXTRACT_FPS` (optional).  
- Logging: `LOG_LEVEL`, `LOG_FORMAT`.  

No hardcoded paths in other modules; use `pathlib` throughout.

## Design Notes

- **Modular, OOP:** config, detector, heatmap, analytics, video_utils are separate.  
- **Type hints and docstrings** in all modules.  
- **Empty detections:** Zero persons in a frame does not crash; heatmap is empty, count is 0.  
- **CPU efficiency:** Resize to `MAX_INFERENCE_WIDTH`, downscaled heatmap grid, minimal device transfers.  
- **Extensible:** Structure allows adding DeepSORT, CSRNet, web dashboard, or PostGIS later.  

**Detectron2 only** – no YOLO.
