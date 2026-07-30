# Stage 2 — CrowdHuman Fine-Tune (head + body detection)

Dense crowds occlude bodies but rarely heads. CrowdHuman annotates both, so a
fine-tuned detector recovers people the COCO-pretrained model misses and gives
a per-person head signal for counting. This directory holds everything needed;
**training runs on a free Colab/Kaggle GPU** (the local machine is CPU-only —
a 30-epoch yolo11s run would take days on CPU vs ~8-16 h on a T4).

Important: this fine-tunes a **detection** model (`yolo11s.pt`), not the pose
model — CrowdHuman has no keypoint labels, and fine-tuning `yolo11n-pose.pt`
on boxes alone would destroy its pose head. In the pipeline the fine-tuned
detector provides counting/density/tracking; `yolo11n-pose` remains the
motion/keypoint model. `yolo_engine.py` auto-detects a `head` class in
whatever weights it is given and reports `head_count` per frame.

## Files

- `download_crowdhuman.py` — pulls the ~19 GB dataset via gdown (`--val-only` for the 3 GB eval subset)
- `convert_crowdhuman.py` — `.odgt` → YOLO labels (person=0 from fbox, head=1 from hbox; ignore regions dropped; hard-links images) + generates `crowdhuman.yaml`
- `train_crowdhuman.py` — fine-tune wrapper (mosaic, close_mosaic, patience preset)
- `eval_detection.py` — mAP@0.5 / mAP@0.5:0.95 per class, fine-tuned vs COCO baseline

## Colab workflow (copy-paste)

```python
# Cell 1 — setup (GPU runtime: Runtime > Change runtime type > T4)
!pip -q install ultralytics gdown
!git clone <your-repo-url> proj   # or upload the training/ folder

# Cell 2 — data (~19 GB download + extract; ~40 GB disk total. Use Colab Pro
# or Kaggle (100 GB) if the free tier runs out; --val-only for eval-only runs)
!python proj/training/download_crowdhuman.py --root /content/crowdhuman
!python proj/training/convert_crowdhuman.py --root /content/crowdhuman

# Cell 3 — train (T4: ~15-30 min/epoch at batch 16, 640px)
!python proj/training/train_crowdhuman.py \
    --data /content/crowdhuman/yolo/crowdhuman.yaml \
    --model yolo11s.pt --epochs 30 --batch 16 --device 0

# Cell 4 — evaluate vs baseline
!python proj/training/eval_detection.py \
    --data /content/crowdhuman/yolo/crowdhuman.yaml \
    --weights runs/crowdhuman/crowdhuman-ft/weights/best.pt yolo11s.pt --device 0

# Cell 5 — save the result to Drive
from google.colab import drive; drive.mount('/content/drive')
!cp runs/crowdhuman/crowdhuman-ft/weights/best.pt \
    /content/drive/MyDrive/crowdhuman_yolo11s.pt
```

Kaggle variant: same commands in a GPU notebook; attach the dataset once via
`kaggle datasets` to skip re-downloading per session.

## Using the fine-tuned weights locally

```powershell
copy crowdhuman_yolo11s.pt "crowd_project\models\"
python crowd_project\main.py --backend yolo --output-dir output_ch   # with:
$env:CROWD_YOLO_WEIGHTS = "crowdhuman_yolo11s.pt"
```

or select the weights file in the dashboard's run form (`yolo_weights`).
With a `head` class present, `metrics.json` rows gain `head_count` — the
occlusion-robust count for the head-vs-body density ablation (novelty #1).

## Report numbers to collect

| Comparison | Metric | Source |
|---|---|---|
| fine-tuned vs COCO yolo11s vs Detectron2 | mAP@0.5 (person, head) | `eval_detection.py` on CrowdHuman val |
| head-count vs body-count vs bbox-density | count MAE stratified by occlusion | hand-labelled clips (Stage 6) |
| MR⁻² (leaderboard metric) | official CrowdHuman toolkit | optional, cite if used |

## Smoke test (no dataset, CPU, ~2 min)

`python training/smoke_test.py` builds a synthetic 4-image odgt dataset,
runs the converter, trains 1 epoch of yolo11n at 320 px, and runs the
pipeline engine with the resulting weights — validates the entire chain
before spending Colab time.
