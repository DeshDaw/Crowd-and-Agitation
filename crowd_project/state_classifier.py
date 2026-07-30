"""
Crowd State Classifier — Calm / Restless / Agitated.

Two implementations behind one interface:

- :class:`HeuristicStateClassifier` — bootstrap rules on the physics signals
  (crowd pressure, acceleration events, directional entropy, Phase I
  agitation index). Used until a trained model exists, and to pre-label
  footage in the labeling tool.
- :class:`MLPStateClassifier` — small PyTorch MLP over windowed features,
  trained with ``training/train_state_classifier.py``. Auto-loaded by the
  pipeline from ``crowd_project/models/state_classifier.pt`` when present.

Features are aggregated over sliding windows of ~1 second (``window_frames``
= round(source_fps), min 1): the mean and max of each per-frame signal.
Missing signals (uncalibrated runs) are fed as zeros — the heuristic then
degrades to agitation-index-only, and a model trained with calibration
should not be applied to uncalibrated runs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import config

logger = logging.getLogger(__name__)

STATES = ["Calm", "Restless", "Agitated"]

# Per-frame record keys used as classifier inputs, in canonical order.
FEATURE_KEYS = [
    "people_count",
    "persons_per_m2",
    "mean_speed",
    "speed_std",
    "crowd_pressure",
    "directional_entropy",
    "accel_event_rate",
    "agitation_index",
    "density_ratio",
]


def frame_feature_vector(record: dict[str, Any]) -> np.ndarray:
    """Per-frame raw feature vector; missing/None values become 0."""
    vals = []
    for k in FEATURE_KEYS:
        v = record.get(k)
        vals.append(float(v) if v is not None else 0.0)
    return np.array(vals, dtype=np.float32)


def window_features(
    records: Sequence[dict[str, Any]],
    window_frames: int,
) -> tuple[np.ndarray, list[int]]:
    """
    Sliding non-overlapping windows over frame records.

    Returns:
        (features, window_starts) where features is (n_windows, 2*len(FEATURE_KEYS))
        — per-window mean and max of each per-frame feature — and
        window_starts holds the first frame index of each window.
    """
    window_frames = max(1, int(window_frames))
    rows: list[np.ndarray] = []
    starts: list[int] = []
    for i in range(0, len(records), window_frames):
        chunk = records[i:i + window_frames]
        if not chunk:
            continue
        m = np.stack([frame_feature_vector(r) for r in chunk])
        rows.append(np.concatenate([m.mean(axis=0), m.max(axis=0)]))
        starts.append(int(chunk[0].get("frame_index", i)))
    if not rows:
        return np.empty((0, 2 * len(FEATURE_KEYS)), dtype=np.float32), []
    return np.stack(rows).astype(np.float32), starts


class HeuristicStateClassifier:
    """
    Threshold rules on the physics signals — a defensible bootstrap, not a
    learned model. Signals vote; two strong votes escalate to Agitated,
    one to Restless.
    """

    name = "heuristic"

    def classify_frame(self, record: dict[str, Any]) -> str:
        pressure = record.get("crowd_pressure") or 0.0
        accel = record.get("accel_event_rate") or 0.0
        entropy = record.get("directional_entropy") or 0.0
        agitation = record.get("agitation_index") or 0.0

        strong = 0
        mild = 0
        for value, (mild_t, strong_t) in (
            (pressure, config.STATE_PRESSURE_THRESHOLDS),
            (accel, config.STATE_ACCEL_THRESHOLDS),
            (agitation, config.STATE_AGITATION_THRESHOLDS),
        ):
            if value >= strong_t:
                strong += 1
            elif value >= mild_t:
                mild += 1
        # Entropy alone never escalates — chaotic-but-slow milling is normal —
        # but it can push a borderline frame up one level.
        entropy_boost = entropy >= config.STATE_ENTROPY_BOOST

        if strong >= 2 or (strong == 1 and (mild >= 1 or entropy_boost)):
            return "Agitated"
        if strong == 1 or mild >= 1 or (mild == 0 and entropy_boost and pressure > 0):
            return "Restless"
        return "Calm"


class MLPStateClassifier:
    """
    Small MLP loaded from a checkpoint saved by train_state_classifier.py.

    Checkpoint format (torch.save dict):
        state_dict, feature_mean, feature_std, feature_keys, states,
        window_frames_hint
    """

    name = "mlp"

    def __init__(self, model_path: Path) -> None:
        import torch
        from torch import nn

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        if ckpt.get("feature_keys") != FEATURE_KEYS:
            raise ValueError(
                "state_classifier checkpoint feature set does not match "
                f"current FEATURE_KEYS (got {ckpt.get('feature_keys')})"
            )
        dim = 2 * len(FEATURE_KEYS)
        self._net = nn.Sequential(
            nn.Linear(dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, len(STATES)),
        )
        self._net.load_state_dict(ckpt["state_dict"])
        self._net.eval()
        self._mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
        self._std = np.asarray(ckpt["feature_std"], dtype=np.float32)
        self._std[self._std == 0] = 1.0
        self._torch = torch
        logger.info("MLP state classifier loaded: %s", model_path)

    def classify_windows(self, features: np.ndarray) -> list[str]:
        """features: (n_windows, 2*len(FEATURE_KEYS)) raw (unnormalized)."""
        if len(features) == 0:
            return []
        x = (features - self._mean) / self._std
        with self._torch.no_grad():
            logits = self._net(self._torch.from_numpy(x))
            idx = logits.argmax(dim=1).tolist()
        return [STATES[i] for i in idx]


def build_mlp(dim: int):
    """The training-side twin of MLPStateClassifier's architecture."""
    from torch import nn

    return nn.Sequential(
        nn.Linear(dim, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, len(STATES)),
    )


def default_model_path() -> Path:
    return config.PROJECT_ROOT / "models" / "state_classifier.pt"


def classify_batch(
    records: Sequence[dict[str, Any]],
    source_fps: float | None,
    model_path: Path | None = None,
) -> tuple[list[str], str]:
    """
    Assign a crowd state to every frame record.

    Uses the trained MLP (windowed) when a checkpoint exists, else the
    per-frame heuristic. Returns (state_per_frame, classifier_name).
    """
    path = model_path if model_path is not None else default_model_path()
    window = max(1, round(source_fps)) if source_fps and source_fps > 0 else 5

    if path.is_file():
        try:
            clf = MLPStateClassifier(path)
            feats, _ = window_features(records, window)
            per_window = clf.classify_windows(feats)
            states: list[str] = []
            for i in range(len(records)):
                w = min(i // window, len(per_window) - 1) if per_window else 0
                states.append(per_window[w] if per_window else "Calm")
            return states, clf.name
        except Exception:  # noqa: BLE001 — a bad checkpoint must not kill the run
            logger.exception("MLP state classifier failed; using heuristic")

    heur = HeuristicStateClassifier()
    return [heur.classify_frame(r) for r in records], heur.name


def save_checkpoint(
    path: Path,
    net: Any,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    window_frames: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist a trained classifier in the format MLPStateClassifier loads."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "state_dict": net.state_dict(),
        "feature_mean": np.asarray(feature_mean, dtype=np.float32),
        "feature_std": np.asarray(feature_std, dtype=np.float32),
        "feature_keys": FEATURE_KEYS,
        "states": STATES,
        "window_frames_hint": window_frames,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    # Human-readable sidecar for the report
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_keys": FEATURE_KEYS,
                "states": STATES,
                "window_frames_hint": window_frames,
                **(extra or {}),
            },
            f, indent=2, default=str,
        )
