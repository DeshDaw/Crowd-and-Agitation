"""
Event Manager — detects Crowd Escalation Events and persists them.

An event is triggered when:
  - The frame is classified as "High Crowd"  AND
  - AgitationIndex exceeds the dynamic threshold.

Saves ``events.json`` and copies the annotated frame to ``escalation_frames/``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class EscalationEvent:
    """Record for a single crowd escalation event."""
    frame_name: str
    frame_index: int
    agitation_score: float
    density_ratio: float
    classification: str
    timestamp: str = ""


class EventManager:
    """
    Accumulates escalation events and saves them to disk.
    """

    def __init__(self, output_dir: Path = config.OUTPUT_DIR) -> None:
        self._events: list[EscalationEvent] = []
        self._escalation_dir = output_dir / "escalation_frames"

    @property
    def events(self) -> list[EscalationEvent]:
        return list(self._events)

    def check(
        self,
        frame_name: str,
        frame_index: int,
        agitation_score: float,
        agitation_threshold: float,
        density_classification: str,
        density_ratio: float,
        annotated_image: np.ndarray | None = None,
    ) -> EscalationEvent | None:
        """
        Evaluate whether the current frame constitutes a Crowd Escalation Event.

        Args:
            frame_name: Filename of the frame.
            frame_index: Index in the processing sequence.
            agitation_score: AgitationIndex for this frame.
            agitation_threshold: Dynamic threshold from batch analysis.
            density_classification: "Low Crowd", "Moderate Crowd", or "High Crowd".
            density_ratio: Raw density ratio for this frame.
            annotated_image: Optional annotated image to save on escalation.

        Returns:
            EscalationEvent if triggered, else None.
        """
        is_high = density_classification == "High Crowd"
        is_agitated = agitation_score > agitation_threshold

        if not (is_high and is_agitated):
            return None

        event = EscalationEvent(
            frame_name=frame_name,
            frame_index=frame_index,
            agitation_score=round(agitation_score, 6),
            density_ratio=round(density_ratio, 6),
            classification=density_classification,
        )
        self._events.append(event)

        logger.warning(
            "ESCALATION EVENT - frame=%s  agitation=%.4f  density=%.4f",
            frame_name, agitation_score, density_ratio,
        )

        if annotated_image is not None:
            self._save_escalation_frame(frame_name, annotated_image)

        return event

    def save_events_json(self, output_path: Path | None = None) -> None:
        """Write all accumulated events to JSON."""
        path = output_path or (config.OUTPUT_DIR / config.EVENTS_JSON)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(e) for e in self._events]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %d events to %s", len(self._events), path)

    def _save_escalation_frame(self, frame_name: str, image: np.ndarray) -> None:
        self._escalation_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._escalation_dir / frame_name
        cv2.imwrite(str(out_path), image)
