"""
Event Manager — detects Crowd Escalation Events and persists them.

An event is triggered when:
  - The frame is classified as "High Crowd"  AND
  - AgitationIndex exceeds the dynamic threshold.

Saves ``event_timeline.json`` and copies the already-saved annotated frame
into ``escalation_frames/`` (a file copy, not a pixel buffer — records no
longer carry image arrays).
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import config

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

    def __init__(self, output_dir: Path | None = None) -> None:
        self._events: list[EscalationEvent] = []
        self._output_dir = output_dir if output_dir is not None else config.OUTPUT_DIR
        self._escalation_dir = self._output_dir / "escalation_frames"

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
        annotated_path: Path | str | None = None,
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
            annotated_path: Path to the already-saved annotated frame; copied
                into escalation_frames/ when the event fires.

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
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._events.append(event)

        logger.warning(
            "ESCALATION EVENT - frame=%s  agitation=%.4f  density=%.4f",
            frame_name, agitation_score, density_ratio,
        )

        if annotated_path is not None:
            self._save_escalation_frame(frame_name, Path(annotated_path))

        return event

    def save_events_json(self, output_path: Path | None = None) -> None:
        """Write all accumulated events to JSON."""
        path = output_path or (self._output_dir / config.EVENTS_JSON)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(e) for e in self._events]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %d events to %s", len(self._events), path)

    def _save_escalation_frame(self, frame_name: str, annotated_path: Path) -> None:
        if not annotated_path.is_file():
            logger.warning(
                "Annotated frame missing, cannot save escalation copy: %s",
                annotated_path,
            )
            return
        self._escalation_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(annotated_path, self._escalation_dir / frame_name)
