"""
SQLite storage layer for structured crowd-analysis metrics.

Uses Python's built-in ``sqlite3`` -- no extra installation required.

Schema:
    frames   -- one row per processed frame.
    persons  -- one row per tracked person per frame (summary metrics only).
    keypoints -- raw keypoint arrays, stored ONLY for escalation-event frames.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import config

logger = logging.getLogger(__name__)

_CREATE_FRAMES = """
CREATE TABLE IF NOT EXISTS frames (
    frame_id        TEXT PRIMARY KEY,
    people_count    INTEGER,
    density_ratio   REAL,
    agitation_index REAL,
    classification  TEXT
);
"""

_CREATE_PERSONS = """
CREATE TABLE IF NOT EXISTS persons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id    TEXT,
    track_id    INTEGER,
    motion_score REAL,
    centroid_x  REAL,
    centroid_y  REAL,
    bbox_area   REAL,
    FOREIGN KEY (frame_id) REFERENCES frames(frame_id)
);
"""

_CREATE_KEYPOINTS = """
CREATE TABLE IF NOT EXISTS keypoints (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id    TEXT,
    track_id    INTEGER,
    keypoints_json TEXT,
    FOREIGN KEY (frame_id) REFERENCES frames(frame_id)
);
"""


class CrowdDatabase:
    """
    Thin wrapper around a SQLite database for crowd analysis results.

    Normal frames store only summary metrics in ``frames`` and ``persons``.
    Raw keypoints are persisted **only** when an escalation event is triggered,
    via :meth:`insert_keypoints`.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._path = db_path or (config.OUTPUT_DIR / config.DATABASE_FILE)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._init_tables()
        logger.info("Database opened: %s", self._path)

    def _init_tables(self) -> None:
        cur = self._conn.cursor()
        cur.execute(_CREATE_FRAMES)
        cur.execute(_CREATE_PERSONS)
        cur.execute(_CREATE_KEYPOINTS)
        self._conn.commit()

    # ------------------------------------------------------------------
    # inserts
    # ------------------------------------------------------------------
    def insert_frame(
        self,
        frame_id: str,
        people_count: int,
        density_ratio: float,
        agitation_index: float,
        classification: str,
    ) -> None:
        """Upsert a frame row."""
        self._conn.execute(
            """INSERT OR REPLACE INTO frames
               (frame_id, people_count, density_ratio, agitation_index, classification)
               VALUES (?, ?, ?, ?, ?)""",
            (frame_id, people_count, density_ratio, agitation_index, classification),
        )

    def insert_persons(
        self,
        frame_id: str,
        persons: Sequence[dict[str, Any]],
    ) -> None:
        """
        Bulk-insert person rows for a frame.

        Each dict must have: track_id, motion_score, centroid_x, centroid_y, bbox_area.
        """
        self._conn.executemany(
            """INSERT INTO persons
               (frame_id, track_id, motion_score, centroid_x, centroid_y, bbox_area)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                (
                    frame_id,
                    p["track_id"],
                    p.get("motion_score", 0.0),
                    p["centroid_x"],
                    p["centroid_y"],
                    p["bbox_area"],
                )
                for p in persons
            ],
        )

    def insert_keypoints(
        self,
        frame_id: str,
        track_keypoints: Sequence[tuple[int, np.ndarray]],
    ) -> None:
        """
        Store raw keypoint arrays -- called ONLY for escalation-event frames.

        Args:
            frame_id: Frame that triggered the event.
            track_keypoints: List of (track_id, keypoints_array) where
                             keypoints_array is (17, 2) float32.
        """
        self._conn.executemany(
            """INSERT INTO keypoints (frame_id, track_id, keypoints_json)
               VALUES (?, ?, ?)""",
            [
                (frame_id, tid, json.dumps(kps.tolist()))
                for tid, kps in track_keypoints
            ],
        )

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()
        logger.info("Database closed: %s", self._path)
