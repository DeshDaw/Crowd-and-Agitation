"""
Camera calibration — image plane to ground plane via homography.

Four clicked image points, corresponding to the corners of a real-world
rectangle of known dimensions (metres), define a perspective transform.
Projected through it, bbox bottom-centres become metric ground positions:
density becomes persons/m², track displacement becomes m/s.

Point order convention: top-left, top-right, bottom-right, bottom-left of
the ground rectangle AS SEEN IN THE IMAGE, mapping to world coordinates
(0,0), (W,0), (W,H), (0,H) in metres.

Run as a script for the interactive click tool:
    python -m crowd_project.calibration --image frame.jpg \
        --width-m 5.0 --height-m 8.0 --out calibration.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraCalibration:
    """Homography from image pixels to ground-plane metres."""

    image_points: list[list[float]]     # 4 x [x, y] pixels (TL, TR, BR, BL)
    width_m: float                      # world rectangle width  (TL -> TR)
    height_m: float                     # world rectangle height (TL -> BL)
    image_size: list[int] = field(default_factory=lambda: [0, 0])  # [w, h] of the calibrated frame
    _H: np.ndarray | None = None        # lazily built 3x3 homography

    @property
    def area_m2(self) -> float:
        return self.width_m * self.height_m

    @property
    def homography(self) -> np.ndarray:
        if self._H is None:
            src = np.array(self.image_points, dtype=np.float32)
            dst = np.array(
                [
                    [0.0, 0.0],
                    [self.width_m, 0.0],
                    [self.width_m, self.height_m],
                    [0.0, self.height_m],
                ],
                dtype=np.float32,
            )
            self._H = cv2.getPerspectiveTransform(src, dst)
        return self._H

    def validate(self) -> None:
        if len(self.image_points) != 4:
            raise ValueError("calibration needs exactly 4 image points")
        if self.width_m <= 0 or self.height_m <= 0:
            raise ValueError("world dimensions must be positive metres")
        # Degenerate quads (collinear points) make getPerspectiveTransform
        # blow up or produce garbage — reject via area check.
        pts = np.array(self.image_points, dtype=np.float64)
        area = 0.5 * abs(
            np.dot(pts[:, 0], np.roll(pts[:, 1], -1))
            - np.dot(pts[:, 1], np.roll(pts[:, 0], -1))
        )
        if area < 100.0:
            raise ValueError("calibration points are (near-)degenerate")

    def image_to_ground(self, points: np.ndarray) -> np.ndarray:
        """
        Project image points to ground metres.

        Args:
            points: (N, 2) pixel coordinates.

        Returns:
            (N, 2) ground coordinates in metres.
        """
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.homography)
        return out.reshape(-1, 2)

    def in_region(self, ground_points: np.ndarray) -> np.ndarray:
        """Boolean mask: which ground points fall inside the calibrated rectangle."""
        if len(ground_points) == 0:
            return np.zeros((0,), dtype=bool)
        g = np.asarray(ground_points)
        return (
            (g[:, 0] >= 0.0) & (g[:, 0] <= self.width_m)
            & (g[:, 1] >= 0.0) & (g[:, 1] <= self.height_m)
        )

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        self.validate()
        data = asdict(self)
        data.pop("_H", None)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Calibration saved: %s", path)

    @classmethod
    def load(cls, path: Path) -> "CameraCalibration":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        calib = cls(
            image_points=data["image_points"],
            width_m=float(data["width_m"]),
            height_m=float(data["height_m"]),
            image_size=data.get("image_size", [0, 0]),
        )
        calib.validate()
        return calib


# =====================================================================
# Interactive click-4-points tool
# =====================================================================

def run_click_tool(image_path: Path, width_m: float, height_m: float, out: Path) -> None:
    """OpenCV window: click TL, TR, BR, BL of a known ground rectangle."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Cannot read image: {image_path}")

    points: list[list[float]] = []
    labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
    win = "calibration — click 4 ground points (r = reset, q = abort)"

    def redraw() -> np.ndarray:
        canvas = img.copy()
        for i, p in enumerate(points):
            cv2.circle(canvas, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1)
            cv2.putText(canvas, labels[i], (int(p[0]) + 8, int(p[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(points) == 4:
            cv2.polylines(canvas, [np.array(points, dtype=np.int32)], True, (0, 255, 0), 2)
        nxt = labels[len(points)] if len(points) < 4 else "ENTER to save"
        cv2.putText(canvas, f"next: {nxt}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return canvas

    def on_mouse(event: int, x: int, y: int, *_: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([float(x), float(y)])

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        cv2.imshow(win, redraw())
        key = cv2.waitKey(30) & 0xFF
        if key == ord("r"):
            points.clear()
        elif key == ord("q") or key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("aborted")
        elif key in (13, 10) and len(points) == 4:
            break
    cv2.destroyAllWindows()

    h, w = img.shape[:2]
    calib = CameraCalibration(
        image_points=points, width_m=width_m, height_m=height_m, image_size=[w, h],
    )
    calib.save(out)
    print(f"saved {out}  (region {width_m} x {height_m} m = {calib.area_m2:.1f} m²)")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--width-m", type=float, required=True,
                    help="Real-world width of the rectangle (TL->TR), metres")
    ap.add_argument("--height-m", type=float, required=True,
                    help="Real-world height of the rectangle (TL->BL), metres")
    ap.add_argument("--out", type=Path, default=Path("calibration.json"))
    args = ap.parse_args()
    run_click_tool(args.image, args.width_m, args.height_m, args.out)


if __name__ == "__main__":
    main()
