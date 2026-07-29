"""
Gaussian density heatmap generation from detection bounding box centers.

Uses downscaled grid for CPU efficiency. Overlays heatmap onto original image.
"""

import logging
from typing import Tuple

import cv2
import numpy as np

from .config import (
    HEATMAP_ALPHA,
    HEATMAP_COLORMAP,
    HEATMAP_DOWNSCALE,
    HEATMAP_SIGMA,
)

logger = logging.getLogger(__name__)


def _center_points_from_boxes(boxes: np.ndarray) -> np.ndarray:
    """
    Compute center (x, y) of each box. boxes are (N, 4) in xyxy format.

    Args:
        boxes: (N, 4) array [x1, y1, x2, y2].

    Returns:
        (N, 2) array of (cx, cy) in same coordinate system.
    """
    if len(boxes) == 0:
        return np.empty((0, 2), dtype=np.float32)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    return np.stack([cx, cy], axis=1)


def build_density_heatmap(
    height: int,
    width: int,
    centers: np.ndarray,
    downscale: int = HEATMAP_DOWNSCALE,
    sigma: float = HEATMAP_SIGMA,
) -> np.ndarray:
    """
    Build a Gaussian density heatmap on a downscaled grid, then upsample to (height, width).

    Args:
        height: Original image height.
        width: Original image width.
        centers: (N, 2) array of (x, y) in original image coordinates.
        downscale: Grid is (height//downscale, width//downscale).
        sigma: Gaussian sigma in downscaled grid units.

    Returns:
        Single-channel float32 density map of shape (height, width), normalized to [0, 1].
    """
    if len(centers) == 0:
        return np.zeros((height, width), dtype=np.float32)

    h_small = max(1, height // downscale)
    w_small = max(1, width // downscale)
    scale_x = (w_small - 1) / max(1, width - 1)
    scale_y = (h_small - 1) / max(1, height - 1)

    # Map centers to small grid coordinates
    cx = (centers[:, 0] * scale_x).astype(np.int32)
    cy = (centers[:, 1] * scale_y).astype(np.int32)
    cx = np.clip(cx, 0, w_small - 1)
    cy = np.clip(cy, 0, h_small - 1)

    # Accumulate Gaussians on small grid (CPU-efficient)
    grid = np.zeros((h_small, w_small), dtype=np.float32)
    k_radius = max(2, int(round(3 * sigma)))
    k_size = 2 * k_radius + 1
    y_k = np.arange(k_size, dtype=np.float32) - k_radius
    x_k = np.arange(k_size, dtype=np.float32) - k_radius
    yy, xx = np.meshgrid(y_k, x_k, indexing="ij")
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    for i in range(len(centers)):
        cy_i, cx_i = cy[i], cx[i]
        y0 = max(0, cy_i - k_radius)
        y1 = min(h_small, cy_i + k_radius + 1)
        x0 = max(0, cx_i - k_radius)
        x1 = min(w_small, cx_i + k_radius + 1)
        ky0 = y0 - (cy_i - k_radius)
        ky1 = ky0 + (y1 - y0)
        kx0 = x0 - (cx_i - k_radius)
        kx1 = kx0 + (x1 - x0)
        grid[y0:y1, x0:x1] += kernel[ky0:ky1, kx0:kx1]

    # Upsample to original size
    heatmap = cv2.resize(grid, (width, height), interpolation=cv2.INTER_LINEAR)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap.astype(np.float32)


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    colormap: int = HEATMAP_COLORMAP,
    alpha: float = HEATMAP_ALPHA,
) -> np.ndarray:
    """
    Overlay a [0,1] single-channel heatmap onto a BGR image.

    Args:
        image: BGR image (H, W, 3).
        heatmap: Single-channel float [0, 1], same H, W.
        colormap: OpenCV colormap (e.g. cv2.COLORMAP_JET = 2).
        alpha: Blend factor for heatmap (0=only image, 1=only heatmap).

    Returns:
        BGR image with heatmap overlaid.
    """
    heatmap_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    out = cv2.addWeighted(image, 1.0 - alpha, heatmap_bgr, alpha, 0)
    return out


def generate_heatmap_visualization(
    image: np.ndarray,
    boxes: np.ndarray,
    downscale: int = HEATMAP_DOWNSCALE,
    sigma: float = HEATMAP_SIGMA,
    colormap: int = HEATMAP_COLORMAP,
    alpha: float = HEATMAP_ALPHA,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate raw density heatmap and overlay visualization from detection boxes.

    Handles empty detections safely (returns zeros and original image).

    Args:
        image: BGR image (H, W, 3).
        boxes: (N, 4) xyxy bounding boxes.
        downscale: Heatmap grid downscale factor.
        sigma: Gaussian sigma.
        colormap: OpenCV colormap for overlay.
        alpha: Overlay blend.

    Returns:
        heatmap: (H, W) float32 density map [0, 1].
        overlay: BGR image with heatmap overlaid.
    """
    height, width = image.shape[:2]
    centers = _center_points_from_boxes(boxes)
    heatmap = build_density_heatmap(height, width, centers, downscale=downscale, sigma=sigma)
    overlay = overlay_heatmap_on_image(image, heatmap, colormap=colormap, alpha=alpha)
    return heatmap, overlay
