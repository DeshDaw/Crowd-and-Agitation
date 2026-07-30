"""
Video and image input utilities.

Load images from a folder or extract frames from video.
"""

import logging
import re
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .config import IMAGE_EXTENSIONS, INPUT_IMAGES_DIR, VIDEO_EXTRACT_FPS

logger = logging.getLogger(__name__)

_DIGIT_RE = re.compile(r"(\d+)")


def _natural_key(name: str) -> tuple:
    """Sort key treating digit runs numerically ('img9' < 'img10')."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in _DIGIT_RE.split(name)
    )


def get_image_paths(input_dir: Path | None = None) -> list[Path]:
    """
    Collect all supported image paths from a directory, in natural sort order
    (numeric filename parts compared numerically, so frame_9 < frame_10).

    Args:
        input_dir: Folder to scan. Defaults to config INPUT_IMAGES_DIR.

    Returns:
        Sorted list of Paths to image files.
    """
    folder = input_dir or INPUT_IMAGES_DIR
    if not folder.is_dir():
        logger.warning("Input directory does not exist: %s", folder)
        return []

    paths = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths.sort(key=lambda p: _natural_key(p.name))
    logger.info("Found %d images in %s", len(paths), folder)
    return paths


def load_image(path: Path) -> np.ndarray | None:
    """
    Load a single image as BGR.

    Args:
        path: Path to image file.

    Returns:
        BGR image (H, W, 3) or None if load fails.
    """
    img = cv2.imread(str(path))
    if img is None:
        logger.warning("Failed to load image: %s", path)
        return None
    return img


def iter_images_from_folder(
    input_dir: Path | None = None,
) -> Iterator[tuple[Path, np.ndarray]]:
    """
    Yield (path, image) for each image in the input folder.
    Skips failed loads.

    Args:
        input_dir: Folder to scan. Defaults to config INPUT_IMAGES_DIR.

    Yields:
        (path, BGR image).
    """
    for path in get_image_paths(input_dir):
        img = load_image(path)
        if img is not None:
            yield path, img


def get_video_fps(video_path: Path) -> float | None:
    """Native frame rate of a video, or None if unreadable/unreported."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps and fps > 0 else None


def extract_frames_from_video(
    video_path: Path,
    output_frames_dir: Path,
    fps: float | None = VIDEO_EXTRACT_FPS,
) -> list[Path]:
    """
    Extract frames from a video file and save as images in output_frames_dir.
    If fps is None, uses the video's FPS (every frame).

    Args:
        video_path: Path to video file.
        output_frames_dir: Directory to write frame images.
        fps: Target FPS for extraction; None = use video FPS.

    Returns:
        List of paths to saved frame images, in order.
    """
    if not video_path.is_file():
        logger.error("Video file not found: %s", video_path)
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Could not open video: %s", video_path)
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        logger.warning(
            "Video reports no FPS (%s); assuming 25.0 — extraction timing "
            "may be inaccurate: %s", video_fps, video_path,
        )
        video_fps = 25.0
    interval = 1.0 / (fps or video_fps)
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    frame_index = 0
    # Start one interval in the past so the frame at t=0 is always taken,
    # regardless of how the requested fps compares to the video fps.
    last_time = -interval
    # Epsilon absorbs IEEE-754 ties when fps == video_fps (extract-every-frame),
    # where t and last_time + interval are mathematically equal.
    eps = 1e-6

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_index / video_fps
        if t >= last_time + interval - eps:
            name = f"frame_{len(frame_paths):06d}.jpg"
            out_path = output_frames_dir / name
            cv2.imwrite(str(out_path), frame)
            frame_paths.append(out_path)
            last_time = t
        frame_index += 1

    cap.release()
    logger.info("Extracted %d frames from %s to %s", len(frame_paths), video_path, output_frames_dir)
    return frame_paths
