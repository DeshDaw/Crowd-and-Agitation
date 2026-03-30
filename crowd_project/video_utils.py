"""
Video and image input utilities.

Load images from a folder or extract frames from video.
"""

import logging
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from config import IMAGE_EXTENSIONS, INPUT_IMAGES_DIR, INPUT_VIDEO_DIR, VIDEO_EXTRACT_FPS

logger = logging.getLogger(__name__)


def get_image_paths(input_dir: Path | None = None) -> list[Path]:
    """
    Collect all supported image paths from a directory, sorted by name.

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
    paths.sort(key=lambda p: p.name)
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

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = 1.0 / (fps or video_fps)
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    frame_index = 0
    last_time = -1.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_index / video_fps
        if t >= last_time + interval:
            name = f"frame_{len(frame_paths):06d}.jpg"
            out_path = output_frames_dir / name
            cv2.imwrite(str(out_path), frame)
            frame_paths.append(out_path)
            last_time = t
        frame_index += 1

    cap.release()
    logger.info("Extracted %d frames from %s to %s", len(frame_paths), video_path, output_frames_dir)
    return frame_paths
