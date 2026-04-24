"""OpenCV-based video reader/writer wrappers."""
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int
    duration_s: float


def probe_video(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return VideoInfo(fps=fps, width=w, height=h, frame_count=n, duration_s=n / fps if fps > 0 else 0.0)


def make_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """size = (width, height). Uses mp4v codec for broad compatibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {path}")
    return writer
