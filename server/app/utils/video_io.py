"""Video I/O wrappers. OpenCV for read/write inside the pipeline; ffmpeg for
input normalization (sane fps metadata) and output transcoding (browser-friendly H.264).
"""
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

FFMPEG = "ffmpeg"


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int
    duration_s: float


def _ffmpeg_available() -> bool:
    return shutil.which(FFMPEG) is not None


def probe_video(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    # Some containers (notably MediaRecorder WebM on Android) report
    # nonsense like 1000 fps because they store 1ms frame intervals. Clamp.
    if fps > 120 or fps < 1:
        fps = 30.0
    return VideoInfo(fps=fps, width=w, height=h, frame_count=n, duration_s=n / fps if fps > 0 else 0.0)


def make_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """size = (width, height). Uses mp4v codec for OpenCV compatibility on Windows;
    output is later transcoded to browser-friendly H.264 via transcode_for_web().
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {path}")
    return writer


def normalize_input(src: Path, dst: Path, target_fps: float = 30.0) -> Path:
    """Re-encode the input video into a normalized H.264 MP4 at a known fps.

    Solves a class of bugs caused by exotic containers (e.g. MediaRecorder WebM
    advertising 1000 fps) that confuse OpenCV's frame-rate probe and break the
    rest of the pipeline. After this call, OpenCV reads `dst` with reliable
    metadata.

    Falls back to a plain copy if ffmpeg is unavailable.
    """
    if not _ffmpeg_available():
        shutil.copy(src, dst)
        return dst

    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-y",
        "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-vsync", "cfr", "-r", f"{target_fps:g}",
        "-an",
        "-movflags", "+faststart",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg normalize_input failed: {proc.stderr[-2000:]}")
    return dst


def transcode_for_web(src: Path, dst: Optional[Path] = None) -> Path:
    """Transcode an OpenCV-written mp4v file into browser-friendly H.264.

    Writes to a sibling .h264.mp4 by default, atomically replacing src with
    the transcoded version (so callers don't need to know about the swap).
    """
    if not _ffmpeg_available():
        return src

    final = dst if dst is not None else src.with_suffix(".h264.mp4")
    cmd = [
        FFMPEG, "-y",
        "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-profile:v", "baseline", "-level", "3.1",
        "-movflags", "+faststart",
        "-an",
        str(final),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg transcode_for_web failed: {proc.stderr[-2000:]}")

    if dst is None:
        # Atomically swap the transcoded file in place of src
        src.unlink(missing_ok=True)
        final.rename(src)
        return src
    return final
