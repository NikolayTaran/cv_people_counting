"""Video output: raw BGR frames piped into ffmpeg -> H.264 MP4.

Why a pipe instead of cv2.VideoWriter: the H.264 baseline profile produced
this way (yuv420p + faststart) plays in every browser for the web dashboard,
while OpenCV's bundled encoders only reliably produce MPEG-4 Part 2.
Falls back to cv2.VideoWriter (mp4v) when no ffmpeg is available.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


def _find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:  # imageio-ffmpeg ships a static ffmpeg binary
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - optional dependency
        return None


class VideoSink:
    """Write BGR frames to an .mp4 file (H.264 if possible)."""

    def __init__(self, path: str | Path, width: int, height: int, fps: float) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # H.264 needs even dimensions
        self.width = int(width) - (int(width) % 2)
        self.height = int(height) - (int(height) % 2)
        self.fps = float(fps)
        self.frames_written = 0

        self._ffmpeg = _find_ffmpeg()
        if self._ffmpeg:
            cmd = [
                self._ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}",
                "-r", str(self.fps), "-i", "pipe:0",
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(self.path),
            ]
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            self._writer = self._pipe_write
        else:  # pragma: no cover - fallback path
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._cv_writer = cv2.VideoWriter(
                str(self.path), fourcc, self.fps, (self.width, self.height))
            self._writer = self._cv_write

    # ------------------------------------------------------------------ writing
    def write(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        if w != self.width or h != self.height:  # odd-size guard
            frame_bgr = frame_bgr[: self.height, : self.width]
        self._writer(frame_bgr)
        self.frames_written += 1

    def _pipe_write(self, frame_bgr: np.ndarray) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(np.ascontiguousarray(frame_bgr, dtype=np.uint8).tobytes())

    def _cv_write(self, frame_bgr: np.ndarray) -> None:  # pragma: no cover
        self._cv_writer.write(frame_bgr)

    # ------------------------------------------------------------------- closing
    def close(self) -> None:
        if getattr(self, "_proc", None) is not None:
            if self._proc.stdin:
                self._proc.stdin.close()
            stderr = self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else ""
            code = self._proc.wait(timeout=120)
            if code != 0:  # pragma: no cover - surfaced loudly if it happens
                raise RuntimeError(f"ffmpeg exited with {code}: {stderr.strip()}")
            self._proc = None
        elif getattr(self, "_cv_writer", None) is not None:  # pragma: no cover
            self._cv_writer.release()
            self._cv_writer = None

    def __enter__(self) -> "VideoSink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
