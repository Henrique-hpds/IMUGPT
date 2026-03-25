"""Video metadata and frame-selection helpers for stage 5.1."""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np


def read_video_metadata(video_path: str | Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(Path(video_path)),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {
            "available": False,
            "video_path": str(Path(video_path).resolve()),
            "reason": "ffprobe_not_found",
        }
    except subprocess.CalledProcessError as exc:
        return {
            "available": False,
            "video_path": str(Path(video_path).resolve()),
            "reason": "ffprobe_failed",
            "stderr": exc.stderr.strip(),
        }

    payload = json.loads(completed.stdout)
    stream = payload.get("streams", [{}])[0]
    format_info = payload.get("format", {})
    fps = _parse_ffprobe_fps(stream.get("avg_frame_rate"))
    num_frames = _parse_optional_int(stream.get("nb_frames"))
    duration_sec = _parse_optional_float(stream.get("duration"))
    if duration_sec is None:
        duration_sec = _parse_optional_float(format_info.get("duration"))
    if num_frames is None and fps is not None and duration_sec is not None:
        num_frames = int(round(fps * duration_sec))

    return {
        "available": True,
        "video_path": str(Path(video_path).resolve()),
        "fps": fps,
        "num_frames": num_frames,
        "duration_sec": duration_sec,
    }


def select_frame_indices(
    num_frames: int,
    fps_original: Optional[float],
    fps_target: int,
) -> Tuple[np.ndarray, Optional[float], np.ndarray]:
    if int(num_frames) <= 0:
        empty = np.empty((0,), dtype=np.int32)
        return empty, None, empty.astype(np.float32)

    if fps_original is None or float(fps_original) <= 0.0:
        indices = np.arange(int(num_frames), dtype=np.int32)
        timestamps = indices.astype(np.float32)
        effective_fps = float(fps_target) if int(fps_target) > 0 else None
        return indices, effective_fps, timestamps

    fps_original = float(fps_original)
    if int(fps_target) <= 0 or fps_original <= float(fps_target):
        indices = np.arange(int(num_frames), dtype=np.int32)
        timestamps = frame_indices_to_timestamps(indices, fps_original)
        return indices, fps_original, timestamps

    step_sec = 1.0 / float(fps_target)
    if int(num_frames) == 1:
        indices = np.asarray([0], dtype=np.int32)
    else:
        duration_sec = float(int(num_frames) - 1) / fps_original
        target_timestamps = np.arange(0.0, duration_sec + (step_sec * 0.5), step_sec, dtype=np.float64)
        if target_timestamps.size == 0:
            target_timestamps = np.asarray([0.0], dtype=np.float64)
        indices = np.rint(target_timestamps * fps_original).astype(np.int32)
        indices = np.clip(indices, 0, int(num_frames) - 1)
        indices = np.unique(indices)

    timestamps = frame_indices_to_timestamps(indices, fps_original)
    return indices, float(fps_target), timestamps


def frame_indices_to_timestamps(
    frame_indices: np.ndarray,
    fps_original: Optional[float],
) -> np.ndarray:
    frame_indices = np.asarray(frame_indices, dtype=np.int32)
    if fps_original is None or float(fps_original) <= 0.0:
        return frame_indices.astype(np.float32)
    return frame_indices.astype(np.float32) / float(fps_original)


def _parse_ffprobe_fps(raw_value: str | None) -> Optional[float]:
    if raw_value is None or raw_value in {"", "0/0"}:
        return None
    try:
        return float(Fraction(str(raw_value)))
    except (ValueError, ZeroDivisionError):
        return None


def _parse_optional_int(raw_value: Any) -> Optional[int]:
    if raw_value in (None, ""):
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def _parse_optional_float(raw_value: Any) -> Optional[float]:
    if raw_value in (None, ""):
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None
