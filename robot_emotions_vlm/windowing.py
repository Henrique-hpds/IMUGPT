"""Shared window partitioning helpers for RobotEmotions VLM flows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pose_module.interfaces import PoseSequence3D


@dataclass(frozen=True)
class PoseManifestEntry:
    """Minimal pose3d-manifest row consumed by windowed flows."""

    clip_id: str
    domain: str | None
    user_id: int | None
    tag_number: int | None
    take_id: str | None
    labels: dict[str, Any]
    source: dict[str, Any]
    video: dict[str, Any]
    pose3d_npz_path: str

    @property
    def source_video_path(self) -> str | None:
        value = self.source.get("video_path")
        return None if value in (None, "") else str(value)

    @property
    def source_rel_dir(self) -> str:
        value = self.source.get("source_rel_dir")
        return "" if value in (None, "") else str(value)


@dataclass(frozen=True)
class WindowSpec:
    """Temporal window carved out of a real pose capture."""

    window_index: int
    prompt_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    source_start_index: int
    source_end_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_index": int(self.window_index),
            "prompt_id": str(self.prompt_id),
            "start_sec": float(self.start_sec),
            "end_sec": float(self.end_sec),
            "duration_sec": float(self.duration_sec),
            "source_start_index": int(self.source_start_index),
            "source_end_index": int(self.source_end_index),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, prompt_id: str | None = None) -> "WindowSpec":
        resolved_prompt_id = str(payload.get("prompt_id", "")).strip()
        if prompt_id is not None:
            resolved_prompt_id = str(prompt_id)
        if not resolved_prompt_id:
            raise ValueError("WindowSpec requires a non-empty prompt_id.")
        return cls(
            window_index=int(payload["window_index"]),
            prompt_id=resolved_prompt_id,
            start_sec=float(payload["start_sec"]),
            end_sec=float(payload["end_sec"]),
            duration_sec=float(payload["duration_sec"]),
            source_start_index=int(payload["source_start_index"]),
            source_end_index=int(payload["source_end_index"]),
        )


def load_pose_manifest_entries(path: str | Path) -> list[PoseManifestEntry]:
    """Load usable pose3d manifest rows that expose a pose3d NPZ path."""

    entries: list[PoseManifestEntry] = []
    manifest_path = Path(path)
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        artifacts = dict(payload.get("artifacts") or {})
        pose3d_npz_path = artifacts.get("pose3d_npz_path")
        status = str(payload.get("status", "")).lower()
        if status not in {"ok", "warning"}:
            continue
        if pose3d_npz_path in (None, ""):
            continue
        entries.append(
            PoseManifestEntry(
                clip_id=str(payload["clip_id"]),
                domain=_optional_string(payload.get("domain")),
                user_id=_optional_int(payload.get("user_id")),
                tag_number=_optional_int(payload.get("tag_number")),
                take_id=_optional_string(payload.get("take_id")),
                labels=dict(payload.get("labels") or {}),
                source=dict(payload.get("source") or {}),
                video=dict(payload.get("video") or {}),
                pose3d_npz_path=str(pose3d_npz_path),
            )
        )
    return entries


def select_pose_entries(
    entries: Sequence[PoseManifestEntry],
    *,
    clip_ids: Sequence[str] | None,
) -> list[PoseManifestEntry]:
    """Filter pose manifest rows by clip id when requested."""

    selected = list(entries)
    if clip_ids is None:
        return selected

    requested = {str(clip_id) for clip_id in clip_ids}
    selected = [entry for entry in selected if entry.clip_id in requested]
    found = {entry.clip_id for entry in selected}
    missing = sorted(requested.difference(found))
    if missing:
        raise ValueError(f"Unknown clip_id values requested in pose3d manifest: {missing}")
    return selected


def load_pose_sequence3d(path: str | Path) -> PoseSequence3D:
    """Read a serialized PoseSequence3D NPZ."""

    with np.load(Path(path), allow_pickle=False) as payload:
        return PoseSequence3D.from_npz_payload({key: payload[key] for key in payload.files})


def resolve_source_times(sequence: PoseSequence3D) -> np.ndarray:
    """Resolve monotonic per-frame times for the pose sequence."""

    num_frames = int(sequence.num_frames)
    if num_frames <= 0:
        raise ValueError("PoseSequence3D must contain at least one frame.")

    timestamps = np.asarray(sequence.timestamps_sec, dtype=np.float32)
    if timestamps.shape == (num_frames,) and np.isfinite(timestamps).all():
        relative = timestamps - timestamps[0]
        if np.all(np.diff(relative) >= -1e-5):
            return relative.astype(np.float32, copy=False)

    fps_candidates = [sequence.fps, sequence.fps_original]
    fps = next((float(value) for value in fps_candidates if value is not None and float(value) > 0.0), None)
    if fps is None:
        raise ValueError("PoseSequence3D must provide either valid timestamps_sec or fps/fps_original.")
    return (np.arange(num_frames, dtype=np.float32) / np.float32(fps)).astype(np.float32, copy=False)


def build_windows(
    *,
    clip_id: str,
    source_times: np.ndarray,
    window_sec: float,
    window_hop_sec: float,
    max_windows_per_clip: int | None,
) -> list[WindowSpec]:
    """Partition the clip into short temporal windows."""

    if source_times.size <= 0:
        return []

    total_duration = _resolve_total_duration(source_times)
    if total_duration <= 0.0:
        return [
            WindowSpec(
                window_index=0,
                prompt_id=f"{clip_id}__w000",
                start_sec=0.0,
                end_sec=0.0,
                duration_sec=1e-6,
                source_start_index=0,
                source_end_index=1,
            )
        ]

    if total_duration <= float(window_sec):
        starts = [0.0]
    else:
        last_start = max(0.0, total_duration - float(window_sec))
        starts = list(np.arange(0.0, last_start + 1e-6, float(window_hop_sec), dtype=np.float32))
        if abs(float(starts[-1]) - float(last_start)) > 1e-4:
            starts.append(float(last_start))

    windows: list[WindowSpec] = []
    for window_index, start_sec in enumerate(starts):
        end_sec = min(float(start_sec) + float(window_sec), total_duration)
        start_index = int(np.searchsorted(source_times, float(start_sec), side="left"))
        end_index = int(np.searchsorted(source_times, float(end_sec), side="left"))
        if end_index <= start_index:
            end_index = min(int(source_times.size), start_index + 1)
        start_sec_value = round(float(start_sec), 6)
        end_sec_value = round(float(end_sec), 6)
        duration_sec_value = round(max(float(end_sec - start_sec), 1e-6), 6)
        windows.append(
            WindowSpec(
                window_index=window_index,
                prompt_id=f"{clip_id}__w{window_index:03d}",
                start_sec=start_sec_value,
                end_sec=end_sec_value,
                duration_sec=duration_sec_value,
                source_start_index=int(start_index),
                source_end_index=int(end_index),
            )
        )
        if max_windows_per_clip is not None and len(windows) >= int(max_windows_per_clip):
            break
    return windows


def resolve_sequence_fps(sequence: PoseSequence3D, source_times: np.ndarray) -> float:
    """Resolve the effective FPS from the sequence contract."""

    if sequence.fps is not None and float(sequence.fps) > 0.0:
        return float(sequence.fps)
    if sequence.fps_original is not None and float(sequence.fps_original) > 0.0:
        return float(sequence.fps_original)
    if source_times.size <= 1:
        return 0.0
    diffs = np.diff(source_times)
    positive_diffs = diffs[diffs > 1e-6]
    if positive_diffs.size == 0:
        return 0.0
    return float(1.0 / np.median(positive_diffs))


def _resolve_total_duration(source_times: np.ndarray) -> float:
    if source_times.size == 1:
        return 0.0
    diffs = np.diff(source_times)
    positive_diffs = diffs[diffs > 1e-6]
    if positive_diffs.size == 0:
        return float(source_times[-1] - source_times[0])
    return float(source_times[-1] + np.median(positive_diffs))


def _optional_string(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
