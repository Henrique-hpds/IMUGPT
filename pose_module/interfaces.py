"""Core interfaces and contracts for the pose pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


COCO_17_JOINT_NAMES: Tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True)
class Pose2DJob:
    clip_id: str
    video_path: str
    fps_target: int
    output_dir: str
    save_debug: bool = True
    device_preference: str = "auto"
    model_alias: str = "vitpose-b"
    detector_category_ids: Tuple[int, ...] = (0,)
    video_fps: Optional[float] = None
    video_num_frames: Optional[int] = None
    video_duration_sec: Optional[float] = None
    raw_prediction_filename: str = "raw_predictions.json"
    debug_overlay_filename: str = "debug_overlay.mp4"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["detector_category_ids"] = [int(value) for value in self.detector_category_ids]
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Pose2DJob":
        return cls(
            clip_id=str(payload["clip_id"]),
            video_path=str(payload["video_path"]),
            fps_target=int(payload["fps_target"]),
            output_dir=str(payload["output_dir"]),
            save_debug=bool(payload.get("save_debug", True)),
            device_preference=str(payload.get("device_preference", "auto")),
            model_alias=str(payload.get("model_alias", "vitpose-b")),
            detector_category_ids=tuple(
                int(value) for value in payload.get("detector_category_ids", [0])
            ),
            video_fps=_optional_float(payload.get("video_fps")),
            video_num_frames=_optional_int(payload.get("video_num_frames")),
            video_duration_sec=_optional_float(payload.get("video_duration_sec")),
            raw_prediction_filename=str(payload.get("raw_prediction_filename", "raw_predictions.json")),
            debug_overlay_filename=str(payload.get("debug_overlay_filename", "debug_overlay.mp4")),
        )

    @property
    def raw_prediction_path(self) -> Path:
        return Path(self.output_dir) / self.raw_prediction_filename

    @property
    def debug_overlay_path(self) -> Path:
        return Path(self.output_dir) / self.debug_overlay_filename


@dataclass(frozen=True)
class Pose2DResult:
    status: str
    effective_fps: Optional[float]
    selected_frame_indices: List[int]
    artifacts: Dict[str, Any]
    quality_report: Dict[str, Any]
    backend: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.effective_fps is not None:
            payload["effective_fps"] = float(self.effective_fps)
        payload["selected_frame_indices"] = [int(value) for value in self.selected_frame_indices]
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Pose2DResult":
        return cls(
            status=str(payload.get("status", "fail")),
            effective_fps=_optional_float(payload.get("effective_fps")),
            selected_frame_indices=[int(value) for value in payload.get("selected_frame_indices", [])],
            artifacts=dict(payload.get("artifacts", {})),
            quality_report=dict(payload.get("quality_report", {})),
            backend=dict(payload.get("backend", {})),
            error=None if payload.get("error") is None else str(payload.get("error")),
        )


@dataclass(frozen=True)
class PoseFrameInstance:
    frame_id: int
    bbox_xyxy: np.ndarray
    bbox_score: float
    keypoints_xy: np.ndarray
    keypoint_scores: np.ndarray


@dataclass(frozen=True)
class PoseSequence2D:
    clip_id: str
    fps: Optional[float]
    fps_original: Optional[float]
    joint_names_2d: Sequence[str]
    keypoints_xy: np.ndarray
    confidence: np.ndarray
    bbox_xywh: np.ndarray
    frame_indices: np.ndarray
    timestamps_sec: np.ndarray
    source: str

    @property
    def num_frames(self) -> int:
        return int(self.keypoints_xy.shape[0])

    def to_npz_payload(self) -> Dict[str, Any]:
        return {
            "clip_id": np.asarray(self.clip_id),
            "joint_names_2d": np.asarray(list(self.joint_names_2d)),
            "keypoints_xy": np.asarray(self.keypoints_xy, dtype=np.float32),
            "confidence": np.asarray(self.confidence, dtype=np.float32),
            "bbox_xywh": np.asarray(self.bbox_xywh, dtype=np.float32),
            "frame_indices": np.asarray(self.frame_indices, dtype=np.int32),
            "timestamps_sec": np.asarray(self.timestamps_sec, dtype=np.float32),
            "fps": np.asarray(-1.0 if self.fps is None else float(self.fps), dtype=np.float32),
            "fps_original": np.asarray(
                -1.0 if self.fps_original is None else float(self.fps_original),
                dtype=np.float32,
            ),
            "num_frames": np.asarray(self.num_frames, dtype=np.int32),
            "source": np.asarray(self.source),
        }


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)
