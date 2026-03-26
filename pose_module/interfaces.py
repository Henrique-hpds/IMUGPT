"""Core interfaces and contracts for the pose pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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

MOTIONBERT_17_JOINT_NAMES: Tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine",
    "left_knee",
    "right_knee",
    "thorax",
    "left_ankle",
    "right_ankle",
    "neck",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

MOTIONBERT_17_PARENT_INDICES: Tuple[int, ...] = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    9,
    6,
    6,
    11,
    12,
    13,
    14,
)

IMUGPT_22_JOINT_NAMES: Tuple[str, ...] = (
    "Pelvis",
    "Left_hip",
    "Right_hip",
    "Spine1",
    "Left_knee",
    "Right_knee",
    "Spine2",
    "Left_ankle",
    "Right_ankle",
    "Spine3",
    "Left_foot",
    "Right_foot",
    "Neck",
    "Left_collar",
    "Right_collar",
    "Head",
    "Left_shoulder",
    "Right_shoulder",
    "Left_elbow",
    "Right_elbow",
    "Left_wrist",
    "Right_wrist",
)

IMUGPT_22_PARENT_INDICES: Tuple[int, ...] = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
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
class MotionBERTJob:
    clip_id: str
    output_dir: str
    window_size: int = 81
    window_overlap: float = 0.5
    include_confidence: bool = True
    backend_name: str = "mmpose_motionbert"
    checkpoint: Optional[str] = None
    config_path: Optional[str] = None
    device: str = "auto"
    pose2d_source: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_id": str(self.clip_id),
            "output_dir": str(self.output_dir),
            "window_size": int(self.window_size),
            "window_overlap": float(self.window_overlap),
            "include_confidence": bool(self.include_confidence),
            "backend_name": str(self.backend_name),
            "checkpoint": None if self.checkpoint is None else str(self.checkpoint),
            "config_path": None if self.config_path is None else str(self.config_path),
            "device": str(self.device),
            "pose2d_source": None if self.pose2d_source is None else str(self.pose2d_source),
            "image_width": None if self.image_width is None else int(self.image_width),
            "image_height": None if self.image_height is None else int(self.image_height),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MotionBERTJob":
        return cls(
            clip_id=str(payload["clip_id"]),
            output_dir=str(payload["output_dir"]),
            window_size=int(payload.get("window_size", 81)),
            window_overlap=float(payload.get("window_overlap", 0.5)),
            include_confidence=bool(payload.get("include_confidence", True)),
            backend_name=str(payload.get("backend_name", "mmpose_motionbert")),
            checkpoint=None if payload.get("checkpoint") in (None, "") else str(payload.get("checkpoint")),
            config_path=None if payload.get("config_path") in (None, "") else str(payload.get("config_path")),
            device=str(payload.get("device", "auto")),
            pose2d_source=None if payload.get("pose2d_source") in (None, "") else str(payload.get("pose2d_source")),
            image_width=_optional_int(payload.get("image_width")),
            image_height=_optional_int(payload.get("image_height")),
        )

    @property
    def pose3d_npz_path(self) -> Path:
        return Path(self.output_dir) / "pose3d.npz"

    @property
    def input_pose2d_path(self) -> Path:
        return Path(self.output_dir) / "motionbert_input_pose2d.npz"

    @property
    def raw_keypoints_3d_path(self) -> Path:
        return Path(self.output_dir) / "3d_keypoints_raw.npy"

    @property
    def run_report_path(self) -> Path:
        return Path(self.output_dir) / "motionbert_run.json"


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
    observed_mask: Optional[np.ndarray] = None
    imputed_mask: Optional[np.ndarray] = None

    @property
    def num_frames(self) -> int:
        return int(self.keypoints_xy.shape[0])

    @property
    def num_joints(self) -> int:
        return int(self.keypoints_xy.shape[1])

    def resolved_observed_mask(self) -> np.ndarray:
        if self.observed_mask is None:
            return (
                np.isfinite(np.asarray(self.keypoints_xy, dtype=np.float32)).all(axis=2)
                & (np.asarray(self.confidence, dtype=np.float32) > 0.0)
            ).astype(bool, copy=False)
        observed_mask = np.asarray(self.observed_mask, dtype=bool)
        if observed_mask.shape != self.keypoints_xy.shape[:2]:
            raise ValueError(
                "PoseSequence2D observed_mask must match keypoints_xy shape [T, J]: "
                f"got {observed_mask.shape} vs {self.keypoints_xy.shape[:2]}."
            )
        return observed_mask

    def resolved_imputed_mask(self) -> np.ndarray:
        if self.imputed_mask is None:
            return np.zeros(self.keypoints_xy.shape[:2], dtype=bool)
        imputed_mask = np.asarray(self.imputed_mask, dtype=bool)
        if imputed_mask.shape != self.keypoints_xy.shape[:2]:
            raise ValueError(
                "PoseSequence2D imputed_mask must match keypoints_xy shape [T, J]: "
                f"got {imputed_mask.shape} vs {self.keypoints_xy.shape[:2]}."
            )
        return imputed_mask

    def to_npz_payload(self) -> Dict[str, Any]:
        return {
            "clip_id": np.asarray(self.clip_id),
            "joint_names_2d": np.asarray(list(self.joint_names_2d)),
            "keypoints_xy": np.asarray(self.keypoints_xy, dtype=np.float32),
            "confidence": np.asarray(self.confidence, dtype=np.float32),
            "observed_mask": np.asarray(self.resolved_observed_mask(), dtype=bool),
            "imputed_mask": np.asarray(self.resolved_imputed_mask(), dtype=bool),
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

    @classmethod
    def from_npz_payload(cls, payload: Mapping[str, Any]) -> "PoseSequence2D":
        fps = float(np.asarray(payload["fps"]).item())
        fps_original = float(np.asarray(payload["fps_original"]).item())
        return cls(
            clip_id=str(np.asarray(payload["clip_id"]).item()),
            fps=None if fps < 0.0 else fps,
            fps_original=None if fps_original < 0.0 else fps_original,
            joint_names_2d=[str(value) for value in np.asarray(payload["joint_names_2d"]).tolist()],
            keypoints_xy=np.asarray(payload["keypoints_xy"], dtype=np.float32),
            confidence=np.asarray(payload["confidence"], dtype=np.float32),
            observed_mask=(
                np.asarray(payload["observed_mask"], dtype=bool)
                if "observed_mask" in payload
                else None
            ),
            imputed_mask=(
                np.asarray(payload["imputed_mask"], dtype=bool)
                if "imputed_mask" in payload
                else None
            ),
            bbox_xywh=np.asarray(payload["bbox_xywh"], dtype=np.float32),
            frame_indices=np.asarray(payload["frame_indices"], dtype=np.int32),
            timestamps_sec=np.asarray(payload["timestamps_sec"], dtype=np.float32),
            source=str(np.asarray(payload["source"]).item()),
        )


@dataclass(frozen=True)
class PoseSequence3D:
    clip_id: str
    fps: Optional[float]
    fps_original: Optional[float]
    joint_names_3d: Sequence[str]
    joint_positions_xyz: np.ndarray
    joint_confidence: np.ndarray
    skeleton_parents: Sequence[int]
    frame_indices: np.ndarray
    timestamps_sec: np.ndarray
    source: str
    coordinate_space: str = "camera"
    observed_mask: Optional[np.ndarray] = None
    imputed_mask: Optional[np.ndarray] = None

    @property
    def num_frames(self) -> int:
        return int(self.joint_positions_xyz.shape[0])

    @property
    def num_joints(self) -> int:
        return int(self.joint_positions_xyz.shape[1])

    def resolved_observed_mask(self) -> np.ndarray:
        if self.observed_mask is None:
            return (
                np.isfinite(np.asarray(self.joint_positions_xyz, dtype=np.float32)).all(axis=2)
                & (np.asarray(self.joint_confidence, dtype=np.float32) > 0.0)
            ).astype(bool, copy=False)
        observed_mask = np.asarray(self.observed_mask, dtype=bool)
        if observed_mask.shape != self.joint_positions_xyz.shape[:2]:
            raise ValueError(
                "PoseSequence3D observed_mask must match joint_positions_xyz shape [T, J]: "
                f"got {observed_mask.shape} vs {self.joint_positions_xyz.shape[:2]}."
            )
        return observed_mask

    def resolved_imputed_mask(self) -> np.ndarray:
        if self.imputed_mask is None:
            return np.zeros(self.joint_positions_xyz.shape[:2], dtype=bool)
        imputed_mask = np.asarray(self.imputed_mask, dtype=bool)
        if imputed_mask.shape != self.joint_positions_xyz.shape[:2]:
            raise ValueError(
                "PoseSequence3D imputed_mask must match joint_positions_xyz shape [T, J]: "
                f"got {imputed_mask.shape} vs {self.joint_positions_xyz.shape[:2]}."
            )
        return imputed_mask

    def to_npz_payload(self) -> Dict[str, Any]:
        return {
            "clip_id": np.asarray(self.clip_id),
            "joint_names_3d": np.asarray(list(self.joint_names_3d)),
            "joint_positions_xyz": np.asarray(self.joint_positions_xyz, dtype=np.float32),
            "joint_confidence": np.asarray(self.joint_confidence, dtype=np.float32),
            "observed_mask": np.asarray(self.resolved_observed_mask(), dtype=bool),
            "imputed_mask": np.asarray(self.resolved_imputed_mask(), dtype=bool),
            "skeleton_parents": np.asarray(list(self.skeleton_parents), dtype=np.int32),
            "frame_indices": np.asarray(self.frame_indices, dtype=np.int32),
            "timestamps_sec": np.asarray(self.timestamps_sec, dtype=np.float32),
            "fps": np.asarray(-1.0 if self.fps is None else float(self.fps), dtype=np.float32),
            "fps_original": np.asarray(
                -1.0 if self.fps_original is None else float(self.fps_original),
                dtype=np.float32,
            ),
            "num_frames": np.asarray(self.num_frames, dtype=np.int32),
            "num_joints": np.asarray(self.num_joints, dtype=np.int32),
            "source": np.asarray(self.source),
            "coordinate_space": np.asarray(self.coordinate_space),
        }

    @classmethod
    def from_npz_payload(cls, payload: Mapping[str, Any]) -> "PoseSequence3D":
        fps = float(np.asarray(payload["fps"]).item())
        fps_original = float(np.asarray(payload["fps_original"]).item())
        return cls(
            clip_id=str(np.asarray(payload["clip_id"]).item()),
            fps=None if fps < 0.0 else fps,
            fps_original=None if fps_original < 0.0 else fps_original,
            joint_names_3d=[str(value) for value in np.asarray(payload["joint_names_3d"]).tolist()],
            joint_positions_xyz=np.asarray(payload["joint_positions_xyz"], dtype=np.float32),
            joint_confidence=np.asarray(payload["joint_confidence"], dtype=np.float32),
            skeleton_parents=[int(value) for value in np.asarray(payload["skeleton_parents"]).tolist()],
            frame_indices=np.asarray(payload["frame_indices"], dtype=np.int32),
            timestamps_sec=np.asarray(payload["timestamps_sec"], dtype=np.float32),
            source=str(np.asarray(payload["source"]).item()),
            coordinate_space=str(np.asarray(payload["coordinate_space"]).item()),
            observed_mask=(
                np.asarray(payload["observed_mask"], dtype=bool)
                if "observed_mask" in payload
                else None
            ),
            imputed_mask=(
                np.asarray(payload["imputed_mask"], dtype=bool)
                if "imputed_mask" in payload
                else None
            ),
        )


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)
