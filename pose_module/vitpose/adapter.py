"""Adapters between raw ViTPose predictions and canonical pose2d tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from pose_module.interfaces import COCO_17_JOINT_NAMES, PoseFrameInstance, PoseSequence2D
from pose_module.io.cache import load_json_file
from pose_module.tracking.person_selector import TrackState


def load_raw_prediction_frames(raw_prediction_path: str | Path) -> List[Dict[str, Any]]:
    payload = load_json_file(raw_prediction_path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected raw prediction payload list, got {type(payload).__name__}")
    normalized = []
    for frame_payload in payload:
        frame_id = int(frame_payload.get("frame_id", len(normalized)))
        instances = []
        for instance in frame_payload.get("instances", []):
            keypoints_xy = _normalize_keypoints_xy(instance.get("keypoints"))
            keypoint_scores = _normalize_keypoint_scores(instance.get("keypoint_scores"))
            bbox_xyxy = _normalize_bbox_xyxy(instance.get("bbox"), keypoints_xy)
            bbox_score = _normalize_scalar(instance.get("bbox_score"), default=0.0)
            instances.append(
                PoseFrameInstance(
                    frame_id=frame_id,
                    bbox_xyxy=bbox_xyxy,
                    bbox_score=bbox_score,
                    keypoints_xy=keypoints_xy,
                    keypoint_scores=keypoint_scores,
                )
            )
        normalized.append(
            {
                "frame_id": frame_id,
                "instances": instances,
            }
        )
    return normalized


def canonicalize_pose_sequence2d(
    *,
    clip_id: str,
    selected_track: TrackState,
    selected_frame_indices: np.ndarray,
    timestamps_sec: np.ndarray,
    effective_fps: Optional[float],
    fps_original: Optional[float],
    source: str = "vitpose-b",
) -> Tuple[PoseSequence2D, Dict[str, Any]]:
    num_selected_frames = int(len(selected_frame_indices))
    keypoints_xy = np.full((num_selected_frames, len(COCO_17_JOINT_NAMES), 2), np.nan, dtype=np.float32)
    confidence = np.zeros((num_selected_frames, len(COCO_17_JOINT_NAMES)), dtype=np.float32)
    bbox_xywh = np.full((num_selected_frames, 4), np.nan, dtype=np.float32)

    warnings: List[str] = []
    detected_frames = 0
    for output_index, frame_id in enumerate(selected_frame_indices.tolist()):
        detection = selected_track.detections.get(int(frame_id))
        if detection is None:
            continue

        normalized_keypoints = _ensure_coco17_shape(detection.keypoints_xy, fill_value=np.nan)
        normalized_scores = _ensure_coco17_scores(detection.keypoint_scores)
        if normalized_keypoints is None or normalized_scores is None:
            warnings.append("unexpected_joint_count_from_backend")
            continue

        keypoints_xy[output_index] = normalized_keypoints
        confidence[output_index] = normalized_scores
        bbox_xywh[output_index] = _bbox_xyxy_to_xywh(detection.bbox_xyxy)
        detected_frames += 1

    visible_joint_ratio = 0.0
    if confidence.size > 0:
        visible_joint_ratio = float(np.count_nonzero(confidence > 0.0) / float(confidence.size))
    valid_scores = confidence[confidence > 0.0]
    mean_confidence = float(np.mean(valid_scores)) if valid_scores.size > 0 else 0.0

    quality_warnings = list(dict.fromkeys(warnings))
    if detected_frames == 0:
        quality_warnings.append("selected_track_missing_on_all_selected_frames")
    elif detected_frames < num_selected_frames:
        quality_warnings.append("selected_track_missing_on_some_selected_frames")

    status = "ok"
    if detected_frames == 0:
        status = "fail"
    elif visible_joint_ratio < 0.8 or mean_confidence < 0.5:
        status = "warning"

    sequence = PoseSequence2D(
        clip_id=str(clip_id),
        fps=None if effective_fps is None else float(effective_fps),
        fps_original=None if fps_original is None else float(fps_original),
        joint_names_2d=list(COCO_17_JOINT_NAMES),
        keypoints_xy=keypoints_xy,
        confidence=confidence,
        bbox_xywh=bbox_xywh,
        frame_indices=selected_frame_indices.astype(np.int32, copy=False),
        timestamps_sec=timestamps_sec.astype(np.float32, copy=False),
        source=str(source),
    )
    quality_report = {
        "clip_id": str(clip_id),
        "status": str(status),
        "fps": None if effective_fps is None else float(effective_fps),
        "fps_original": None if fps_original is None else float(fps_original),
        "num_selected_frames": int(num_selected_frames),
        "frames_with_selected_track": int(detected_frames),
        "visible_joint_ratio": float(visible_joint_ratio),
        "mean_confidence": float(mean_confidence),
        "notes": quality_warnings,
    }
    return sequence, quality_report


def write_pose_sequence_npz(sequence: PoseSequence2D, pose2d_npz_path: str | Path) -> None:
    pose2d_npz_path = Path(pose2d_npz_path)
    pose2d_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pose2d_npz_path, **sequence.to_npz_payload())


def _normalize_keypoints_xy(raw_value: Any) -> np.ndarray:
    array = np.asarray(raw_value, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[-1] < 2:
        return np.empty((0, 2), dtype=np.float32)
    return array[:, :2].astype(np.float32, copy=False)


def _normalize_keypoint_scores(raw_value: Any) -> np.ndarray:
    array = np.asarray(raw_value, dtype=np.float32)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        return np.empty((0,), dtype=np.float32)
    return array.astype(np.float32, copy=False)


def _normalize_bbox_xyxy(raw_value: Any, keypoints_xy: np.ndarray) -> np.ndarray:
    array = np.asarray(raw_value, dtype=np.float32)
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    flat = array.reshape(-1) if array.size > 0 else np.empty((0,), dtype=np.float32)
    if flat.size >= 4:
        return flat[:4].astype(np.float32, copy=False)
    valid_mask = np.isfinite(keypoints_xy).all(axis=1)
    if np.any(valid_mask):
        valid_points = keypoints_xy[valid_mask]
        min_xy = np.min(valid_points, axis=0)
        max_xy = np.max(valid_points, axis=0)
        return np.asarray([min_xy[0], min_xy[1], max_xy[0], max_xy[1]], dtype=np.float32)
    return np.asarray([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)


def _normalize_scalar(raw_value: Any, *, default: float) -> float:
    array = np.asarray(raw_value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return float(default)
    return float(array[0])


def _ensure_coco17_shape(array: np.ndarray, *, fill_value: float) -> Optional[np.ndarray]:
    if array.ndim != 2 or array.shape[1] < 2:
        return None
    normalized = np.full((len(COCO_17_JOINT_NAMES), 2), fill_value, dtype=np.float32)
    count = min(int(array.shape[0]), len(COCO_17_JOINT_NAMES))
    normalized[:count] = array[:count, :2]
    return normalized


def _ensure_coco17_scores(array: np.ndarray) -> Optional[np.ndarray]:
    if array.ndim != 1:
        return None
    normalized = np.zeros((len(COCO_17_JOINT_NAMES),), dtype=np.float32)
    count = min(int(array.shape[0]), len(COCO_17_JOINT_NAMES))
    normalized[:count] = array[:count]
    return normalized


def _bbox_xyxy_to_xywh(bbox_xyxy: np.ndarray) -> np.ndarray:
    if bbox_xyxy.size < 4 or not np.isfinite(bbox_xyxy[:4]).all():
        return np.asarray([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy[:4]]
    return np.asarray([x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], dtype=np.float32)
