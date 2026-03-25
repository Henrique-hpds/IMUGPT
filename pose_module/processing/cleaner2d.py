"""Stage 5.4: clean 2D pose tracks and adapt them to the MotionBERT contract."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from pose_module.interfaces import (
    MOTIONBERT_17_JOINT_NAMES,
    MOTIONBERT_17_PARENT_INDICES,
    PoseSequence2D,
)

from .temporal_filters import interpolate_short_gaps, savgol_smooth

DEFAULT_MAX_GAP_INTERP = 5
DEFAULT_SAVGOL_WINDOW = 9
DEFAULT_SAVGOL_POLYORDER = 2
DEFAULT_LOW_CONF_THRESHOLD = 0.2
DEFAULT_MAX_ANGULAR_VELOCITY_DEG_PER_SEC = 1080.0
DEFAULT_BBOX_MARGIN_RATIO = 0.2
DEFAULT_MIN_VISIBLE_JOINT_RATIO = 0.8
DEFAULT_MIN_MEAN_CONFIDENCE = 0.5
DEFAULT_MAX_OUTLIER_RATIO = 0.1
DEFAULT_MAX_FRAME_MISSING_JOINT_RATIO = 0.2
DEFAULT_MAX_FRAMES_OVER_MISSING_RATIO = 0.3
DEFAULT_MAX_TEMPORAL_JITTER_SCORE = 0.12

_MOTIONBERT_INDEX = {name: index for index, name in enumerate(MOTIONBERT_17_JOINT_NAMES)}
_DIRECT_JOINT_MAPPINGS = {
    "left_hip": "left_hip",
    "right_hip": "right_hip",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
    "left_ankle": "left_ankle",
    "right_ankle": "right_ankle",
    "left_shoulder": "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
}
_HEAD_SOURCE_JOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")


def clean_pose_sequence2d(
    sequence: PoseSequence2D,
    *,
    track_report: Optional[Mapping[str, Any]] = None,
    max_gap_interp: int = DEFAULT_MAX_GAP_INTERP,
    savgol_window: int = DEFAULT_SAVGOL_WINDOW,
    savgol_polyorder: int = DEFAULT_SAVGOL_POLYORDER,
    low_conf_threshold: float = DEFAULT_LOW_CONF_THRESHOLD,
    max_angular_velocity_deg_per_sec: float = DEFAULT_MAX_ANGULAR_VELOCITY_DEG_PER_SEC,
    bbox_margin_ratio: float = DEFAULT_BBOX_MARGIN_RATIO,
    min_visible_joint_ratio: float = DEFAULT_MIN_VISIBLE_JOINT_RATIO,
    min_mean_confidence: float = DEFAULT_MIN_MEAN_CONFIDENCE,
    max_outlier_ratio: float = DEFAULT_MAX_OUTLIER_RATIO,
    max_frame_missing_joint_ratio: float = DEFAULT_MAX_FRAME_MISSING_JOINT_RATIO,
    max_frames_over_missing_ratio: float = DEFAULT_MAX_FRAMES_OVER_MISSING_RATIO,
    max_temporal_jitter_score: float = DEFAULT_MAX_TEMPORAL_JITTER_SCORE,
) -> Tuple[PoseSequence2D, Dict[str, Any], Dict[str, np.ndarray]]:
    """Return a MotionBERT-ready 2D sequence plus stage-specific quality metrics."""

    raw_motionbert_xy, raw_motionbert_conf = _map_vitpose_to_motionbert17(sequence)
    points = raw_motionbert_xy.copy()
    confidence = raw_motionbert_conf.copy()

    points, confidence, bbox_outlier_mask, valid_bbox_mask = _invalidate_points_outside_bbox(
        points,
        confidence,
        sequence.bbox_xywh,
        bbox_margin_ratio=float(bbox_margin_ratio),
    )
    points, confidence, angular_outlier_mask = _clip_impossible_angular_velocity(
        points,
        confidence,
        fps=sequence.fps,
        max_angular_velocity_deg_per_sec=float(max_angular_velocity_deg_per_sec),
    )

    points, confidence, interpolated_joint_mask = _interpolate_motionbert_gaps(
        points,
        confidence,
        max_gap_interp=int(max_gap_interp),
        low_conf_threshold=float(low_conf_threshold),
    )

    points = _smooth_motionbert_sequence(
        points,
        confidence,
        window_length=int(savgol_window),
        polyorder=int(savgol_polyorder),
        low_conf_threshold=float(low_conf_threshold),
    )

    invalid_joint_mask = (~np.isfinite(points).all(axis=2)) | (confidence < float(low_conf_threshold))
    confidence[invalid_joint_mask] = 0.0
    points[invalid_joint_mask] = np.nan

    normalized_points, centers_xy, scales = _normalize_for_motionbert(
        points,
        confidence,
        sequence.bbox_xywh,
    )

    visible_joint_ratio = _visible_joint_ratio(confidence)
    mean_confidence = _mean_confidence(confidence)
    per_frame_missing_joint_ratio = 1.0 - np.mean(confidence > 0.0, axis=1)
    frames_over_missing_ratio = float(
        np.mean(per_frame_missing_joint_ratio > float(max_frame_missing_joint_ratio))
    )
    invalid_bbox_ratio = float(np.mean(~valid_bbox_mask)) if valid_bbox_mask.size > 0 else 0.0
    outlier_joint_mask = bbox_outlier_mask | angular_outlier_mask
    outlier_ratio = float(np.mean(outlier_joint_mask)) if outlier_joint_mask.size > 0 else 0.0
    temporal_jitter_score = _compute_temporal_jitter_score(normalized_points, confidence)
    frames_interpolated = int(np.count_nonzero(np.any(interpolated_joint_mask, axis=1)))
    interpolated_joint_ratio = (
        float(np.count_nonzero(interpolated_joint_mask)) / float(interpolated_joint_mask.size)
        if interpolated_joint_mask.size > 0
        else 0.0
    )

    notes = []
    track_report = {} if track_report is None else dict(track_report)
    if track_report.get("status") == "fail":
        notes.append("selected_track_unavailable")
    if "selected_track_has_gaps" in track_report.get("warnings", []):
        notes.append("selected_track_discontinuous")
    if frames_over_missing_ratio > float(max_frames_over_missing_ratio):
        notes.append("too_many_frames_with_excess_missing_joints")
    if invalid_bbox_ratio > 0.0:
        notes.append("bbox_collapsed_or_subject_partially_out_of_scene")
    if outlier_ratio > float(max_outlier_ratio):
        notes.append("high_outlier_ratio_after_cleaning")
    if temporal_jitter_score > float(max_temporal_jitter_score):
        notes.append("temporal_jitter_above_threshold")
    if visible_joint_ratio < float(min_visible_joint_ratio):
        notes.append("visible_joint_ratio_below_threshold")
    if mean_confidence < float(min_mean_confidence):
        notes.append("mean_confidence_below_threshold")

    status = "ok"
    if (
        track_report.get("status") == "fail"
        or "selected_track_has_gaps" in track_report.get("warnings", [])
        or frames_over_missing_ratio > float(max_frames_over_missing_ratio)
        or invalid_bbox_ratio > float(max_frames_over_missing_ratio)
        or visible_joint_ratio < float(min_visible_joint_ratio)
        or mean_confidence < float(min_mean_confidence)
    ):
        status = "fail"
    elif (
        outlier_ratio > float(max_outlier_ratio)
        or temporal_jitter_score > float(max_temporal_jitter_score)
        or len(notes) > 0
    ):
        status = "warning"

    cleaned_sequence = PoseSequence2D(
        clip_id=str(sequence.clip_id),
        fps=None if sequence.fps is None else float(sequence.fps),
        fps_original=None if sequence.fps_original is None else float(sequence.fps_original),
        joint_names_2d=list(MOTIONBERT_17_JOINT_NAMES),
        keypoints_xy=normalized_points.astype(np.float32, copy=False),
        confidence=confidence.astype(np.float32, copy=False),
        bbox_xywh=np.asarray(sequence.bbox_xywh, dtype=np.float32),
        frame_indices=np.asarray(sequence.frame_indices, dtype=np.int32),
        timestamps_sec=np.asarray(sequence.timestamps_sec, dtype=np.float32),
        source=f"{sequence.source}_motionbert17_clean",
    )
    quality_report = {
        "clip_id": str(sequence.clip_id),
        "status": str(status),
        "fps": None if sequence.fps is None else float(sequence.fps),
        "fps_original": None if sequence.fps_original is None else float(sequence.fps_original),
        "num_frames": int(sequence.num_frames),
        "input_joint_format": list(sequence.joint_names_2d),
        "output_joint_format": list(MOTIONBERT_17_JOINT_NAMES),
        "visible_joint_ratio": float(visible_joint_ratio),
        "mean_confidence": float(mean_confidence),
        "temporal_jitter_score": float(temporal_jitter_score),
        "outlier_ratio": float(outlier_ratio),
        "invalid_bbox_ratio": float(invalid_bbox_ratio),
        "frames_interpolated": int(frames_interpolated),
        "interpolated_joint_ratio": float(interpolated_joint_ratio),
        "frames_over_missing_joint_threshold": float(frames_over_missing_ratio),
        "normalization_mode": "pelvis_centered_bbox_scale",
        "notes": list(dict.fromkeys(notes)),
    }
    artifacts = {
        "raw_motionbert17_xy": raw_motionbert_xy.astype(np.float32, copy=False),
        "clean_motionbert17_xy_pixels": points.astype(np.float32, copy=False),
        "clean_motionbert17_xy": normalized_points.astype(np.float32, copy=False),
        "normalization_centers_xy": centers_xy.astype(np.float32, copy=False),
        "normalization_scales": scales.astype(np.float32, copy=False),
    }
    return cleaned_sequence, quality_report, artifacts


def _map_vitpose_to_motionbert17(sequence: PoseSequence2D) -> Tuple[np.ndarray, np.ndarray]:
    source_index = {str(name): idx for idx, name in enumerate(sequence.joint_names_2d)}
    num_frames = int(sequence.num_frames)
    target_xy = np.full((num_frames, len(MOTIONBERT_17_JOINT_NAMES), 2), np.nan, dtype=np.float32)
    target_conf = np.zeros((num_frames, len(MOTIONBERT_17_JOINT_NAMES)), dtype=np.float32)

    keypoints_xy = np.asarray(sequence.keypoints_xy, dtype=np.float32)
    confidence = np.asarray(sequence.confidence, dtype=np.float32)

    for target_name, source_name in _DIRECT_JOINT_MAPPINGS.items():
        if source_name not in source_index:
            continue
        source_joint_index = source_index[source_name]
        target_joint_index = _MOTIONBERT_INDEX[target_name]
        target_xy[:, target_joint_index] = keypoints_xy[:, source_joint_index]
        target_conf[:, target_joint_index] = confidence[:, source_joint_index]

    pelvis_xy, pelvis_conf = _midpoint_series(
        keypoints_xy[:, source_index["left_hip"]],
        confidence[:, source_index["left_hip"]],
        keypoints_xy[:, source_index["right_hip"]],
        confidence[:, source_index["right_hip"]],
    )
    target_xy[:, _MOTIONBERT_INDEX["pelvis"]] = pelvis_xy
    target_conf[:, _MOTIONBERT_INDEX["pelvis"]] = pelvis_conf

    neck_xy, neck_conf = _midpoint_series(
        keypoints_xy[:, source_index["left_shoulder"]],
        confidence[:, source_index["left_shoulder"]],
        keypoints_xy[:, source_index["right_shoulder"]],
        confidence[:, source_index["right_shoulder"]],
    )
    target_xy[:, _MOTIONBERT_INDEX["neck"]] = neck_xy
    target_conf[:, _MOTIONBERT_INDEX["neck"]] = neck_conf

    head_inputs_xy = []
    head_inputs_conf = []
    for source_name in _HEAD_SOURCE_JOINTS:
        if source_name not in source_index:
            continue
        source_joint_index = source_index[source_name]
        head_inputs_xy.append(keypoints_xy[:, source_joint_index])
        head_inputs_conf.append(confidence[:, source_joint_index])
    head_xy, head_conf = _weighted_centroid_series(
        np.stack(head_inputs_xy, axis=1),
        np.stack(head_inputs_conf, axis=1),
    )
    target_xy[:, _MOTIONBERT_INDEX["head"]] = head_xy
    target_conf[:, _MOTIONBERT_INDEX["head"]] = head_conf

    spine_xy, spine_conf = _interpolate_body_axis_series(
        pelvis_xy,
        pelvis_conf,
        neck_xy,
        neck_conf,
        alpha=1.0 / 3.0,
    )
    thorax_xy, thorax_conf = _interpolate_body_axis_series(
        pelvis_xy,
        pelvis_conf,
        neck_xy,
        neck_conf,
        alpha=2.0 / 3.0,
    )
    target_xy[:, _MOTIONBERT_INDEX["spine"]] = spine_xy
    target_conf[:, _MOTIONBERT_INDEX["spine"]] = spine_conf
    target_xy[:, _MOTIONBERT_INDEX["thorax"]] = thorax_xy
    target_conf[:, _MOTIONBERT_INDEX["thorax"]] = thorax_conf

    return target_xy, target_conf


def _midpoint_series(
    left_xy: np.ndarray,
    left_conf: np.ndarray,
    right_xy: np.ndarray,
    right_conf: np.ndarray,
    *,
    fallback_conf_scale: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    output_xy = np.full_like(left_xy, np.nan, dtype=np.float32)
    output_conf = np.zeros_like(left_conf, dtype=np.float32)

    left_valid = np.isfinite(left_xy).all(axis=1) & (left_conf > 0.0)
    right_valid = np.isfinite(right_xy).all(axis=1) & (right_conf > 0.0)
    both_valid = left_valid & right_valid
    output_xy[both_valid] = 0.5 * (left_xy[both_valid] + right_xy[both_valid])
    output_conf[both_valid] = 0.5 * (left_conf[both_valid] + right_conf[both_valid])

    left_only = left_valid & ~right_valid
    output_xy[left_only] = left_xy[left_only]
    output_conf[left_only] = left_conf[left_only] * float(fallback_conf_scale)

    right_only = right_valid & ~left_valid
    output_xy[right_only] = right_xy[right_only]
    output_conf[right_only] = right_conf[right_only] * float(fallback_conf_scale)
    return output_xy, output_conf


def _weighted_centroid_series(points_xy: np.ndarray, points_conf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    output_xy = np.full((points_xy.shape[0], 2), np.nan, dtype=np.float32)
    output_conf = np.zeros((points_conf.shape[0],), dtype=np.float32)

    valid_mask = np.isfinite(points_xy).all(axis=2) & (points_conf > 0.0)
    weights = np.where(valid_mask, points_conf, 0.0).astype(np.float32)
    weight_sum = np.sum(weights, axis=1)
    non_zero_weight = weight_sum > 0.0
    if np.any(non_zero_weight):
        numerator = np.sum(points_xy * weights[..., None], axis=1)
        output_xy[non_zero_weight] = numerator[non_zero_weight] / weight_sum[non_zero_weight, None]
        valid_counts = np.sum(valid_mask, axis=1)
        confidence_sum = np.sum(np.where(valid_mask, points_conf, 0.0), axis=1)
        mean_confidence = np.divide(
            confidence_sum,
            np.maximum(valid_counts, 1),
            out=np.zeros_like(confidence_sum, dtype=np.float32),
            where=valid_counts > 0,
        )
        output_conf[non_zero_weight] = mean_confidence[non_zero_weight]
    return output_xy, np.nan_to_num(output_conf, nan=0.0)


def _interpolate_body_axis_series(
    pelvis_xy: np.ndarray,
    pelvis_conf: np.ndarray,
    neck_xy: np.ndarray,
    neck_conf: np.ndarray,
    *,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    output_xy = np.full_like(pelvis_xy, np.nan, dtype=np.float32)
    output_conf = np.zeros_like(pelvis_conf, dtype=np.float32)

    pelvis_valid = np.isfinite(pelvis_xy).all(axis=1) & (pelvis_conf > 0.0)
    neck_valid = np.isfinite(neck_xy).all(axis=1) & (neck_conf > 0.0)
    both_valid = pelvis_valid & neck_valid
    output_xy[both_valid] = pelvis_xy[both_valid] + (float(alpha) * (neck_xy[both_valid] - pelvis_xy[both_valid]))
    output_conf[both_valid] = np.minimum(pelvis_conf[both_valid], neck_conf[both_valid])
    return output_xy, output_conf


def _invalidate_points_outside_bbox(
    points_xy: np.ndarray,
    confidence: np.ndarray,
    bbox_xywh: np.ndarray,
    *,
    bbox_margin_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_xy, dtype=np.float32).copy()
    conf = np.asarray(confidence, dtype=np.float32).copy()
    bbox = np.asarray(bbox_xywh, dtype=np.float32)

    valid_bbox_mask = (
        np.isfinite(bbox).all(axis=1)
        & (bbox[:, 2] > 1.0)
        & (bbox[:, 3] > 1.0)
    )
    outlier_mask = np.zeros(conf.shape, dtype=bool)

    for frame_index in range(points.shape[0]):
        if not valid_bbox_mask[frame_index]:
            continue
        x, y, width, height = [float(value) for value in bbox[frame_index]]
        margin = float(bbox_margin_ratio) * max(width, height)
        x_min = x - margin
        y_min = y - margin
        x_max = x + width + margin
        y_max = y + height + margin
        joint_valid = np.isfinite(points[frame_index]).all(axis=1) & (conf[frame_index] > 0.0)
        joint_x = points[frame_index, :, 0]
        joint_y = points[frame_index, :, 1]
        joint_outside = joint_valid & (
            (joint_x < x_min) | (joint_x > x_max) | (joint_y < y_min) | (joint_y > y_max)
        )
        outlier_mask[frame_index] = joint_outside

    points[outlier_mask] = np.nan
    conf[outlier_mask] = 0.0
    return points, conf, outlier_mask, valid_bbox_mask


def _clip_impossible_angular_velocity(
    points_xy: np.ndarray,
    confidence: np.ndarray,
    *,
    fps: Optional[float],
    max_angular_velocity_deg_per_sec: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_xy, dtype=np.float32).copy()
    conf = np.asarray(confidence, dtype=np.float32).copy()
    outlier_mask = np.zeros(conf.shape, dtype=bool)

    if fps is None or float(fps) <= 0.0 or points.shape[0] < 2:
        return points, conf, outlier_mask

    max_delta_radians = np.deg2rad(float(max_angular_velocity_deg_per_sec)) / float(fps)
    for child_index, parent_index in enumerate(MOTIONBERT_17_PARENT_INDICES):
        if parent_index < 0:
            continue
        previous_vectors = points[:-1, child_index] - points[:-1, parent_index]
        current_vectors = points[1:, child_index] - points[1:, parent_index]
        previous_valid = (
            np.isfinite(previous_vectors).all(axis=1)
            & (conf[:-1, child_index] > 0.0)
            & (conf[:-1, parent_index] > 0.0)
        )
        current_valid = (
            np.isfinite(current_vectors).all(axis=1)
            & (conf[1:, child_index] > 0.0)
            & (conf[1:, parent_index] > 0.0)
        )
        previous_norm = np.linalg.norm(previous_vectors, axis=1)
        current_norm = np.linalg.norm(current_vectors, axis=1)
        valid_transition = previous_valid & current_valid & (previous_norm > 1e-4) & (current_norm > 1e-4)
        if not np.any(valid_transition):
            continue

        cross = (previous_vectors[:, 0] * current_vectors[:, 1]) - (
            previous_vectors[:, 1] * current_vectors[:, 0]
        )
        dot = np.sum(previous_vectors * current_vectors, axis=1)
        angle_delta = np.abs(np.arctan2(cross, dot))
        bad_transition = valid_transition & (angle_delta > max_delta_radians)
        outlier_mask[1:, child_index] |= bad_transition

    points[outlier_mask] = np.nan
    conf[outlier_mask] = 0.0
    return points, conf, outlier_mask


def _interpolate_motionbert_gaps(
    points_xy: np.ndarray,
    confidence: np.ndarray,
    *,
    max_gap_interp: int,
    low_conf_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_xy, dtype=np.float32).copy()
    conf = np.asarray(confidence, dtype=np.float32).copy()
    interpolated_joint_mask = np.zeros(conf.shape, dtype=bool)

    for joint_index in range(points.shape[1]):
        joint_valid = np.isfinite(points[:, joint_index]).all(axis=1) & (conf[:, joint_index] >= low_conf_threshold)
        interpolated_xy, interpolated_xy_mask = interpolate_short_gaps(
            points[:, joint_index],
            np.repeat(joint_valid[:, None], 2, axis=1),
            max_gap=int(max_gap_interp),
        )
        interpolated_conf, interpolated_conf_mask = interpolate_short_gaps(
            conf[:, joint_index],
            joint_valid,
            max_gap=int(max_gap_interp),
        )

        joint_interpolated_mask = np.any(interpolated_xy_mask, axis=1) | interpolated_conf_mask.astype(bool)
        points[:, joint_index] = interpolated_xy
        conf[:, joint_index] = interpolated_conf
        interpolated_joint_mask[:, joint_index] = joint_interpolated_mask

    return points, conf, interpolated_joint_mask


def _smooth_motionbert_sequence(
    points_xy: np.ndarray,
    confidence: np.ndarray,
    *,
    window_length: int,
    polyorder: int,
    low_conf_threshold: float,
) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).copy()
    conf = np.asarray(confidence, dtype=np.float32)
    for joint_index in range(points.shape[1]):
        joint_valid = np.isfinite(points[:, joint_index]).all(axis=1) & (conf[:, joint_index] >= low_conf_threshold)
        smoothed_xy = savgol_smooth(
            points[:, joint_index],
            window_length=int(window_length),
            polyorder=int(polyorder),
            valid_mask=np.repeat(joint_valid[:, None], 2, axis=1),
        )
        points[:, joint_index] = smoothed_xy
        points[~joint_valid, joint_index] = np.nan
    return points


def _normalize_for_motionbert(
    points_xy: np.ndarray,
    confidence: np.ndarray,
    bbox_xywh: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_xy, dtype=np.float32)
    conf = np.asarray(confidence, dtype=np.float32)
    bbox = np.asarray(bbox_xywh, dtype=np.float32)

    num_frames = int(points.shape[0])
    centers_xy = np.full((num_frames, 2), np.nan, dtype=np.float32)
    scales = np.full((num_frames,), np.nan, dtype=np.float32)
    pelvis_index = _MOTIONBERT_INDEX["pelvis"]

    valid_bbox_mask = (
        np.isfinite(bbox).all(axis=1)
        & (bbox[:, 2] > 1.0)
        & (bbox[:, 3] > 1.0)
    )
    pelvis_valid = np.isfinite(points[:, pelvis_index]).all(axis=1) & (conf[:, pelvis_index] > 0.0)

    for frame_index in range(num_frames):
        if pelvis_valid[frame_index]:
            centers_xy[frame_index] = points[frame_index, pelvis_index]
        elif valid_bbox_mask[frame_index]:
            x, y, width, height = [float(value) for value in bbox[frame_index]]
            centers_xy[frame_index] = np.asarray([x + (0.5 * width), y + (0.5 * height)], dtype=np.float32)
        else:
            valid_joints = np.isfinite(points[frame_index]).all(axis=1) & (conf[frame_index] > 0.0)
            if np.any(valid_joints):
                centers_xy[frame_index] = np.mean(points[frame_index, valid_joints], axis=0)

        if valid_bbox_mask[frame_index]:
            scales[frame_index] = max(float(bbox[frame_index, 2]), float(bbox[frame_index, 3]))
        else:
            valid_joints = np.isfinite(points[frame_index]).all(axis=1) & (conf[frame_index] > 0.0)
            if np.any(valid_joints):
                valid_points = points[frame_index, valid_joints]
                extent = np.max(valid_points, axis=0) - np.min(valid_points, axis=0)
                scales[frame_index] = max(float(extent[0]), float(extent[1]), 1.0)

    centers_xy = _fill_missing_vectors(centers_xy)
    scales = _fill_missing_scalars(scales)
    normalized = np.full_like(points, np.nan, dtype=np.float32)
    for frame_index in range(num_frames):
        scale_value = max(float(scales[frame_index]), 1.0)
        normalized[frame_index] = (points[frame_index] - centers_xy[frame_index]) / scale_value
    return normalized, centers_xy, scales


def _fill_missing_vectors(values_xy: np.ndarray) -> np.ndarray:
    output = np.asarray(values_xy, dtype=np.float32).copy()
    num_frames = int(output.shape[0])
    frame_axis = np.arange(num_frames, dtype=np.float32)
    for dim_index in range(output.shape[1]):
        valid_mask = np.isfinite(output[:, dim_index])
        if not np.any(valid_mask):
            output[:, dim_index] = 0.0
            continue
        output[:, dim_index] = np.interp(frame_axis, frame_axis[valid_mask], output[valid_mask, dim_index])
    return output


def _fill_missing_scalars(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32).copy()
    valid_mask = np.isfinite(output) & (output > 0.0)
    if not np.any(valid_mask):
        output[:] = 1.0
        return output
    frame_axis = np.arange(output.shape[0], dtype=np.float32)
    output[:] = np.interp(frame_axis, frame_axis[valid_mask], output[valid_mask])
    output[:] = np.maximum(output, 1.0)
    return output


def _visible_joint_ratio(confidence: np.ndarray) -> float:
    if confidence.size == 0:
        return 0.0
    return float(np.count_nonzero(confidence > 0.0) / float(confidence.size))


def _mean_confidence(confidence: np.ndarray) -> float:
    valid_scores = np.asarray(confidence, dtype=np.float32)
    valid_scores = valid_scores[valid_scores > 0.0]
    if valid_scores.size == 0:
        return 0.0
    return float(np.mean(valid_scores))


def _compute_temporal_jitter_score(points_xy: np.ndarray, confidence: np.ndarray) -> float:
    points = np.asarray(points_xy, dtype=np.float32)
    conf = np.asarray(confidence, dtype=np.float32)
    if points.shape[0] < 3:
        return 0.0

    valid = np.isfinite(points).all(axis=2) & (conf > 0.0)
    acceleration = points[2:] - (2.0 * points[1:-1]) + points[:-2]
    valid_triplets = valid[2:] & valid[1:-1] & valid[:-2]
    if not np.any(valid_triplets):
        return 0.0
    return float(np.mean(np.linalg.norm(acceleration[valid_triplets], axis=1)))
