"""Source-aware quality metrics for prompt-generated pose sequences."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from pose_module.interfaces import IMUGPT_22_JOINT_NAMES, IMUGPT_22_PARENT_INDICES, PoseSequence3D


_PELVIS_INDEX = IMUGPT_22_JOINT_NAMES.index("Pelvis")
_LEFT_SHOULDER_INDEX = IMUGPT_22_JOINT_NAMES.index("Left_shoulder")
_RIGHT_SHOULDER_INDEX = IMUGPT_22_JOINT_NAMES.index("Right_shoulder")
_LEFT_WRIST_INDEX = IMUGPT_22_JOINT_NAMES.index("Left_wrist")
_RIGHT_WRIST_INDEX = IMUGPT_22_JOINT_NAMES.index("Right_wrist")
_LEFT_FOOT_INDEX = IMUGPT_22_JOINT_NAMES.index("Left_foot")
_RIGHT_FOOT_INDEX = IMUGPT_22_JOINT_NAMES.index("Right_foot")


def build_prompt_pose_quality_report(
    sequence: PoseSequence3D,
    *,
    generation_backend: str,
    source_kind: str = "prompt",
    modality_domain: str = "synthetic",
) -> Dict[str, Any]:
    joint_positions_xyz = np.asarray(sequence.joint_positions_xyz, dtype=np.float32)
    finite_mask = np.isfinite(joint_positions_xyz)
    nan_ratio = 1.0 - float(np.mean(finite_mask)) if joint_positions_xyz.size > 0 else 1.0

    fps = _resolve_fps(sequence)
    root_translation = sequence.resolved_root_translation_m()
    if root_translation is None:
        root_translation = joint_positions_xyz[:, _PELVIS_INDEX, :]
    root_motion_energy = _mean_norm_first_difference(root_translation, fps=fps)
    bone_length_cv = _compute_bone_length_cv(joint_positions_xyz)
    foot_sliding_score = _compute_foot_sliding_score(joint_positions_xyz, root_translation, fps=fps)

    notes: list[str] = []
    status = "ok"
    if nan_ratio > 0.0:
        status = "fail"
        notes.append(f"prompt_pose_contains_nan_ratio:{nan_ratio:.6f}")
    elif bone_length_cv > 0.15 or foot_sliding_score > 0.35:
        status = "warning"
    if bone_length_cv > 0.15:
        notes.append(f"bone_length_cv_above_threshold:{bone_length_cv:.4f}")
    if foot_sliding_score > 0.35:
        notes.append(f"foot_sliding_score_above_threshold:{foot_sliding_score:.4f}")

    return {
        "clip_id": str(sequence.clip_id),
        "status": str(status),
        "source_kind": str(source_kind),
        "modality_domain": str(modality_domain),
        "generation_backend": str(generation_backend),
        "fps": None if sequence.fps is None else float(sequence.fps),
        "fps_original": None if sequence.fps_original is None else float(sequence.fps_original),
        "num_frames": int(sequence.num_frames),
        "num_joints": int(sequence.num_joints),
        "coordinate_space": str(sequence.coordinate_space),
        "nan_ratio": float(nan_ratio),
        "bone_length_cv": float(bone_length_cv),
        "root_motion_energy": float(root_motion_energy),
        "foot_sliding_score": float(foot_sliding_score),
        "mean_arm_span_m": float(_compute_mean_arm_span(joint_positions_xyz)),
        "mean_wrist_height_m": float(
            np.mean(
                joint_positions_xyz[:, [_LEFT_WRIST_INDEX, _RIGHT_WRIST_INDEX], 1],
                dtype=np.float32,
            )
        ),
        "notes": list(notes),
    }


def _compute_bone_length_cv(joint_positions_xyz: np.ndarray) -> float:
    bone_lengths: list[np.ndarray] = []
    for joint_index, parent_index in enumerate(IMUGPT_22_PARENT_INDICES):
        if parent_index < 0:
            continue
        segment = joint_positions_xyz[:, joint_index, :] - joint_positions_xyz[:, int(parent_index), :]
        bone_lengths.append(np.linalg.norm(segment, axis=1))
    if len(bone_lengths) == 0:
        return 0.0
    stacked = np.stack(bone_lengths, axis=1)
    mean_lengths = np.mean(stacked, axis=0)
    valid_mask = mean_lengths > 1e-6
    if not np.any(valid_mask):
        return 0.0
    return float(np.mean(np.std(stacked[:, valid_mask], axis=0) / mean_lengths[valid_mask]))


def _compute_foot_sliding_score(
    joint_positions_xyz: np.ndarray,
    root_translation: np.ndarray,
    *,
    fps: float,
) -> float:
    if joint_positions_xyz.shape[0] <= 1:
        return 0.0
    dt = max(float(1.0 / max(fps, 1e-6)), 1e-6)
    feet_world = joint_positions_xyz[:, [_LEFT_FOOT_INDEX, _RIGHT_FOOT_INDEX], :]
    feet_height = feet_world[..., 1]
    grounded_mask = feet_height <= (
        np.min(feet_height, axis=0, keepdims=True) + np.float32(0.03)
    )
    feet_horizontal_speed = np.linalg.norm(np.diff(feet_world[..., (0, 2)], axis=0) / np.float32(dt), axis=2)
    grounded_pairs = grounded_mask[1:]
    if not np.any(grounded_pairs):
        return 0.0
    grounded_speed = feet_horizontal_speed[grounded_pairs]
    root_speed = np.linalg.norm(np.diff(root_translation[:, (0, 2)], axis=0) / np.float32(dt), axis=1)
    root_speed_scale = max(float(np.mean(root_speed)), 1e-6)
    return float(np.mean(grounded_speed) / root_speed_scale)


def _compute_mean_arm_span(joint_positions_xyz: np.ndarray) -> float:
    shoulders = joint_positions_xyz[:, [_LEFT_SHOULDER_INDEX, _RIGHT_SHOULDER_INDEX], :]
    wrists = joint_positions_xyz[:, [_LEFT_WRIST_INDEX, _RIGHT_WRIST_INDEX], :]
    return float(np.mean(np.linalg.norm(wrists - shoulders, axis=2)))


def _mean_norm_first_difference(values: np.ndarray, *, fps: float) -> float:
    if values.shape[0] <= 1:
        return 0.0
    dt = max(float(1.0 / max(fps, 1e-6)), 1e-6)
    diffs = np.diff(values, axis=0) / np.float32(dt)
    return float(np.mean(np.linalg.norm(diffs, axis=1)))


def _resolve_fps(sequence: PoseSequence3D) -> float:
    if sequence.fps is not None and float(sequence.fps) > 0.0:
        return float(sequence.fps)
    timestamps_sec = np.asarray(sequence.timestamps_sec, dtype=np.float32)
    if timestamps_sec.shape[0] <= 1:
        return 20.0
    deltas = np.diff(timestamps_sec)
    valid = deltas > 1e-6
    if not np.any(valid):
        return 20.0
    return float(1.0 / np.median(deltas[valid]))
