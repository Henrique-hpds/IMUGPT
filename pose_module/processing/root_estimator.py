"""Estimate a smooth pseudo-global root trajectory."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from pose_module.interfaces import IMUGPT_22_JOINT_NAMES, IMUGPT_22_PARENT_INDICES, PoseSequence3D
from pose_module.processing.metric_normalizer import BODY_METRIC_LOCAL_COORDINATE_SPACE
from pose_module.processing.temporal_filters import savgol_smooth

DEFAULT_ROOT_SAVGOL_WINDOW_LENGTH = 9
DEFAULT_ROOT_SAVGOL_POLYORDER = 2
DEFAULT_PLANARIZE_VERTICAL = False
DEFAULT_ROOT_DISCONTINUITY_WARNING_M = 0.15
DEFAULT_ROOT_DRIFT_SCORE_WARNING = 0.05
PSEUDO_GLOBAL_METRIC_COORDINATE_SPACE = "pseudo_global_metric"
_VERTICAL_AXIS_INDEX = 1
_EPSILON = 1e-6

_IMUGPT22_INDEX = {name: index for index, name in enumerate(IMUGPT_22_JOINT_NAMES)}


def run_root_trajectory_estimator(
    metric_pose_sequence: PoseSequence3D,
    *,
    normalization_result: Mapping[str, Any],
    smoothing_window_length: int = DEFAULT_ROOT_SAVGOL_WINDOW_LENGTH,
    smoothing_polyorder: int = DEFAULT_ROOT_SAVGOL_POLYORDER,
    planarize_vertical: bool = DEFAULT_PLANARIZE_VERTICAL,
    discontinuity_warning_m: float = DEFAULT_ROOT_DISCONTINUITY_WARNING_M,
    drift_score_warning: float = DEFAULT_ROOT_DRIFT_SCORE_WARNING,
) -> Dict[str, Any]:
    """Estimate a plausible root translation and compose a pseudo-global pose."""

    _validate_metric_pose_sequence(metric_pose_sequence)

    joint_positions_3d_norm = np.asarray(normalization_result["joint_positions_3d_norm"], dtype=np.float32)
    scale_factor = float(normalization_result["scale_factor"])
    if joint_positions_3d_norm.shape != metric_pose_sequence.joint_positions_xyz.shape:
        raise ValueError(
            "Root estimator expects normalization_result['joint_positions_3d_norm'] to match "
            "metric_pose_sequence.joint_positions_xyz shape."
        )
    if not np.isfinite(joint_positions_3d_norm).all():
        raise ValueError("Root estimator expects finite normalization_result['joint_positions_3d_norm'].")
    if not np.isfinite(scale_factor) or scale_factor <= 0.0:
        raise ValueError("Root estimator expects a positive finite normalization scale_factor.")

    pelvis_index = _IMUGPT22_INDEX["Pelvis"]
    raw_root_translation = (
        joint_positions_3d_norm[:, pelvis_index, :] * np.float32(scale_factor)
    ).astype(np.float32, copy=False)
    smoothed_root_translation = savgol_smooth(
        raw_root_translation,
        window_length=int(smoothing_window_length),
        polyorder=int(smoothing_polyorder),
    ).astype(np.float32, copy=False)
    root_translation = smoothed_root_translation.copy()
    if bool(planarize_vertical) and root_translation.shape[0] > 0:
        root_translation[:, _VERTICAL_AXIS_INDEX] = np.float32(
            np.median(smoothed_root_translation[:, _VERTICAL_AXIS_INDEX])
        )

    metric_positions = np.asarray(metric_pose_sequence.joint_positions_xyz, dtype=np.float32)
    joint_positions_global_m = (
        metric_positions + root_translation[:, None, :]
    ).astype(np.float32, copy=False)
    joint_confidence = np.asarray(metric_pose_sequence.joint_confidence, dtype=np.float32)
    observed_mask = np.asarray(metric_pose_sequence.resolved_observed_mask(), dtype=bool)
    imputed_mask = np.asarray(metric_pose_sequence.resolved_imputed_mask(), dtype=bool)

    pose_sequence = PoseSequence3D(
        clip_id=str(metric_pose_sequence.clip_id),
        fps=None if metric_pose_sequence.fps is None else float(metric_pose_sequence.fps),
        fps_original=(
            None if metric_pose_sequence.fps_original is None else float(metric_pose_sequence.fps_original)
        ),
        joint_names_3d=list(metric_pose_sequence.joint_names_3d),
        joint_positions_xyz=joint_positions_global_m,
        joint_confidence=joint_confidence.astype(np.float32, copy=False),
        skeleton_parents=list(metric_pose_sequence.skeleton_parents),
        frame_indices=np.asarray(metric_pose_sequence.frame_indices, dtype=np.int32),
        timestamps_sec=np.asarray(metric_pose_sequence.timestamps_sec, dtype=np.float32),
        source=f"{metric_pose_sequence.source}_root",
        coordinate_space=PSEUDO_GLOBAL_METRIC_COORDINATE_SPACE,
        root_translation_m=root_translation.astype(np.float32, copy=False),
        observed_mask=observed_mask.astype(bool, copy=False),
        imputed_mask=imputed_mask.astype(bool, copy=False),
    )

    trajectory_result = {
        "root_translation_raw_m": raw_root_translation.astype(np.float32, copy=False),
        "root_translation_smoothed_m": smoothed_root_translation.astype(np.float32, copy=False),
        "root_translation_m": root_translation.astype(np.float32, copy=False),
        "joint_positions_global_m": joint_positions_global_m.astype(np.float32, copy=False),
    }
    root_statistics = _compute_root_statistics(root_translation)
    quality_report = _build_root_quality_report(
        metric_pose_sequence=metric_pose_sequence,
        pose_sequence=pose_sequence,
        scale_factor=float(scale_factor),
        root_statistics=root_statistics,
        smoothing_window_length=int(smoothing_window_length),
        smoothing_polyorder=int(smoothing_polyorder),
        planarize_vertical=bool(planarize_vertical),
        discontinuity_warning_m=float(discontinuity_warning_m),
        drift_score_warning=float(drift_score_warning),
    )
    artifacts = {
        "root_translation_raw_m": raw_root_translation.astype(np.float32, copy=False),
        "root_translation_smoothed_m": smoothed_root_translation.astype(np.float32, copy=False),
        "root_translation_m": root_translation.astype(np.float32, copy=False),
        "root_statistics": root_statistics,
    }
    return {
        "pose_sequence": pose_sequence,
        "root_translation_m": root_translation.astype(np.float32, copy=False),
        "trajectory_result": trajectory_result,
        "quality_report": quality_report,
        "artifacts": artifacts,
    }


def _validate_metric_pose_sequence(sequence: PoseSequence3D) -> None:
    joint_names = [str(name) for name in sequence.joint_names_3d]
    if joint_names != list(IMUGPT_22_JOINT_NAMES):
        raise ValueError("Root estimator expects output ordered as IMUGPT_22_JOINT_NAMES.")
    if str(sequence.coordinate_space) != BODY_METRIC_LOCAL_COORDINATE_SPACE:
        raise ValueError(
            "Root estimator expects output in the body_metric_local coordinate space."
        )
    points = np.asarray(sequence.joint_positions_xyz, dtype=np.float32)
    confidence = np.asarray(sequence.joint_confidence, dtype=np.float32)
    if points.ndim != 3 or points.shape[1:] != (len(IMUGPT_22_JOINT_NAMES), 3):
        raise ValueError("Root estimator expects joint_positions_xyz with shape [T, 22, 3].")
    if confidence.shape != points.shape[:2]:
        raise ValueError("Root estimator expects joint_confidence with shape [T, 22].")
    if not np.isfinite(points).all():
        raise ValueError("Root estimator expects finite metric positions.")


def _compute_root_statistics(root_translation: np.ndarray) -> Dict[str, float]:
    root = np.asarray(root_translation, dtype=np.float32)
    if root.ndim != 2 or root.shape[1] != 3:
        raise ValueError("root_translation must have shape [T, 3].")
    if root.shape[0] == 0:
        return {
            "root_path_length_m": 0.0,
            "root_horizontal_path_length_m": 0.0,
            "root_vertical_span_m": 0.0,
            "root_max_step_m": 0.0,
            "root_mean_step_m": 0.0,
            "root_drift_score": 0.0,
        }

    root_steps = np.diff(root, axis=0)
    if root_steps.shape[0] == 0:
        return {
            "root_path_length_m": 0.0,
            "root_horizontal_path_length_m": 0.0,
            "root_vertical_span_m": float(np.ptp(root[:, _VERTICAL_AXIS_INDEX])),
            "root_max_step_m": 0.0,
            "root_mean_step_m": 0.0,
            "root_drift_score": 0.0,
        }

    step_norms = np.linalg.norm(root_steps, axis=1)
    horizontal_step_norms = np.linalg.norm(root_steps[:, (0, 2)], axis=1)
    root_second_diff = np.diff(root, n=2, axis=0)
    if root_second_diff.shape[0] == 0:
        drift_score = 0.0
    else:
        drift_score = float(np.mean(np.linalg.norm(root_second_diff, axis=1)))
    return {
        "root_path_length_m": float(np.sum(step_norms)),
        "root_horizontal_path_length_m": float(np.sum(horizontal_step_norms)),
        "root_vertical_span_m": float(np.ptp(root[:, _VERTICAL_AXIS_INDEX])),
        "root_max_step_m": float(np.max(step_norms)),
        "root_mean_step_m": float(np.mean(step_norms)),
        "root_drift_score": float(drift_score),
    }


def _build_root_quality_report(
    *,
    metric_pose_sequence: PoseSequence3D,
    pose_sequence: PoseSequence3D,
    scale_factor: float,
    root_statistics: Mapping[str, float],
    smoothing_window_length: int,
    smoothing_polyorder: int,
    planarize_vertical: bool,
    discontinuity_warning_m: float,
    drift_score_warning: float,
) -> Dict[str, Any]:
    notes = []
    root_translation = pose_sequence.resolved_root_translation_m()
    if root_translation is None:
        raise ValueError("Root-estimated PoseSequence3D must include root_translation_m.")

    finite_root = np.isfinite(root_translation).all()
    finite_global_pose = np.isfinite(np.asarray(pose_sequence.joint_positions_xyz, dtype=np.float32)).all()
    pelvis_index = _IMUGPT22_INDEX["Pelvis"]
    pelvis_matches_root = np.allclose(
        np.asarray(pose_sequence.joint_positions_xyz, dtype=np.float32)[:, pelvis_index, :],
        root_translation,
        atol=1e-5,
    )
    contract_ok = list(pose_sequence.joint_names_3d) == list(IMUGPT_22_JOINT_NAMES)
    parents_ok = [int(parent) for parent in pose_sequence.skeleton_parents] == list(IMUGPT_22_PARENT_INDICES)
    root_translation_ok = bool(
        finite_root
        and finite_global_pose
        and pelvis_matches_root
        and contract_ok
        and parents_ok
    )
    if not finite_root:
        notes.append("root_translation_contains_nan")
    if not finite_global_pose:
        notes.append("pseudo_global_pose_contains_nan")
    if not pelvis_matches_root:
        notes.append("pelvis_root_inconsistency")
    if not contract_ok:
        notes.append("root_output_joint_contract_mismatch")
    if not parents_ok:
        notes.append("root_output_parent_contract_mismatch")

    root_max_step_m = float(root_statistics["root_max_step_m"])
    root_drift_score = float(root_statistics["root_drift_score"])
    if root_max_step_m > float(discontinuity_warning_m):
        notes.append(f"root_max_step_above_threshold:{root_max_step_m:.4f}")
    if root_drift_score > float(drift_score_warning):
        notes.append(f"root_drift_score_above_threshold:{root_drift_score:.4f}")
    if bool(planarize_vertical):
        notes.append("vertical_root_planarized_to_clip_median")

    status = "ok"
    if not root_translation_ok:
        status = "fail"
    elif (
        root_max_step_m > float(discontinuity_warning_m)
        or root_drift_score > float(drift_score_warning)
    ):
        status = "warning"

    return {
        "clip_id": str(pose_sequence.clip_id),
        "status": str(status),
        "fps": None if pose_sequence.fps is None else float(pose_sequence.fps),
        "fps_original": None if pose_sequence.fps_original is None else float(pose_sequence.fps_original),
        "num_frames": int(pose_sequence.num_frames),
        "num_joints": int(pose_sequence.num_joints),
        "input_joint_format": list(metric_pose_sequence.joint_names_3d),
        "output_joint_format": list(pose_sequence.joint_names_3d),
        "input_coordinate_space": str(metric_pose_sequence.coordinate_space),
        "coordinate_space": str(pose_sequence.coordinate_space),
        "root_translation_ok": bool(root_translation_ok),
        "scale_factor": float(scale_factor),
        "smoothing_window_length": int(smoothing_window_length),
        "smoothing_polyorder": int(smoothing_polyorder),
        "planarize_vertical": bool(planarize_vertical),
        "root_path_length_m": float(root_statistics["root_path_length_m"]),
        "root_horizontal_path_length_m": float(root_statistics["root_horizontal_path_length_m"]),
        "root_vertical_span_m": float(root_statistics["root_vertical_span_m"]),
        "root_max_step_m": float(root_max_step_m),
        "root_mean_step_m": float(root_statistics["root_mean_step_m"]),
        "root_drift_score": float(root_drift_score),
        "assumptions": [
            "pelvis_track_as_root_proxy",
            "camera_static_or_low_egomotion",
            "single_subject_per_clip",
            "pseudo_global_pose_is_approximate",
        ],
        "limitations": [
            "not_absolute_trajectory",
            "sensitive_to_scale_error",
            "sensitive_to_occlusion",
            "camera_motion_can_induce_drift",
        ],
        "notes": list(dict.fromkeys(notes)),
    }
