"""Stage 5.7: normalize mapped 3D poses into a local metric body frame."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from pose_module.interfaces import IMUGPT_22_JOINT_NAMES, PoseSequence3D
from pose_module.processing.temporal_filters import savgol_smooth

DEFAULT_TARGET_FEMUR_LENGTH_M = 0.45
DEFAULT_TARGET_TIBIA_LENGTH_M = 0.40
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.20
DEFAULT_SAVGOL_WINDOW_LENGTH = 9
DEFAULT_SAVGOL_POLYORDER = 2
DEFAULT_CORRECTED_SAVGOL_WINDOW_LENGTH = 3
DEFAULT_OBSERVED_WINDOW_FRAMES = 15
DEFAULT_SEATED_KNEE_ANGLE_THRESHOLD_DEG = 140.0
DEFAULT_STANDING_KNEE_ANGLE_THRESHOLD_DEG = 160.0
BODY_METRIC_LOCAL_COORDINATE_SPACE = "body_metric_local"
_EPSILON = 1e-6

_IMUGPT22_INDEX = {name: index for index, name in enumerate(IMUGPT_22_JOINT_NAMES)}
_LEG_CONFIGS = {
    "left_leg": {
        "hip": _IMUGPT22_INDEX["Left_hip"],
        "knee": _IMUGPT22_INDEX["Left_knee"],
        "ankle": _IMUGPT22_INDEX["Left_ankle"],
    },
    "right_leg": {
        "hip": _IMUGPT22_INDEX["Right_hip"],
        "knee": _IMUGPT22_INDEX["Right_knee"],
        "ankle": _IMUGPT22_INDEX["Right_ankle"],
    },
}


def run_metric_normalizer(
    sequence: PoseSequence3D,
    *,
    target_femur_length_m: float = DEFAULT_TARGET_FEMUR_LENGTH_M,
    target_tibia_length_m: float = DEFAULT_TARGET_TIBIA_LENGTH_M,
    low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    smoothing_window_length: int = DEFAULT_SAVGOL_WINDOW_LENGTH,
    smoothing_polyorder: int = DEFAULT_SAVGOL_POLYORDER,
    corrected_smoothing_window_length: int = DEFAULT_CORRECTED_SAVGOL_WINDOW_LENGTH,
    lower_limb_correction_masks: Mapping[str, np.ndarray] | None = None,
    observed_window_frames: int = DEFAULT_OBSERVED_WINDOW_FRAMES,
    seated_knee_angle_threshold_deg: float = DEFAULT_SEATED_KNEE_ANGLE_THRESHOLD_DEG,
    standing_knee_angle_threshold_deg: float = DEFAULT_STANDING_KNEE_ANGLE_THRESHOLD_DEG,
) -> Dict[str, Any]:
    """Convert an IMUGPT22 pose into a smoothed local metric body-frame pose."""

    _validate_imugpt22_sequence(sequence)

    joint_positions_3d_norm = np.asarray(sequence.joint_positions_xyz, dtype=np.float32).copy()
    joint_confidence = np.asarray(sequence.joint_confidence, dtype=np.float32)
    observed_mask = np.asarray(sequence.resolved_observed_mask(), dtype=bool)
    imputed_mask = np.asarray(sequence.resolved_imputed_mask(), dtype=bool)
    pelvis_index = _IMUGPT22_INDEX["Pelvis"]
    centered_positions = joint_positions_3d_norm - joint_positions_3d_norm[:, pelvis_index : pelvis_index + 1, :]

    body_frame_rotations, body_frame_fallback_mask = _build_body_frame_rotation_matrices(
        joint_positions_3d_norm,
        joint_confidence=joint_confidence,
        observed_mask=observed_mask,
        imputed_mask=imputed_mask,
        low_confidence_threshold=float(low_confidence_threshold),
    )
    joint_positions_body_frame = np.einsum(
        "tki,tij->tkj",
        centered_positions,
        body_frame_rotations,
        dtype=np.float32,
    ).astype(np.float32, copy=False)

    scale_factor, observed_femur_length_model_units, used_identity_scale = _estimate_scale_factor(
        joint_positions_body_frame,
        joint_confidence=joint_confidence,
        imputed_mask=imputed_mask,
        target_femur_length_m=float(target_femur_length_m),
        low_confidence_threshold=float(low_confidence_threshold),
    )
    joint_positions_metric_local = (
        joint_positions_body_frame * np.float32(scale_factor)
    ).astype(np.float32, copy=False)

    normalized_correction_masks = _normalize_leg_correction_masks(
        lower_limb_correction_masks,
        num_frames=int(sequence.num_frames),
    )
    tibia_corrected_positions, tibia_prior_applied_mask = _apply_tibia_length_prior(
        joint_positions_metric_local,
        target_tibia_length_m=float(target_tibia_length_m),
        leg_correction_masks=normalized_correction_masks,
    )
    corrected_joint_mask = _build_corrected_joint_mask(
        tibia_prior_applied_mask,
        num_frames=int(sequence.num_frames),
    )
    joint_positions_smoothed = _smooth_metric_local_pose(
        tibia_corrected_positions,
        corrected_joint_mask=corrected_joint_mask,
        window_length=int(smoothing_window_length),
        corrected_window_length=int(corrected_smoothing_window_length),
        polyorder=int(smoothing_polyorder),
    )

    metric_pose_sequence = PoseSequence3D(
        clip_id=str(sequence.clip_id),
        fps=None if sequence.fps is None else float(sequence.fps),
        fps_original=None if sequence.fps_original is None else float(sequence.fps_original),
        joint_names_3d=list(sequence.joint_names_3d),
        joint_positions_xyz=joint_positions_smoothed.astype(np.float32, copy=False),
        joint_confidence=joint_confidence.astype(np.float32, copy=False),
        skeleton_parents=list(sequence.skeleton_parents),
        frame_indices=np.asarray(sequence.frame_indices, dtype=np.int32),
        timestamps_sec=np.asarray(sequence.timestamps_sec, dtype=np.float32),
        source=f"{sequence.source}_metric",
        coordinate_space=BODY_METRIC_LOCAL_COORDINATE_SPACE,
        observed_mask=observed_mask.astype(bool, copy=False),
        imputed_mask=imputed_mask.astype(bool, copy=False),
    )

    lower_body_report = _build_lower_body_report(
        joint_positions=joint_positions_smoothed,
        joint_confidence=joint_confidence,
        observed_mask=observed_mask,
        correction_masks=tibia_prior_applied_mask,
        observed_window_frames=int(observed_window_frames),
        seated_knee_angle_threshold_deg=float(seated_knee_angle_threshold_deg),
        standing_knee_angle_threshold_deg=float(standing_knee_angle_threshold_deg),
    )

    normalization_result = {
        "joint_positions_3d_norm": joint_positions_3d_norm.astype(np.float32, copy=False),
        "joint_positions_body_frame": joint_positions_body_frame.astype(np.float32, copy=False),
        "joint_positions_metric_local": joint_positions_metric_local.astype(np.float32, copy=False),
        "joint_positions_metric_tibia_corrected": tibia_corrected_positions.astype(np.float32, copy=False),
        "joint_positions_smoothed": joint_positions_smoothed.astype(np.float32, copy=False),
        "scale_factor": float(scale_factor),
    }
    quality_report = _build_metric_normalizer_quality_report(
        sequence=sequence,
        metric_pose_sequence=metric_pose_sequence,
        scale_factor=float(scale_factor),
        target_femur_length_m=float(target_femur_length_m),
        target_tibia_length_m=float(target_tibia_length_m),
        observed_femur_length_model_units=observed_femur_length_model_units,
        body_frame_fallback_mask=body_frame_fallback_mask,
        tibia_prior_applied_mask=tibia_prior_applied_mask,
        lower_body_report=lower_body_report,
        smoothing_window_length=int(smoothing_window_length),
        corrected_smoothing_window_length=int(corrected_smoothing_window_length),
        smoothing_polyorder=int(smoothing_polyorder),
        used_identity_scale=bool(used_identity_scale),
    )
    artifacts = {
        "body_frame_fallback_mask": body_frame_fallback_mask.astype(bool, copy=False),
        "body_frame_rotation_matrices": body_frame_rotations.astype(np.float32, copy=False),
        "tibia_prior_applied_mask": {
            leg_name: mask.astype(bool, copy=False)
            for leg_name, mask in tibia_prior_applied_mask.items()
        },
        "corrected_joint_mask": corrected_joint_mask.astype(bool, copy=False),
        "lower_body_report": lower_body_report,
    }
    return {
        "pose_sequence": metric_pose_sequence,
        "normalization_result": normalization_result,
        "quality_report": quality_report,
        "artifacts": artifacts,
    }


def _build_metric_normalizer_quality_report(
    *,
    sequence: PoseSequence3D,
    metric_pose_sequence: PoseSequence3D,
    scale_factor: float,
    target_femur_length_m: float,
    target_tibia_length_m: float,
    observed_femur_length_model_units: float | None,
    body_frame_fallback_mask: np.ndarray,
    tibia_prior_applied_mask: Mapping[str, np.ndarray],
    lower_body_report: Mapping[str, Any],
    smoothing_window_length: int,
    corrected_smoothing_window_length: int,
    smoothing_polyorder: int,
    used_identity_scale: bool,
) -> Dict[str, Any]:
    notes = []
    body_frame_fallback_frames = int(np.count_nonzero(body_frame_fallback_mask))
    if body_frame_fallback_frames > 0:
        notes.append(f"body_frame_fallback_frames:{body_frame_fallback_frames}")
    if used_identity_scale:
        notes.append("scale_factor_fallback_to_identity")

    tibia_prior_frames = int(
        np.count_nonzero(np.asarray(tibia_prior_applied_mask["left_leg"], dtype=bool))
        + np.count_nonzero(np.asarray(tibia_prior_applied_mask["right_leg"], dtype=bool))
    )
    if tibia_prior_frames > 0:
        notes.append(f"tibia_prior_applied_frames:{tibia_prior_frames}")

    finite_positions = np.isfinite(np.asarray(metric_pose_sequence.joint_positions_xyz, dtype=np.float32)).all()
    contract_ok = list(metric_pose_sequence.joint_names_3d) == list(IMUGPT_22_JOINT_NAMES)
    metric_pose_ok = bool(finite_positions and contract_ok and scale_factor > 0.0)
    if not finite_positions:
        notes.append("metric_pose_contains_nan")
    if not contract_ok:
        notes.append("metric_joint_contract_mismatch")
    if not (scale_factor > 0.0):
        notes.append("invalid_scale_factor")

    status = "ok"
    if not metric_pose_ok:
        status = "fail"
    elif body_frame_fallback_frames > 0 or used_identity_scale or tibia_prior_frames > 0:
        status = "warning"

    return {
        "clip_id": str(metric_pose_sequence.clip_id),
        "status": str(status),
        "fps": None if metric_pose_sequence.fps is None else float(metric_pose_sequence.fps),
        "fps_original": (
            None if metric_pose_sequence.fps_original is None else float(metric_pose_sequence.fps_original)
        ),
        "num_frames": int(metric_pose_sequence.num_frames),
        "num_joints": int(metric_pose_sequence.num_joints),
        "input_joint_format": list(sequence.joint_names_3d),
        "output_joint_format": list(metric_pose_sequence.joint_names_3d),
        "input_coordinate_space": str(sequence.coordinate_space),
        "coordinate_space": str(metric_pose_sequence.coordinate_space),
        "metric_pose_ok": bool(metric_pose_ok),
        "scale_factor": float(scale_factor),
        "target_femur_length_m": float(target_femur_length_m),
        "target_tibia_length_m": float(target_tibia_length_m),
        "observed_femur_length_model_units": (
            None if observed_femur_length_model_units is None else float(observed_femur_length_model_units)
        ),
        "scale_reference": "median_femur_length",
        "body_frame_fallback_frames": int(body_frame_fallback_frames),
        "body_frame_fallback_ratio": (
            float(body_frame_fallback_frames) / float(metric_pose_sequence.num_frames)
            if metric_pose_sequence.num_frames > 0
            else 0.0
        ),
        "tibia_prior_applied_frames": int(tibia_prior_frames),
        "smoothing_window_length": int(smoothing_window_length),
        "corrected_smoothing_window_length": int(corrected_smoothing_window_length),
        "smoothing_polyorder": int(smoothing_polyorder),
        "assumptions": [
            "body_frame_from_confident_hips_and_neck",
            "single_subject_per_clip",
            "anthropometric_scale_prior",
            "conservative_tibia_prior_under_occlusion",
        ],
        "limitations": [
            "not_global_pose",
            "not_absolute_depth",
            "heuristic_metric_scale",
        ],
        "lower_body_report": dict(lower_body_report),
        "notes": list(dict.fromkeys(notes)),
    }


def _validate_imugpt22_sequence(sequence: PoseSequence3D) -> None:
    joint_names = [str(name) for name in sequence.joint_names_3d]
    if joint_names != list(IMUGPT_22_JOINT_NAMES):
        raise ValueError("Metric normalizer expects stage-5.6 output ordered as IMUGPT_22_JOINT_NAMES.")
    points = np.asarray(sequence.joint_positions_xyz, dtype=np.float32)
    confidence = np.asarray(sequence.joint_confidence, dtype=np.float32)
    if points.ndim != 3 or points.shape[1:] != (len(IMUGPT_22_JOINT_NAMES), 3):
        raise ValueError("Metric normalizer expects joint_positions_xyz with shape [T, 22, 3].")
    if confidence.shape != points.shape[:2]:
        raise ValueError("Metric normalizer expects joint_confidence with shape [T, 22].")
    if not np.isfinite(points).all():
        raise ValueError("Metric normalizer expects finite joint_positions_xyz without NaN.")


def _build_body_frame_rotation_matrices(
    positions: np.ndarray,
    *,
    joint_confidence: np.ndarray,
    observed_mask: np.ndarray,
    imputed_mask: np.ndarray,
    low_confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(positions, dtype=np.float32)
    confidence = np.asarray(joint_confidence, dtype=np.float32)
    observed = np.asarray(observed_mask, dtype=bool)
    imputed = np.asarray(imputed_mask, dtype=bool)
    num_frames = int(points.shape[0])
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], num_frames, axis=0)
    fallback_mask = np.zeros((num_frames,), dtype=bool)
    previous_valid_rotation = None

    required_joint_indices = (
        _IMUGPT22_INDEX["Pelvis"],
        _IMUGPT22_INDEX["Left_hip"],
        _IMUGPT22_INDEX["Right_hip"],
        _IMUGPT22_INDEX["Neck"],
    )

    for frame_index in range(num_frames):
        if not _frame_has_confident_body_axes(
            points=points,
            confidence=confidence,
            observed_mask=observed,
            imputed_mask=imputed,
            frame_index=frame_index,
            joint_indices=required_joint_indices,
            low_confidence_threshold=float(low_confidence_threshold),
        ):
            fallback_mask[frame_index] = True
            if previous_valid_rotation is not None:
                rotations[frame_index] = previous_valid_rotation
            continue

        pelvis = points[frame_index, _IMUGPT22_INDEX["Pelvis"]]
        neck = points[frame_index, _IMUGPT22_INDEX["Neck"]]
        left_hip = points[frame_index, _IMUGPT22_INDEX["Left_hip"]]
        right_hip = points[frame_index, _IMUGPT22_INDEX["Right_hip"]]

        lateral = right_hip - left_hip
        vertical = neck - pelvis
        lateral_norm = np.linalg.norm(lateral)
        vertical_norm = np.linalg.norm(vertical)
        if lateral_norm > _EPSILON and vertical_norm > _EPSILON:
            x_axis = lateral / lateral_norm
            y_seed = vertical / vertical_norm
            z_axis = np.cross(x_axis, y_seed)
            z_norm = np.linalg.norm(z_axis)
            if z_norm > _EPSILON:
                z_axis = z_axis / z_norm
                y_axis = np.cross(z_axis, x_axis)
                y_norm = np.linalg.norm(y_axis)
                if y_norm > _EPSILON:
                    y_axis = y_axis / y_norm
                    rotations[frame_index] = np.stack([x_axis, y_axis, z_axis], axis=1).astype(
                        np.float32,
                        copy=False,
                    )
                    previous_valid_rotation = rotations[frame_index].copy()
                    continue

        fallback_mask[frame_index] = True
        if previous_valid_rotation is not None:
            rotations[frame_index] = previous_valid_rotation

    return rotations, fallback_mask


def _frame_has_confident_body_axes(
    *,
    points: np.ndarray,
    confidence: np.ndarray,
    observed_mask: np.ndarray,
    imputed_mask: np.ndarray,
    frame_index: int,
    joint_indices: tuple[int, ...],
    low_confidence_threshold: float,
) -> bool:
    for joint_index in joint_indices:
        if not np.isfinite(points[frame_index, joint_index]).all():
            return False
        if float(confidence[frame_index, joint_index]) < float(low_confidence_threshold):
            return False
        if bool(imputed_mask[frame_index, joint_index]):
            return False
        if not bool(observed_mask[frame_index, joint_index]):
            return False
    return True


def _estimate_scale_factor(
    body_positions: np.ndarray,
    *,
    joint_confidence: np.ndarray,
    imputed_mask: np.ndarray,
    target_femur_length_m: float,
    low_confidence_threshold: float,
) -> tuple[float, float | None, bool]:
    points = np.asarray(body_positions, dtype=np.float32)
    confidence = np.asarray(joint_confidence, dtype=np.float32)
    imputed = np.asarray(imputed_mask, dtype=bool)
    observed_segments = []

    for hip_name, knee_name in (("Left_hip", "Left_knee"), ("Right_hip", "Right_knee")):
        hip_index = _IMUGPT22_INDEX[hip_name]
        knee_index = _IMUGPT22_INDEX[knee_name]
        segment = points[:, knee_index] - points[:, hip_index]
        lengths = np.linalg.norm(segment, axis=1)
        valid_mask = (
            np.isfinite(lengths)
            & (lengths > _EPSILON)
            & (confidence[:, hip_index] >= float(low_confidence_threshold))
            & (confidence[:, knee_index] >= float(low_confidence_threshold))
            & ~imputed[:, hip_index]
            & ~imputed[:, knee_index]
        )
        if np.any(valid_mask):
            observed_segments.append(lengths[valid_mask])

    if len(observed_segments) == 0:
        return 1.0, None, True

    observed_femur_length = float(np.median(np.concatenate(observed_segments, axis=0)))
    if observed_femur_length <= _EPSILON:
        return 1.0, observed_femur_length, True
    return float(target_femur_length_m) / observed_femur_length, observed_femur_length, False


def _normalize_leg_correction_masks(
    correction_masks: Mapping[str, np.ndarray] | None,
    *,
    num_frames: int,
) -> Dict[str, np.ndarray]:
    normalized = {
        "left_leg": np.zeros((int(num_frames),), dtype=bool),
        "right_leg": np.zeros((int(num_frames),), dtype=bool),
    }
    if correction_masks is None:
        return normalized
    for leg_name in normalized:
        if leg_name not in correction_masks:
            continue
        leg_mask = np.asarray(correction_masks[leg_name], dtype=bool)
        if leg_mask.shape != normalized[leg_name].shape:
            raise ValueError(
                f"Metric normalizer correction mask for {leg_name} must have shape {normalized[leg_name].shape}, "
                f"got {leg_mask.shape}."
            )
        normalized[leg_name] = leg_mask
    return normalized


def _apply_tibia_length_prior(
    metric_local_positions: np.ndarray,
    *,
    target_tibia_length_m: float,
    leg_correction_masks: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    points = np.asarray(metric_local_positions, dtype=np.float32).copy()
    correction_masks = {
        leg_name: np.asarray(mask, dtype=bool).copy()
        for leg_name, mask in leg_correction_masks.items()
    }
    previous_direction = {
        "left_leg": None,
        "right_leg": None,
    }

    for leg_name, leg_config in _LEG_CONFIGS.items():
        knee_index = int(leg_config["knee"])
        ankle_index = int(leg_config["ankle"])
        for frame_index in range(points.shape[0]):
            if not bool(correction_masks[leg_name][frame_index]):
                current_direction = _safe_normalize(
                    points[frame_index, ankle_index] - points[frame_index, knee_index]
                )
                if current_direction is not None:
                    previous_direction[leg_name] = current_direction
                continue

            shank = points[frame_index, ankle_index] - points[frame_index, knee_index]
            direction = _safe_normalize(shank)
            if direction is None:
                direction = previous_direction[leg_name]
            if direction is None:
                direction = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
            previous_direction[leg_name] = direction
            points[frame_index, ankle_index] = (
                points[frame_index, knee_index] + (direction * np.float32(target_tibia_length_m))
            ).astype(np.float32, copy=False)

    return points.astype(np.float32, copy=False), correction_masks


def _build_corrected_joint_mask(
    tibia_prior_applied_mask: Mapping[str, np.ndarray],
    *,
    num_frames: int,
) -> np.ndarray:
    corrected_mask = np.zeros((int(num_frames), len(IMUGPT_22_JOINT_NAMES)), dtype=bool)
    corrected_mask[:, _IMUGPT22_INDEX["Left_knee"]] = np.asarray(
        tibia_prior_applied_mask["left_leg"],
        dtype=bool,
    )
    corrected_mask[:, _IMUGPT22_INDEX["Left_ankle"]] = np.asarray(
        tibia_prior_applied_mask["left_leg"],
        dtype=bool,
    )
    corrected_mask[:, _IMUGPT22_INDEX["Right_knee"]] = np.asarray(
        tibia_prior_applied_mask["right_leg"],
        dtype=bool,
    )
    corrected_mask[:, _IMUGPT22_INDEX["Right_ankle"]] = np.asarray(
        tibia_prior_applied_mask["right_leg"],
        dtype=bool,
    )
    return corrected_mask


def _smooth_metric_local_pose(
    metric_local_positions: np.ndarray,
    *,
    corrected_joint_mask: np.ndarray,
    window_length: int,
    corrected_window_length: int,
    polyorder: int,
) -> np.ndarray:
    points = np.asarray(metric_local_positions, dtype=np.float32).copy()
    corrected_mask = np.asarray(corrected_joint_mask, dtype=bool)
    for joint_index in range(points.shape[1]):
        requested_window = (
            int(corrected_window_length)
            if bool(np.any(corrected_mask[:, joint_index]))
            else int(window_length)
        )
        points[:, joint_index] = savgol_smooth(
            points[:, joint_index],
            window_length=int(requested_window),
            polyorder=int(polyorder),
        )
    return points.astype(np.float32, copy=False)


def _build_lower_body_report(
    *,
    joint_positions: np.ndarray,
    joint_confidence: np.ndarray,
    observed_mask: np.ndarray,
    correction_masks: Mapping[str, np.ndarray],
    observed_window_frames: int,
    seated_knee_angle_threshold_deg: float,
    standing_knee_angle_threshold_deg: float,
) -> Dict[str, Dict[str, Any]]:
    report: Dict[str, Dict[str, Any]] = {}
    for leg_name, leg_config in _LEG_CONFIGS.items():
        hip_index = int(leg_config["hip"])
        knee_index = int(leg_config["knee"])
        ankle_index = int(leg_config["ankle"])
        knee_angles = _compute_knee_angles(
            joint_positions[:, hip_index],
            joint_positions[:, knee_index],
            joint_positions[:, ankle_index],
        )
        tibia_length = np.linalg.norm(
            joint_positions[:, ankle_index] - joint_positions[:, knee_index],
            axis=1,
        )
        observed_ratio = _compute_trailing_observed_ratio(
            observed_mask[:, [knee_index, ankle_index]],
            window_frames=int(observed_window_frames),
        )
        mean_confidence = np.mean(joint_confidence[:, [knee_index, ankle_index]], axis=1, dtype=np.float32)
        posture_state = [
            _classify_posture(
                knee_angle_deg=float(angle),
                seated_threshold_deg=float(seated_knee_angle_threshold_deg),
                standing_threshold_deg=float(standing_knee_angle_threshold_deg),
            )
            for angle in knee_angles
        ]
        report[leg_name] = {
            "knee_angle_deg": [None if np.isnan(value) else float(value) for value in knee_angles],
            "tibia_length_m": [None if np.isnan(value) else float(value) for value in tibia_length],
            "observed_ratio": [float(value) for value in observed_ratio],
            "mean_confidence": [float(value) for value in mean_confidence],
            "correction_applied": [bool(value) for value in np.asarray(correction_masks[leg_name], dtype=bool)],
            "posture_state": posture_state,
        }
    return report


def _compute_trailing_observed_ratio(observed_mask: np.ndarray, *, window_frames: int) -> np.ndarray:
    mask = np.asarray(observed_mask, dtype=bool)
    ratio = np.zeros((mask.shape[0],), dtype=np.float32)
    for frame_index in range(mask.shape[0]):
        start = max(0, frame_index - int(window_frames) + 1)
        ratio[frame_index] = float(np.mean(mask[start : frame_index + 1])) if frame_index >= start else 0.0
    return ratio


def _compute_knee_angles(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> np.ndarray:
    thigh = np.asarray(hip, dtype=np.float32) - np.asarray(knee, dtype=np.float32)
    shank = np.asarray(ankle, dtype=np.float32) - np.asarray(knee, dtype=np.float32)
    thigh_norm = np.linalg.norm(thigh, axis=1)
    shank_norm = np.linalg.norm(shank, axis=1)
    valid_mask = (
        np.isfinite(thigh).all(axis=1)
        & np.isfinite(shank).all(axis=1)
        & (thigh_norm > _EPSILON)
        & (shank_norm > _EPSILON)
    )
    angles = np.full((thigh.shape[0],), np.nan, dtype=np.float32)
    if not np.any(valid_mask):
        return angles
    cosine = np.sum(thigh[valid_mask] * shank[valid_mask], axis=1) / (thigh_norm[valid_mask] * shank_norm[valid_mask])
    cosine = np.clip(cosine, -1.0, 1.0)
    angles[valid_mask] = np.rad2deg(np.arccos(cosine)).astype(np.float32, copy=False)
    return angles


def _classify_posture(
    *,
    knee_angle_deg: float,
    seated_threshold_deg: float,
    standing_threshold_deg: float,
) -> str:
    if not np.isfinite(knee_angle_deg):
        return "uncertain"
    if float(knee_angle_deg) < float(seated_threshold_deg):
        return "seated"
    if float(knee_angle_deg) > float(standing_threshold_deg):
        return "standing"
    return "uncertain"


def _safe_normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(np.asarray(vector, dtype=np.float32)))
    if not np.isfinite(norm) or norm <= _EPSILON:
        return None
    return (np.asarray(vector, dtype=np.float32) / norm).astype(np.float32, copy=False)
