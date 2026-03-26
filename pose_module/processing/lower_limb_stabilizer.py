"""Stage 5.5b: conservative lower-limb stabilization under lower-body occlusion."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from pose_module.interfaces import MOTIONBERT_17_JOINT_NAMES, MOTIONBERT_17_PARENT_INDICES, PoseSequence3D

DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.20
DEFAULT_MIN_OBSERVED_RATIO = 0.5
DEFAULT_OBSERVED_WINDOW_FRAMES = 15
DEFAULT_HISTORY_WINDOW_FRAMES = 30
DEFAULT_SEATED_KNEE_ANGLE_THRESHOLD_DEG = 140.0
DEFAULT_STANDING_KNEE_ANGLE_THRESHOLD_DEG = 160.0
DEFAULT_CORRECTED_JOINT_CONFIDENCE = 0.10
_EPSILON = 1e-6

_MB17_INDEX = {name: index for index, name in enumerate(MOTIONBERT_17_JOINT_NAMES)}
_LEG_CONFIGS = {
    "left_leg": {
        "hip": _MB17_INDEX["left_hip"],
        "knee": _MB17_INDEX["left_knee"],
        "ankle": _MB17_INDEX["left_ankle"],
    },
    "right_leg": {
        "hip": _MB17_INDEX["right_hip"],
        "knee": _MB17_INDEX["right_knee"],
        "ankle": _MB17_INDEX["right_ankle"],
    },
}


def run_lower_limb_stabilizer(
    sequence: PoseSequence3D,
    *,
    low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    min_observed_ratio: float = DEFAULT_MIN_OBSERVED_RATIO,
    observed_window_frames: int = DEFAULT_OBSERVED_WINDOW_FRAMES,
    history_window_frames: int = DEFAULT_HISTORY_WINDOW_FRAMES,
    seated_knee_angle_threshold_deg: float = DEFAULT_SEATED_KNEE_ANGLE_THRESHOLD_DEG,
    standing_knee_angle_threshold_deg: float = DEFAULT_STANDING_KNEE_ANGLE_THRESHOLD_DEG,
    corrected_joint_confidence: float = DEFAULT_CORRECTED_JOINT_CONFIDENCE,
) -> Dict[str, Any]:
    _validate_motionbert17_sequence(sequence)

    points = np.asarray(sequence.joint_positions_xyz, dtype=np.float32).copy()
    confidence = np.asarray(sequence.joint_confidence, dtype=np.float32).copy()
    observed_mask = np.asarray(sequence.resolved_observed_mask(), dtype=bool).copy()
    imputed_mask = np.asarray(sequence.resolved_imputed_mask(), dtype=bool).copy()

    lower_body_report: Dict[str, Dict[str, Any]] = {}
    correction_masks: Dict[str, np.ndarray] = {}
    notes = []
    corrected_joint_confidence = float(corrected_joint_confidence)

    for leg_name, leg_config in _LEG_CONFIGS.items():
        leg_result = _stabilize_leg(
            points=points,
            confidence=confidence,
            observed_mask=observed_mask,
            imputed_mask=imputed_mask,
            hip_index=int(leg_config["hip"]),
            knee_index=int(leg_config["knee"]),
            ankle_index=int(leg_config["ankle"]),
            low_confidence_threshold=float(low_confidence_threshold),
            min_observed_ratio=float(min_observed_ratio),
            observed_window_frames=int(observed_window_frames),
            history_window_frames=int(history_window_frames),
            seated_knee_angle_threshold_deg=float(seated_knee_angle_threshold_deg),
            standing_knee_angle_threshold_deg=float(standing_knee_angle_threshold_deg),
            corrected_joint_confidence=float(corrected_joint_confidence),
        )
        lower_body_report[leg_name] = leg_result["report"]
        correction_masks[leg_name] = leg_result["correction_mask"].astype(bool, copy=False)
        if int(np.count_nonzero(leg_result["correction_mask"])) > 0:
            notes.append(f"{leg_name}_correction_frames:{int(np.count_nonzero(leg_result['correction_mask']))}")
        if int(np.count_nonzero(leg_result["posture_states"] == "uncertain")) > 0:
            notes.append(
                f"{leg_name}_uncertain_frames:{int(np.count_nonzero(leg_result['posture_states'] == 'uncertain'))}"
            )

    stabilized_sequence = PoseSequence3D(
        clip_id=str(sequence.clip_id),
        fps=None if sequence.fps is None else float(sequence.fps),
        fps_original=None if sequence.fps_original is None else float(sequence.fps_original),
        joint_names_3d=list(sequence.joint_names_3d),
        joint_positions_xyz=points.astype(np.float32, copy=False),
        joint_confidence=confidence.astype(np.float32, copy=False),
        skeleton_parents=list(sequence.skeleton_parents),
        frame_indices=np.asarray(sequence.frame_indices, dtype=np.int32),
        timestamps_sec=np.asarray(sequence.timestamps_sec, dtype=np.float32),
        source=f"{sequence.source}_lower_limb_stabilized",
        coordinate_space=str(sequence.coordinate_space),
        observed_mask=observed_mask.astype(bool, copy=False),
        imputed_mask=imputed_mask.astype(bool, copy=False),
    )

    status = "ok"
    if any(int(np.count_nonzero(mask)) > 0 for mask in correction_masks.values()):
        status = "warning"

    quality_report = {
        "clip_id": str(sequence.clip_id),
        "status": str(status),
        "fps": None if sequence.fps is None else float(sequence.fps),
        "fps_original": None if sequence.fps_original is None else float(sequence.fps_original),
        "num_frames": int(sequence.num_frames),
        "num_joints": int(sequence.num_joints),
        "input_joint_format": list(sequence.joint_names_3d),
        "output_joint_format": list(stabilized_sequence.joint_names_3d),
        "coordinate_space": str(sequence.coordinate_space),
        "low_confidence_threshold": float(low_confidence_threshold),
        "min_observed_ratio": float(min_observed_ratio),
        "observed_window_frames": int(observed_window_frames),
        "history_window_frames": int(history_window_frames),
        "seated_knee_angle_threshold_deg": float(seated_knee_angle_threshold_deg),
        "standing_knee_angle_threshold_deg": float(standing_knee_angle_threshold_deg),
        "left_leg_correction_frames": int(np.count_nonzero(correction_masks["left_leg"])),
        "right_leg_correction_frames": int(np.count_nonzero(correction_masks["right_leg"])),
        "left_leg_uncertain_frames": int(
            np.count_nonzero(np.asarray(lower_body_report["left_leg"]["posture_state"], dtype=object) == "uncertain")
        ),
        "right_leg_uncertain_frames": int(
            np.count_nonzero(np.asarray(lower_body_report["right_leg"]["posture_state"], dtype=object) == "uncertain")
        ),
        "lower_body_report": lower_body_report,
        "notes": list(dict.fromkeys(notes)),
    }
    artifacts = {
        "correction_masks": {
            leg_name: mask.astype(bool, copy=False)
            for leg_name, mask in correction_masks.items()
        },
        "lower_body_report": lower_body_report,
    }
    return {
        "pose_sequence": stabilized_sequence,
        "quality_report": quality_report,
        "artifacts": artifacts,
    }


def _stabilize_leg(
    *,
    points: np.ndarray,
    confidence: np.ndarray,
    observed_mask: np.ndarray,
    imputed_mask: np.ndarray,
    hip_index: int,
    knee_index: int,
    ankle_index: int,
    low_confidence_threshold: float,
    min_observed_ratio: float,
    observed_window_frames: int,
    history_window_frames: int,
    seated_knee_angle_threshold_deg: float,
    standing_knee_angle_threshold_deg: float,
    corrected_joint_confidence: float,
) -> Dict[str, Any]:
    knee_angles = _compute_knee_angles(
        points[:, hip_index],
        points[:, knee_index],
        points[:, ankle_index],
    )
    observed_ratio = _compute_trailing_observed_ratio(
        observed_mask[:, [knee_index, ankle_index]],
        window_frames=int(observed_window_frames),
    )
    mean_confidence = np.mean(confidence[:, [knee_index, ankle_index]], axis=1, dtype=np.float32)
    correction_mask = np.zeros((points.shape[0],), dtype=bool)
    posture_states = np.asarray(
        [_classify_posture(angle, seated_knee_angle_threshold_deg, standing_knee_angle_threshold_deg) for angle in knee_angles],
        dtype=object,
    )

    reliable_frame_mask = (
        np.isfinite(points[:, [hip_index, knee_index, ankle_index]]).all(axis=(1, 2))
        & (confidence[:, knee_index] >= float(low_confidence_threshold))
        & (confidence[:, ankle_index] >= float(low_confidence_threshold))
        & observed_mask[:, knee_index]
        & observed_mask[:, ankle_index]
        & ~imputed_mask[:, knee_index]
        & ~imputed_mask[:, ankle_index]
    )

    for frame_index in range(points.shape[0]):
        low_conf_frame = bool(
            not np.isfinite(points[frame_index, [hip_index, knee_index, ankle_index]]).all()
            or confidence[frame_index, knee_index] < float(low_confidence_threshold)
            or confidence[frame_index, ankle_index] < float(low_confidence_threshold)
            or imputed_mask[frame_index, knee_index]
            or imputed_mask[frame_index, ankle_index]
            or observed_ratio[frame_index] < float(min_observed_ratio)
        )
        if not low_conf_frame:
            continue

        correction_mask[frame_index] = True
        anchor_index = _find_anchor_frame(
            reliable_frame_mask=reliable_frame_mask,
            knee_angles=knee_angles,
            frame_index=frame_index,
            history_window_frames=int(history_window_frames),
            seated_knee_angle_threshold_deg=float(seated_knee_angle_threshold_deg),
        )
        if anchor_index is not None:
            posture_states[frame_index] = "seated"
            _apply_anchor_stabilization(
                points=points,
                confidence=confidence,
                observed_mask=observed_mask,
                imputed_mask=imputed_mask,
                frame_index=frame_index,
                anchor_index=int(anchor_index),
                hip_index=int(hip_index),
                knee_index=int(knee_index),
                ankle_index=int(ankle_index),
                corrected_joint_confidence=float(corrected_joint_confidence),
                low_confidence_threshold=float(low_confidence_threshold),
            )
        else:
            posture_states[frame_index] = "uncertain"
            _mark_low_confidence_leg(
                confidence=confidence,
                observed_mask=observed_mask,
                imputed_mask=imputed_mask,
                frame_index=frame_index,
                knee_index=int(knee_index),
                ankle_index=int(ankle_index),
                corrected_joint_confidence=float(corrected_joint_confidence),
                low_confidence_threshold=float(low_confidence_threshold),
            )
            _ensure_leg_is_finite(
                points=points,
                frame_index=frame_index,
                hip_index=int(hip_index),
                knee_index=int(knee_index),
                ankle_index=int(ankle_index),
                reliable_frame_mask=reliable_frame_mask,
            )
            _limit_uncertain_knee_extension(
                points=points,
                frame_index=frame_index,
                hip_index=int(hip_index),
                knee_index=int(knee_index),
                ankle_index=int(ankle_index),
                standing_knee_angle_threshold_deg=float(standing_knee_angle_threshold_deg),
            )

    tibia_length = np.linalg.norm(points[:, ankle_index] - points[:, knee_index], axis=1)
    corrected_knee_angles = _compute_knee_angles(
        points[:, hip_index],
        points[:, knee_index],
        points[:, ankle_index],
    )
    report = {
        "knee_angle_deg": [None if np.isnan(value) else float(value) for value in corrected_knee_angles],
        "tibia_length_model_units": [None if np.isnan(value) else float(value) for value in tibia_length],
        "observed_ratio": [float(value) for value in observed_ratio],
        "mean_confidence": [float(value) for value in mean_confidence],
        "correction_applied": [bool(value) for value in correction_mask],
        "posture_state": [str(value) for value in posture_states.tolist()],
    }
    return {
        "report": report,
        "correction_mask": correction_mask,
        "posture_states": posture_states,
    }


def _apply_anchor_stabilization(
    *,
    points: np.ndarray,
    confidence: np.ndarray,
    observed_mask: np.ndarray,
    imputed_mask: np.ndarray,
    frame_index: int,
    anchor_index: int,
    hip_index: int,
    knee_index: int,
    ankle_index: int,
    corrected_joint_confidence: float,
    low_confidence_threshold: float,
) -> None:
    current_hip = points[frame_index, hip_index]
    anchor_hip = points[anchor_index, hip_index]
    anchor_knee = points[anchor_index, knee_index]
    anchor_ankle = points[anchor_index, ankle_index]

    if not np.isfinite(current_hip).all():
        current_hip = anchor_hip

    knee_is_low_conf = bool(
        not np.isfinite(points[frame_index, knee_index]).all()
        or confidence[frame_index, knee_index] < float(low_confidence_threshold)
        or imputed_mask[frame_index, knee_index]
    )
    knee_position = points[frame_index, knee_index]
    if knee_is_low_conf:
        anchor_femur = anchor_knee - anchor_hip
        femur_direction = _safe_normalize(anchor_femur)
        femur_length = _vector_norm(points[frame_index, knee_index] - current_hip)
        if not np.isfinite(femur_length) or femur_length <= _EPSILON:
            femur_length = max(_vector_norm(anchor_femur), _EPSILON)
        knee_position = current_hip + (femur_direction * femur_length)
        points[frame_index, knee_index] = knee_position.astype(np.float32, copy=False)
        confidence[frame_index, knee_index] = min(
            float(confidence[frame_index, knee_index]),
            float(corrected_joint_confidence),
        )
        observed_mask[frame_index, knee_index] = False
        imputed_mask[frame_index, knee_index] = True

    anchor_tibia = anchor_ankle - anchor_knee
    tibia_direction = _safe_normalize(anchor_tibia)
    tibia_length = _vector_norm(points[frame_index, ankle_index] - knee_position)
    if not np.isfinite(tibia_length) or tibia_length <= _EPSILON:
        tibia_length = max(_vector_norm(anchor_tibia), _EPSILON)
    ankle_position = knee_position + (tibia_direction * tibia_length)
    points[frame_index, ankle_index] = ankle_position.astype(np.float32, copy=False)
    confidence[frame_index, ankle_index] = min(
        float(confidence[frame_index, ankle_index]),
        float(corrected_joint_confidence),
    )
    observed_mask[frame_index, ankle_index] = False
    imputed_mask[frame_index, ankle_index] = True


def _mark_low_confidence_leg(
    *,
    confidence: np.ndarray,
    observed_mask: np.ndarray,
    imputed_mask: np.ndarray,
    frame_index: int,
    knee_index: int,
    ankle_index: int,
    corrected_joint_confidence: float,
    low_confidence_threshold: float,
) -> None:
    for joint_index in (knee_index, ankle_index):
        if confidence[frame_index, joint_index] < float(low_confidence_threshold) or imputed_mask[frame_index, joint_index]:
            confidence[frame_index, joint_index] = min(
                float(confidence[frame_index, joint_index]),
                float(corrected_joint_confidence),
            )
            observed_mask[frame_index, joint_index] = False
            imputed_mask[frame_index, joint_index] = True


def _ensure_leg_is_finite(
    *,
    points: np.ndarray,
    frame_index: int,
    hip_index: int,
    knee_index: int,
    ankle_index: int,
    reliable_frame_mask: np.ndarray,
) -> None:
    if np.isfinite(points[frame_index, [hip_index, knee_index, ankle_index]]).all():
        return

    previous_reliable = np.flatnonzero(reliable_frame_mask[:frame_index])
    if previous_reliable.size == 0:
        return

    anchor_index = int(previous_reliable[-1])
    hip_position = points[frame_index, hip_index]
    if not np.isfinite(hip_position).all():
        hip_position = points[anchor_index, hip_index]
        points[frame_index, hip_index] = hip_position.astype(np.float32, copy=False)

    if not np.isfinite(points[frame_index, knee_index]).all():
        anchor_femur = points[anchor_index, knee_index] - points[anchor_index, hip_index]
        points[frame_index, knee_index] = (hip_position + anchor_femur).astype(np.float32, copy=False)
    if not np.isfinite(points[frame_index, ankle_index]).all():
        anchor_tibia = points[anchor_index, ankle_index] - points[anchor_index, knee_index]
        points[frame_index, ankle_index] = (
            points[frame_index, knee_index] + anchor_tibia
        ).astype(np.float32, copy=False)


def _limit_uncertain_knee_extension(
    *,
    points: np.ndarray,
    frame_index: int,
    hip_index: int,
    knee_index: int,
    ankle_index: int,
    standing_knee_angle_threshold_deg: float,
) -> None:
    hip = points[frame_index, hip_index]
    knee = points[frame_index, knee_index]
    ankle = points[frame_index, ankle_index]
    angle = _compute_knee_angles(hip[None, :], knee[None, :], ankle[None, :])[0]
    if not np.isfinite(angle) or float(angle) <= float(standing_knee_angle_threshold_deg):
        return

    target_angle_deg = min(float(standing_knee_angle_threshold_deg) - 10.0, 150.0)
    thigh_dir = _safe_normalize(hip - knee)
    shank = ankle - knee
    shank_len = _vector_norm(shank)
    if thigh_dir is None or not np.isfinite(shank_len) or shank_len <= _EPSILON:
        return

    orthogonal_component = shank - (np.dot(shank, thigh_dir) * thigh_dir)
    bend_dir = _safe_normalize(orthogonal_component)
    if bend_dir is None:
        bend_dir = _orthogonal_unit_vector(thigh_dir)

    target_angle_rad = np.deg2rad(target_angle_deg)
    new_shank_dir = (
        (np.cos(target_angle_rad) * thigh_dir)
        + (np.sin(target_angle_rad) * bend_dir)
    ).astype(np.float32, copy=False)
    new_shank_dir = _safe_normalize(new_shank_dir)
    if new_shank_dir is None:
        return
    points[frame_index, ankle_index] = (knee + (new_shank_dir * shank_len)).astype(np.float32, copy=False)


def _find_anchor_frame(
    *,
    reliable_frame_mask: np.ndarray,
    knee_angles: np.ndarray,
    frame_index: int,
    history_window_frames: int,
    seated_knee_angle_threshold_deg: float,
) -> int | None:
    history_start = max(0, int(frame_index) - int(history_window_frames))
    candidates = np.flatnonzero(
        reliable_frame_mask[history_start:frame_index]
        & np.isfinite(knee_angles[history_start:frame_index])
        & (knee_angles[history_start:frame_index] < float(seated_knee_angle_threshold_deg))
    )
    if candidates.size == 0:
        return None
    return int(history_start + candidates[-1])


def _compute_trailing_observed_ratio(observed_mask: np.ndarray, *, window_frames: int) -> np.ndarray:
    mask = np.asarray(observed_mask, dtype=bool)
    ratio = np.zeros((mask.shape[0],), dtype=np.float32)
    for frame_index in range(mask.shape[0]):
        start = max(0, frame_index - int(window_frames) + 1)
        window = mask[start : frame_index + 1]
        ratio[frame_index] = float(np.mean(window)) if window.size > 0 else 0.0
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


def _safe_normalize(vector: np.ndarray) -> np.ndarray:
    norm = _vector_norm(vector)
    if not np.isfinite(norm) or norm <= _EPSILON:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return (np.asarray(vector, dtype=np.float32) / norm).astype(np.float32, copy=False)


def _vector_norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=np.float32)))


def _orthogonal_unit_vector(vector: np.ndarray) -> np.ndarray:
    candidate = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(candidate, vector))) > 0.9:
        candidate = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    orthogonal = candidate - (np.dot(candidate, vector) * vector)
    normalized = _safe_normalize(orthogonal)
    return np.asarray([1.0, 0.0, 0.0], dtype=np.float32) if normalized is None else normalized


def _validate_motionbert17_sequence(sequence: PoseSequence3D) -> None:
    if list(sequence.joint_names_3d) != list(MOTIONBERT_17_JOINT_NAMES):
        raise ValueError(
            "Lower-limb stabilizer expects MotionBERT17 output ordered as MOTIONBERT_17_JOINT_NAMES."
        )
    points = np.asarray(sequence.joint_positions_xyz, dtype=np.float32)
    confidence = np.asarray(sequence.joint_confidence, dtype=np.float32)
    if points.ndim != 3 or points.shape[1:] != (len(MOTIONBERT_17_JOINT_NAMES), 3):
        raise ValueError("Lower-limb stabilizer expects joint_positions_xyz with shape [T, 17, 3].")
    if confidence.shape != points.shape[:2]:
        raise ValueError("Lower-limb stabilizer expects joint_confidence with shape [T, 17].")
    if list(sequence.skeleton_parents) != list(MOTIONBERT_17_PARENT_INDICES):
        raise ValueError("Lower-limb stabilizer expects MotionBERT17 parent indices.")
