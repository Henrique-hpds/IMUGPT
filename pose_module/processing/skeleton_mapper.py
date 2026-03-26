"""Stage 5.6: map MotionBERT MB17 poses to the fixed IMUGPT22 skeleton."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from pose_module.interfaces import (
    IMUGPT_22_JOINT_NAMES,
    IMUGPT_22_PARENT_INDICES,
    MOTIONBERT_17_JOINT_NAMES,
    PoseSequence3D,
)

DEFAULT_FOOT_LENGTH_M = 0.12
_EPSILON = 1e-6

_MB17_INDEX = {name: index for index, name in enumerate(MOTIONBERT_17_JOINT_NAMES)}
_IMUGPT22_INDEX = {name: index for index, name in enumerate(IMUGPT_22_JOINT_NAMES)}
_DIRECT_COPY_MAP = {
    "Pelvis": "pelvis",
    "Left_hip": "left_hip",
    "Right_hip": "right_hip",
    "Left_knee": "left_knee",
    "Right_knee": "right_knee",
    "Left_ankle": "left_ankle",
    "Right_ankle": "right_ankle",
    "Neck": "neck",
    "Head": "head",
    "Left_shoulder": "left_shoulder",
    "Right_shoulder": "right_shoulder",
    "Left_elbow": "left_elbow",
    "Right_elbow": "right_elbow",
    "Left_wrist": "left_wrist",
    "Right_wrist": "right_wrist",
}
_LEFT_RIGHT_MB17_PAIRS = (
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
)


def map_pose_sequence_to_imugpt22(
    sequence: PoseSequence3D,
    *,
    foot_length_m: float = DEFAULT_FOOT_LENGTH_M,
) -> Tuple[PoseSequence3D, Dict[str, Any], Dict[str, np.ndarray]]:
    """Expand a strict MB17 3D pose sequence to the frozen IMUGPT22 contract."""

    _validate_motionbert17_sequence(sequence)

    source_xyz = np.asarray(sequence.joint_positions_xyz, dtype=np.float32).copy()
    source_conf = np.asarray(sequence.joint_confidence, dtype=np.float32).copy()
    source_observed = np.asarray(sequence.resolved_observed_mask(), dtype=bool).copy()
    source_imputed = np.asarray(sequence.resolved_imputed_mask(), dtype=bool).copy()
    source_xyz, source_conf, handedness_swap_mask = _correct_anatomical_handedness(source_xyz, source_conf)
    source_observed, source_imputed = _swap_masked_handedness(
        source_observed,
        source_imputed,
        handedness_swap_mask,
    )

    num_frames = int(sequence.num_frames)
    mapped_xyz = np.full((num_frames, len(IMUGPT_22_JOINT_NAMES), 3), np.nan, dtype=np.float32)
    mapped_conf = np.zeros((num_frames, len(IMUGPT_22_JOINT_NAMES)), dtype=np.float32)
    mapped_observed = np.zeros((num_frames, len(IMUGPT_22_JOINT_NAMES)), dtype=bool)
    mapped_imputed = np.zeros((num_frames, len(IMUGPT_22_JOINT_NAMES)), dtype=bool)

    for target_name, source_name in _DIRECT_COPY_MAP.items():
        target_index = _IMUGPT22_INDEX[target_name]
        source_index = _MB17_INDEX[source_name]
        mapped_xyz[:, target_index] = source_xyz[:, source_index]
        mapped_conf[:, target_index] = source_conf[:, source_index]
        mapped_observed[:, target_index] = source_observed[:, source_index]
        mapped_imputed[:, target_index] = source_imputed[:, source_index]

    pelvis = source_xyz[:, _MB17_INDEX["pelvis"]]
    thorax = source_xyz[:, _MB17_INDEX["thorax"]]
    pelvis_valid = np.isfinite(pelvis).all(axis=1)
    thorax_valid = np.isfinite(thorax).all(axis=1)
    spine_valid = pelvis_valid & thorax_valid

    spine_line = thorax - pelvis
    mapped_xyz[spine_valid, _IMUGPT22_INDEX["Spine1"]] = (
        pelvis[spine_valid] + (np.float32(1.0 / 3.0) * spine_line[spine_valid])
    )
    mapped_xyz[spine_valid, _IMUGPT22_INDEX["Spine2"]] = (
        pelvis[spine_valid] + (np.float32(2.0 / 3.0) * spine_line[spine_valid])
    )
    mapped_xyz[thorax_valid, _IMUGPT22_INDEX["Spine3"]] = thorax[thorax_valid]

    spine_conf = np.mean(
        np.stack(
            [
                source_conf[:, _MB17_INDEX["pelvis"]],
                source_conf[:, _MB17_INDEX["spine"]],
                source_conf[:, _MB17_INDEX["thorax"]],
            ],
            axis=1,
        ),
        axis=1,
        dtype=np.float32,
    )
    mapped_conf[:, _IMUGPT22_INDEX["Spine1"]] = spine_conf
    mapped_conf[:, _IMUGPT22_INDEX["Spine2"]] = spine_conf
    mapped_conf[:, _IMUGPT22_INDEX["Spine3"]] = spine_conf
    spine_observed = (
        source_observed[:, _MB17_INDEX["pelvis"]]
        & source_observed[:, _MB17_INDEX["spine"]]
        & source_observed[:, _MB17_INDEX["thorax"]]
    )
    spine_imputed = (
        source_imputed[:, _MB17_INDEX["pelvis"]]
        | source_imputed[:, _MB17_INDEX["spine"]]
        | source_imputed[:, _MB17_INDEX["thorax"]]
        | ~spine_observed
    )
    mapped_observed[:, _IMUGPT22_INDEX["Spine1"]] = spine_observed & spine_valid
    mapped_observed[:, _IMUGPT22_INDEX["Spine2"]] = spine_observed & spine_valid
    mapped_observed[:, _IMUGPT22_INDEX["Spine3"]] = spine_observed & thorax_valid
    mapped_imputed[:, _IMUGPT22_INDEX["Spine1"]] = spine_imputed & spine_valid
    mapped_imputed[:, _IMUGPT22_INDEX["Spine2"]] = spine_imputed & spine_valid
    mapped_imputed[:, _IMUGPT22_INDEX["Spine3"]] = spine_imputed & thorax_valid

    spine3 = mapped_xyz[:, _IMUGPT22_INDEX["Spine3"]]
    left_shoulder = mapped_xyz[:, _IMUGPT22_INDEX["Left_shoulder"]]
    right_shoulder = mapped_xyz[:, _IMUGPT22_INDEX["Right_shoulder"]]
    left_collar_valid = np.isfinite(spine3).all(axis=1) & np.isfinite(left_shoulder).all(axis=1)
    right_collar_valid = np.isfinite(spine3).all(axis=1) & np.isfinite(right_shoulder).all(axis=1)
    mapped_xyz[left_collar_valid, _IMUGPT22_INDEX["Left_collar"]] = 0.5 * (
        spine3[left_collar_valid] + left_shoulder[left_collar_valid]
    )
    mapped_xyz[right_collar_valid, _IMUGPT22_INDEX["Right_collar"]] = 0.5 * (
        spine3[right_collar_valid] + right_shoulder[right_collar_valid]
    )
    mapped_conf[:, _IMUGPT22_INDEX["Left_collar"]] = np.minimum(
        mapped_conf[:, _IMUGPT22_INDEX["Spine3"]],
        mapped_conf[:, _IMUGPT22_INDEX["Left_shoulder"]],
    )
    mapped_conf[:, _IMUGPT22_INDEX["Right_collar"]] = np.minimum(
        mapped_conf[:, _IMUGPT22_INDEX["Spine3"]],
        mapped_conf[:, _IMUGPT22_INDEX["Right_shoulder"]],
    )
    mapped_observed[:, _IMUGPT22_INDEX["Left_collar"]] = (
        mapped_observed[:, _IMUGPT22_INDEX["Spine3"]] & mapped_observed[:, _IMUGPT22_INDEX["Left_shoulder"]]
    ) & left_collar_valid
    mapped_observed[:, _IMUGPT22_INDEX["Right_collar"]] = (
        mapped_observed[:, _IMUGPT22_INDEX["Spine3"]] & mapped_observed[:, _IMUGPT22_INDEX["Right_shoulder"]]
    ) & right_collar_valid
    mapped_imputed[:, _IMUGPT22_INDEX["Left_collar"]] = (
        mapped_imputed[:, _IMUGPT22_INDEX["Spine3"]] | mapped_imputed[:, _IMUGPT22_INDEX["Left_shoulder"]]
    ) & left_collar_valid
    mapped_imputed[:, _IMUGPT22_INDEX["Right_collar"]] = (
        mapped_imputed[:, _IMUGPT22_INDEX["Spine3"]] | mapped_imputed[:, _IMUGPT22_INDEX["Right_shoulder"]]
    ) & right_collar_valid

    forward_vectors, forward_fallback_mask = _build_forward_vectors(source_xyz)
    left_ankle = mapped_xyz[:, _IMUGPT22_INDEX["Left_ankle"]]
    right_ankle = mapped_xyz[:, _IMUGPT22_INDEX["Right_ankle"]]
    left_ankle_valid = np.isfinite(left_ankle).all(axis=1)
    right_ankle_valid = np.isfinite(right_ankle).all(axis=1)
    mapped_xyz[left_ankle_valid, _IMUGPT22_INDEX["Left_foot"]] = (
        left_ankle[left_ankle_valid] + (np.float32(foot_length_m) * forward_vectors[left_ankle_valid])
    )
    mapped_xyz[right_ankle_valid, _IMUGPT22_INDEX["Right_foot"]] = (
        right_ankle[right_ankle_valid] + (np.float32(foot_length_m) * forward_vectors[right_ankle_valid])
    )
    mapped_conf[:, _IMUGPT22_INDEX["Left_foot"]] = (
        mapped_conf[:, _IMUGPT22_INDEX["Left_ankle"]] * np.float32(0.8)
    )
    mapped_conf[:, _IMUGPT22_INDEX["Right_foot"]] = (
        mapped_conf[:, _IMUGPT22_INDEX["Right_ankle"]] * np.float32(0.8)
    )
    mapped_imputed[:, _IMUGPT22_INDEX["Left_foot"]] = left_ankle_valid
    mapped_imputed[:, _IMUGPT22_INDEX["Right_foot"]] = right_ankle_valid

    mapped_sequence = PoseSequence3D(
        clip_id=str(sequence.clip_id),
        fps=None if sequence.fps is None else float(sequence.fps),
        fps_original=None if sequence.fps_original is None else float(sequence.fps_original),
        joint_names_3d=list(IMUGPT_22_JOINT_NAMES),
        joint_positions_xyz=mapped_xyz.astype(np.float32, copy=False),
        joint_confidence=mapped_conf.astype(np.float32, copy=False),
        skeleton_parents=list(IMUGPT_22_PARENT_INDICES),
        frame_indices=np.asarray(sequence.frame_indices, dtype=np.int32),
        timestamps_sec=np.asarray(sequence.timestamps_sec, dtype=np.float32),
        source=f"{sequence.source}_imugpt22",
        coordinate_space=str(sequence.coordinate_space),
        observed_mask=mapped_observed.astype(bool, copy=False),
        imputed_mask=mapped_imputed.astype(bool, copy=False),
    )
    quality_report = _build_skeleton_mapper_quality_report(
        sequence=sequence,
        mapped_sequence=mapped_sequence,
        foot_length_m=float(foot_length_m),
        handedness_swap_mask=handedness_swap_mask,
        forward_fallback_mask=forward_fallback_mask,
    )
    artifacts = {
        "handedness_swap_mask": handedness_swap_mask.astype(bool, copy=False),
        "forward_fallback_mask": forward_fallback_mask.astype(bool, copy=False),
    }
    return mapped_sequence, quality_report, artifacts


def _build_skeleton_mapper_quality_report(
    *,
    sequence: PoseSequence3D,
    mapped_sequence: PoseSequence3D,
    foot_length_m: float,
    handedness_swap_mask: np.ndarray,
    forward_fallback_mask: np.ndarray,
) -> Dict[str, Any]:
    notes = []
    handedness_swapped_frames = int(np.count_nonzero(handedness_swap_mask))
    forward_fallback_frames = int(np.count_nonzero(forward_fallback_mask))
    if handedness_swapped_frames > 0:
        notes.append(f"handedness_swapped_frames:{handedness_swapped_frames}")
    if forward_fallback_frames > 0:
        notes.append(f"forward_vector_fallback_frames:{forward_fallback_frames}")

    contract_ok = list(mapped_sequence.joint_names_3d) == list(IMUGPT_22_JOINT_NAMES)
    parents_ok = list(mapped_sequence.skeleton_parents) == list(IMUGPT_22_PARENT_INDICES)
    finite_positions = np.isfinite(np.asarray(mapped_sequence.joint_positions_xyz, dtype=np.float32)).all()
    skeleton_mapping_ok = bool(contract_ok and parents_ok and finite_positions)
    if not finite_positions:
        notes.append("mapped_pose_contains_nan")
    if not contract_ok:
        notes.append("mapped_joint_contract_mismatch")
    if not parents_ok:
        notes.append("mapped_parent_contract_mismatch")

    status = "ok"
    if not skeleton_mapping_ok:
        status = "fail"
    elif handedness_swapped_frames > 0 or forward_fallback_frames > 0:
        status = "warning"

    return {
        "clip_id": str(mapped_sequence.clip_id),
        "status": str(status),
        "fps": None if mapped_sequence.fps is None else float(mapped_sequence.fps),
        "fps_original": None if mapped_sequence.fps_original is None else float(mapped_sequence.fps_original),
        "num_frames": int(mapped_sequence.num_frames),
        "num_joints": int(mapped_sequence.num_joints),
        "input_joint_format": list(sequence.joint_names_3d),
        "output_joint_format": list(mapped_sequence.joint_names_3d),
        "coordinate_space": str(mapped_sequence.coordinate_space),
        "skeleton_mapping_ok": bool(skeleton_mapping_ok),
        "foot_length_m": float(foot_length_m),
        "handedness_swapped_frames": int(handedness_swapped_frames),
        "handedness_swap_ratio": (
            float(handedness_swapped_frames) / float(mapped_sequence.num_frames)
            if mapped_sequence.num_frames > 0
            else 0.0
        ),
        "forward_vector_fallback_frames": int(forward_fallback_frames),
        "forward_vector_fallback_ratio": (
            float(forward_fallback_frames) / float(mapped_sequence.num_frames)
            if mapped_sequence.num_frames > 0
            else 0.0
        ),
        "notes": list(dict.fromkeys(notes)),
    }


def _validate_motionbert17_sequence(sequence: PoseSequence3D) -> None:
    joint_names = [str(name) for name in sequence.joint_names_3d]
    if joint_names != list(MOTIONBERT_17_JOINT_NAMES):
        raise ValueError(
            "Skeleton mapper expects stage-5.5 output ordered as MOTIONBERT_17_JOINT_NAMES."
        )
    points = np.asarray(sequence.joint_positions_xyz, dtype=np.float32)
    confidence = np.asarray(sequence.joint_confidence, dtype=np.float32)
    if points.ndim != 3 or points.shape[1:] != (len(MOTIONBERT_17_JOINT_NAMES), 3):
        raise ValueError("Skeleton mapper expects joint_positions_xyz with shape [T, 17, 3].")
    if confidence.shape != points.shape[:2]:
        raise ValueError("Skeleton mapper expects joint_confidence with shape [T, 17].")


def _swap_masked_handedness(
    observed_mask: np.ndarray,
    imputed_mask: np.ndarray,
    swap_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    corrected_observed = np.asarray(observed_mask, dtype=bool).copy()
    corrected_imputed = np.asarray(imputed_mask, dtype=bool).copy()
    if not np.any(swap_mask):
        return corrected_observed, corrected_imputed

    swap_frame_indices = np.flatnonzero(swap_mask)
    for left_name, right_name in _LEFT_RIGHT_MB17_PAIRS:
        left_index = _MB17_INDEX[left_name]
        right_index = _MB17_INDEX[right_name]

        left_observed = corrected_observed[swap_frame_indices, left_index].copy()
        right_observed = corrected_observed[swap_frame_indices, right_index].copy()
        corrected_observed[swap_frame_indices, left_index] = right_observed
        corrected_observed[swap_frame_indices, right_index] = left_observed

        left_imputed = corrected_imputed[swap_frame_indices, left_index].copy()
        right_imputed = corrected_imputed[swap_frame_indices, right_index].copy()
        corrected_imputed[swap_frame_indices, left_index] = right_imputed
        corrected_imputed[swap_frame_indices, right_index] = left_imputed

    return corrected_observed, corrected_imputed


def _correct_anatomical_handedness(
    positions: np.ndarray,
    confidence: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    corrected_positions = np.asarray(positions, dtype=np.float32).copy()
    corrected_confidence = np.asarray(confidence, dtype=np.float32).copy()

    left_hip_x = corrected_positions[:, _MB17_INDEX["left_hip"], 0]
    right_hip_x = corrected_positions[:, _MB17_INDEX["right_hip"], 0]
    can_validate = np.isfinite(left_hip_x) & np.isfinite(right_hip_x)
    swap_mask = can_validate & ((right_hip_x - left_hip_x) <= 0.0)

    if not np.any(swap_mask):
        return corrected_positions, corrected_confidence, swap_mask

    swap_frame_indices = np.flatnonzero(swap_mask)
    for left_name, right_name in _LEFT_RIGHT_MB17_PAIRS:
        left_index = _MB17_INDEX[left_name]
        right_index = _MB17_INDEX[right_name]

        left_positions = corrected_positions[swap_frame_indices, left_index].copy()
        right_positions = corrected_positions[swap_frame_indices, right_index].copy()
        corrected_positions[swap_frame_indices, left_index] = right_positions
        corrected_positions[swap_frame_indices, right_index] = left_positions

        left_confidence = corrected_confidence[swap_frame_indices, left_index].copy()
        right_confidence = corrected_confidence[swap_frame_indices, right_index].copy()
        corrected_confidence[swap_frame_indices, left_index] = right_confidence
        corrected_confidence[swap_frame_indices, right_index] = left_confidence

    return corrected_positions, corrected_confidence, swap_mask


def _build_forward_vectors(positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(positions, dtype=np.float32)
    forward_vectors = np.zeros((points.shape[0], 3), dtype=np.float32)
    fallback_mask = np.zeros((points.shape[0],), dtype=bool)
    previous_valid_forward = None
    default_forward = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

    for frame_index in range(points.shape[0]):
        left_hip = points[frame_index, _MB17_INDEX["left_hip"]]
        right_hip = points[frame_index, _MB17_INDEX["right_hip"]]
        pelvis = points[frame_index, _MB17_INDEX["pelvis"]]
        neck = points[frame_index, _MB17_INDEX["neck"]]

        if (
            np.isfinite(left_hip).all()
            and np.isfinite(right_hip).all()
            and np.isfinite(pelvis).all()
            and np.isfinite(neck).all()
        ):
            lateral = right_hip - left_hip
            up = neck - pelvis
            lateral_norm = np.linalg.norm(lateral)
            up_norm = np.linalg.norm(up)
            if lateral_norm > _EPSILON and up_norm > _EPSILON:
                v_right = lateral / lateral_norm
                v_up = up / up_norm
                v_forward = np.cross(v_right, v_up)
                forward_norm = np.linalg.norm(v_forward)
                if forward_norm > _EPSILON:
                    forward_vectors[frame_index] = (v_forward / forward_norm).astype(np.float32, copy=False)
                    previous_valid_forward = forward_vectors[frame_index].copy()
                    continue

        fallback_mask[frame_index] = True
        forward_vectors[frame_index] = (
            default_forward if previous_valid_forward is None else previous_valid_forward
        )

    return forward_vectors, fallback_mask
