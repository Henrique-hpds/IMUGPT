"""Adapt pseudo-global IMUGPT22 poses to an IK-style contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from pose_module.export.bvh import (
    _build_children_list,
    _collect_depth_first_order,
    _estimate_joint_global_rotation,
    _estimate_rest_offsets,
    _resolve_sequence_fps,
    _write_bvh_file,
)
from pose_module.interfaces import IKSequence, PoseSequence3D
from pose_module.io.cache import write_json_file

DEFAULT_IK_BVH_FILENAME = "pose3d_ik.bvh"
DEFAULT_IK_SEQUENCE_FILENAME = "ik_sequence.npz"
DEFAULT_IK_REPORT_FILENAME = "ik_report.json"
DEFAULT_RECONSTRUCTION_WARNING_M = 0.05
_EPSILON = 1e-6


def run_ik(
    joint_positions_global_m: np.ndarray | PoseSequence3D,
    joint_names: Sequence[str] | None = None,
    parents: Sequence[int] | None = None,
    fps: float | None = None,
    *,
    clip_id: str | None = None,
    fps_original: float | None = None,
    frame_indices: np.ndarray | None = None,
    timestamps_sec: np.ndarray | None = None,
    root_translation_m: np.ndarray | None = None,
    output_dir: str | Path | None = None,
    write_bvh: bool = True,
    bvh_filename: str = DEFAULT_IK_BVH_FILENAME,
    ik_sequence_filename: str = DEFAULT_IK_SEQUENCE_FILENAME,
    report_filename: str = DEFAULT_IK_REPORT_FILENAME,
    reconstruction_warning_m: float = DEFAULT_RECONSTRUCTION_WARNING_M,
) -> Dict[str, Any]:
    """Estimate local joint rotations and root translation from global joint positions."""

    metadata = _normalize_ik_inputs(
        joint_positions_global_m=joint_positions_global_m,
        joint_names=joint_names,
        parents=parents,
        fps=fps,
        clip_id=clip_id,
        fps_original=fps_original,
        frame_indices=frame_indices,
        timestamps_sec=timestamps_sec,
        root_translation_m=root_translation_m
    )

    positions = metadata["joint_positions_global_m"]
    joint_names = metadata["joint_names"]
    parents = metadata["parents"]
    root_translation = metadata["root_translation_m"]
    children = _build_children_list(parents)
    root_index = int(np.where(np.asarray(parents, dtype=np.int32) == -1)[0][0])
    traversal_order = _collect_depth_first_order(children, root_index=root_index)
    offsets = _estimate_rest_offsets(positions, parents)
    local_rotation_matrices, global_rotation_matrices = _estimate_local_rotation_matrices(
        positions,
        offsets=offsets,
        parents=parents,
        children=children,
        traversal_order=traversal_order,
        root_index=root_index
    )
    local_joint_rotations = _rotation_matrices_to_wxyz_quaternions(local_rotation_matrices)
    reconstructed_positions = _forward_kinematics_positions(
        local_rotation_matrices=local_rotation_matrices,
        root_translation_m=root_translation,
        joint_offsets_m=offsets,
        parents=parents
    )
    reconstruction_errors = np.linalg.norm(reconstructed_positions - positions, axis=2)
    reconstruction_error_mean_m = float(np.mean(reconstruction_errors))
    reconstruction_error_max_m = float(np.max(reconstruction_errors))

    ik_sequence = IKSequence(
        clip_id=str(metadata["clip_id"]),
        fps=None if metadata["fps"] is None else float(metadata["fps"]),
        fps_original=(None if metadata["fps_original"] is None else float(metadata["fps_original"])),
        joint_names_3d=list(joint_names),
        local_joint_rotations=local_joint_rotations.astype(np.float32, copy=False),
        root_translation_m=root_translation.astype(np.float32, copy=False),
        joint_offsets_m=offsets.astype(np.float32, copy=False),
        skeleton_parents=list(parents),
        frame_indices=np.asarray(metadata["frame_indices"], dtype=np.int32),
        timestamps_sec=np.asarray(metadata["timestamps_sec"], dtype=np.float32),
        source=str(metadata["source"]),
        rotation_representation="quaternion_wxyz"
    )

    quality_report = _build_ik_quality_report(
        ik_sequence=ik_sequence,
        reconstruction_error_mean_m=reconstruction_error_mean_m,
        reconstruction_error_max_m=reconstruction_error_max_m,
        reconstruction_warning_m=float(reconstruction_warning_m)
    )

    output_dir_path = None if output_dir is None else Path(output_dir)
    artifacts: Dict[str, Any] = {
        "ik_sequence_npz_path": None,
        "ik_report_json_path": None,
        "ik_bvh_path": None
    }
    if output_dir_path is not None:
        output_dir_path.mkdir(parents=True, exist_ok=True)
        ik_sequence_path = output_dir_path / str(ik_sequence_filename)
        np.savez_compressed(ik_sequence_path, **ik_sequence.to_npz_payload())
        artifacts["ik_sequence_npz_path"] = str(ik_sequence_path.resolve())

        report_path = output_dir_path / str(report_filename)
        write_json_file(quality_report, report_path)
        artifacts["ik_report_json_path"] = str(report_path.resolve())

        if bool(write_bvh):
            bvh_path = output_dir_path / str(bvh_filename)
            local_rotations_zyx = Rotation.from_matrix(
                local_rotation_matrices.reshape(-1, 3, 3).astype(np.float64, copy=False)
            ).as_euler("zyx", degrees=True).reshape(local_rotation_matrices.shape[:2] + (3,))
            _write_bvh_file(
                output_path=bvh_path,
                joint_names=joint_names,
                parents=parents,
                children=children,
                offsets=offsets,
                root_positions=root_translation,
                local_rotations_zyx=local_rotations_zyx.astype(np.float32, copy=False),
                traversal_order=traversal_order,
                root_index=root_index,
                frame_time_sec=1.0 / float(metadata["fps_resolved"])
            )
            artifacts["ik_bvh_path"] = str(bvh_path.resolve())

    return {
        "ik_sequence": ik_sequence,
        "local_joint_rotations": np.asarray(ik_sequence.local_joint_rotations, dtype=np.float32),
        "root_translation_m": np.asarray(ik_sequence.root_translation_m, dtype=np.float32),
        "joint_offsets_m": np.asarray(ik_sequence.joint_offsets_m, dtype=np.float32),
        "reconstructed_joint_positions_global_m": reconstructed_positions.astype(np.float32, copy=False),
        "global_rotation_matrices": global_rotation_matrices.astype(np.float32, copy=False),
        "quality_report": quality_report,
        "artifacts": artifacts,
        "bvh_path": artifacts["ik_bvh_path"]
    }


def forward_kinematics_from_ik_sequence(ik_sequence: IKSequence) -> Dict[str, np.ndarray]:
    """Reconstruct global joint positions and orientations from an IKSequence."""

    local_rotation_matrices = _wxyz_quaternions_to_rotation_matrices(np.asarray(ik_sequence.local_joint_rotations, dtype=np.float32))
    global_positions, global_rotations = _forward_kinematics(
        local_rotation_matrices=local_rotation_matrices,
        root_translation_m=np.asarray(ik_sequence.root_translation_m, dtype=np.float32),
        joint_offsets_m=np.asarray(ik_sequence.joint_offsets_m, dtype=np.float32),
        parents=[int(parent) for parent in ik_sequence.skeleton_parents]
    )
    return {
        "joint_positions_global_m": global_positions.astype(np.float32, copy=False),
        "joint_rotation_global_matrices": global_rotations.astype(np.float32, copy=False)
    }


def _normalize_ik_inputs(
    *,
    joint_positions_global_m: np.ndarray | PoseSequence3D,
    joint_names: Sequence[str] | None,
    parents: Sequence[int] | None,
    fps: float | None,
    clip_id: str | None,
    fps_original: float | None,
    frame_indices: np.ndarray | None,
    timestamps_sec: np.ndarray | None,
    root_translation_m: np.ndarray | None,
) -> Dict[str, Any]:
    if isinstance(joint_positions_global_m, PoseSequence3D):
        sequence = joint_positions_global_m
        positions = np.asarray(sequence.joint_positions_xyz, dtype=np.float32)
        joint_names_resolved = [str(name) for name in sequence.joint_names_3d]
        parents_resolved = [int(parent) for parent in sequence.skeleton_parents]
        clip_id_resolved = str(sequence.clip_id if clip_id is None else clip_id)
        fps_resolved = sequence.fps if fps is None else fps
        fps_original_resolved = sequence.fps_original if fps_original is None else fps_original
        frame_indices_resolved = (
            np.asarray(sequence.frame_indices, dtype=np.int32)
            if frame_indices is None
            else np.asarray(frame_indices, dtype=np.int32)
        )
        timestamps_resolved = (
            np.asarray(sequence.timestamps_sec, dtype=np.float32)
            if timestamps_sec is None
            else np.asarray(timestamps_sec, dtype=np.float32)
        )
        root_translation_resolved = sequence.resolved_root_translation_m()
        if root_translation_m is not None:
            root_translation_resolved = np.asarray(root_translation_m, dtype=np.float32)
        source = f"{sequence.source}_ik"
    else:
        positions = np.asarray(joint_positions_global_m, dtype=np.float32)
        if joint_names is None or parents is None:
            raise ValueError("run_ik requires joint_names and parents when raw positions are provided.")
        joint_names_resolved = [str(name) for name in joint_names]
        parents_resolved = [int(parent) for parent in parents]
        clip_id_resolved = str("clip_ik" if clip_id is None else clip_id)
        fps_resolved = fps
        fps_original_resolved = fps_original
        frame_indices_resolved = (
            np.arange(positions.shape[0], dtype=np.int32)
            if frame_indices is None
            else np.asarray(frame_indices, dtype=np.int32)
        )
        if timestamps_sec is None:
            if fps_resolved is None or float(fps_resolved) <= 0.0:
                timestamps_resolved = np.arange(positions.shape[0], dtype=np.float32)
            else:
                timestamps_resolved = (np.arange(positions.shape[0], dtype=np.float32) / np.float32(float(fps_resolved)))
        else:
            timestamps_resolved = np.asarray(timestamps_sec, dtype=np.float32)
        root_translation_resolved = (None if root_translation_m is None else np.asarray(root_translation_m, dtype=np.float32))
        source = "pose3d_ik"

    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("run_ik expects joint_positions_global_m with shape [T, J, 3].")
    if positions.shape[0] == 0 or positions.shape[1] == 0:
        raise ValueError("run_ik expects at least one frame and one joint.")
    if not np.isfinite(positions).all():
        raise ValueError("run_ik expects finite joint_positions_global_m without NaN.")
    if len(joint_names_resolved) != positions.shape[1]:
        raise ValueError("run_ik expects joint_names length to match the joint dimension.")
    if len(parents_resolved) != positions.shape[1]:
        raise ValueError("run_ik expects parents length to match the joint dimension.")

    root_indices = np.flatnonzero(np.asarray(parents_resolved, dtype=np.int32) == -1)
    if root_indices.size != 1:
        raise ValueError("run_ik expects exactly one root joint with parent index -1.")
    root_index = int(root_indices[0])
    if root_translation_resolved is None:
        root_translation_resolved = positions[:, root_index, :].astype(np.float32, copy=False)
    root_translation_resolved = np.asarray(root_translation_resolved, dtype=np.float32)
    if root_translation_resolved.shape != (positions.shape[0], 3):
        raise ValueError("run_ik expects root_translation_m with shape [T, 3].")
    if not np.isfinite(root_translation_resolved).all():
        raise ValueError("run_ik expects finite root_translation_m without NaN.")

    fps_stub_sequence = PoseSequence3D(
        clip_id=str(clip_id_resolved),
        fps=None if fps_resolved is None else float(fps_resolved),
        fps_original=None if fps_original_resolved is None else float(fps_original_resolved),
        joint_names_3d=list(joint_names_resolved),
        joint_positions_xyz=positions,
        joint_confidence=np.ones(positions.shape[:2], dtype=np.float32),
        skeleton_parents=list(parents_resolved),
        frame_indices=np.asarray(frame_indices_resolved, dtype=np.int32),
        timestamps_sec=np.asarray(timestamps_resolved, dtype=np.float32),
        source=str(source),
        coordinate_space="pseudo_global_metric",
        root_translation_m=root_translation_resolved
    )
    fps_final = _resolve_sequence_fps(fps_stub_sequence)
    return {
        "clip_id": str(clip_id_resolved),
        "fps": None if fps_resolved is None else float(fps_resolved),
        "fps_original": None if fps_original_resolved is None else float(fps_original_resolved),
        "fps_resolved": float(fps_final),
        "joint_positions_global_m": positions.astype(np.float32, copy=False),
        "joint_names": list(joint_names_resolved),
        "parents": list(parents_resolved),
        "frame_indices": np.asarray(frame_indices_resolved, dtype=np.int32),
        "timestamps_sec": np.asarray(timestamps_resolved, dtype=np.float32),
        "root_translation_m": root_translation_resolved.astype(np.float32, copy=False),
        "source": str(source)
    }


def _estimate_local_rotation_matrices(
    joint_positions_global_m: np.ndarray,
    *,
    offsets: np.ndarray,
    parents: Sequence[int],
    children: Sequence[Sequence[int]],
    traversal_order: Sequence[int],
    root_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_frames = int(joint_positions_global_m.shape[0])
    num_joints = int(joint_positions_global_m.shape[1])
    global_rotations = np.repeat(np.eye(3, dtype=np.float32)[None, None, :, :], num_frames, axis=0)
    global_rotations = np.repeat(global_rotations, num_joints, axis=1)
    local_rotations = np.repeat(np.eye(3, dtype=np.float32)[None, None, :, :], num_frames, axis=0)
    local_rotations = np.repeat(local_rotations, num_joints, axis=1)

    for frame_index in range(num_frames):
        for joint_index in traversal_order:
            child_indices = [int(child_index) for child_index in children[joint_index]]
            if int(joint_index) == int(root_index):
                parent_global_rotation = np.eye(3, dtype=np.float32)
            else:
                parent_global_rotation = global_rotations[frame_index, int(parents[joint_index])]

            global_rotation = _estimate_joint_global_rotation(
                frame_positions=joint_positions_global_m[frame_index],
                offsets=offsets,
                joint_index=int(joint_index),
                child_indices=child_indices,
                fallback_rotation=parent_global_rotation,
            )
            global_rotations[frame_index, joint_index] = global_rotation.astype(np.float32, copy=False)
            local_rotations[frame_index, joint_index] = (
                parent_global_rotation.T @ global_rotation
            ).astype(np.float32, copy=False)

    return local_rotations.astype(np.float32, copy=False), global_rotations.astype(np.float32, copy=False)


def _rotation_matrices_to_wxyz_quaternions(rotation_matrices: np.ndarray) -> np.ndarray:
    quaternions_xyzw = Rotation.from_matrix(
        np.asarray(rotation_matrices, dtype=np.float64).reshape(-1, 3, 3)
    ).as_quat()
    quaternions_wxyz = np.concatenate(
        [quaternions_xyzw[:, 3:4], quaternions_xyzw[:, :3]],
        axis=1,
    )
    return quaternions_wxyz.reshape(rotation_matrices.shape[:2] + (4,)).astype(np.float32, copy=False)


def _wxyz_quaternions_to_rotation_matrices(quaternions_wxyz: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions_wxyz, dtype=np.float64)
    if quaternions.ndim != 3 or quaternions.shape[-1] != 4:
        raise ValueError("Expected quaternion local_joint_rotations with shape [T, J, 4].")
    quaternions_xyzw = np.concatenate([quaternions[..., 1:], quaternions[..., 0:1]], axis=2)
    rotation_matrices = Rotation.from_quat(quaternions_xyzw.reshape(-1, 4)).as_matrix()
    return rotation_matrices.reshape(quaternions.shape[:2] + (3, 3)).astype(np.float32, copy=False)


def _forward_kinematics_positions(
    *,
    local_rotation_matrices: np.ndarray,
    root_translation_m: np.ndarray,
    joint_offsets_m: np.ndarray,
    parents: Sequence[int]
) -> np.ndarray:
    positions, _ = _forward_kinematics(
        local_rotation_matrices=local_rotation_matrices,
        root_translation_m=root_translation_m,
        joint_offsets_m=joint_offsets_m,
        parents=parents
    )
    return positions


def _forward_kinematics(
    *,
    local_rotation_matrices: np.ndarray,
    root_translation_m: np.ndarray,
    joint_offsets_m: np.ndarray,
    parents: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    local_rotations = np.asarray(local_rotation_matrices, dtype=np.float32)
    root_translation = np.asarray(root_translation_m, dtype=np.float32)
    offsets = np.asarray(joint_offsets_m, dtype=np.float32)
    num_frames, num_joints = local_rotations.shape[:2]
    global_positions = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    global_rotations = np.repeat(np.eye(3, dtype=np.float32)[None, None, :, :], num_frames, axis=0)
    global_rotations = np.repeat(global_rotations, num_joints, axis=1)

    root_indices = np.flatnonzero(np.asarray(parents, dtype=np.int32) == -1)
    if root_indices.size != 1:
        raise ValueError("Forward kinematics expects exactly one root joint.")
    root_index = int(root_indices[0])

    traversal_order = _collect_depth_first_order(_build_children_list(parents), root_index=root_index)
    for frame_index in range(num_frames):
        for joint_index in traversal_order:
            if int(joint_index) == int(root_index):
                global_rotations[frame_index, joint_index] = local_rotations[frame_index, joint_index]
                global_positions[frame_index, joint_index] = root_translation[frame_index]
                continue

            parent_index = int(parents[joint_index])
            global_rotations[frame_index, joint_index] = (
                global_rotations[frame_index, parent_index] @ local_rotations[frame_index, joint_index]
            ).astype(np.float32, copy=False)
            global_positions[frame_index, joint_index] = (
                global_positions[frame_index, parent_index]
                + global_rotations[frame_index, parent_index] @ offsets[joint_index]
            ).astype(np.float32, copy=False)

    return global_positions.astype(np.float32, copy=False), global_rotations.astype(np.float32, copy=False)


def _build_ik_quality_report(
    *,
    ik_sequence: IKSequence,
    reconstruction_error_mean_m: float,
    reconstruction_error_max_m: float,
    reconstruction_warning_m: float
) -> Dict[str, Any]:
    notes = []
    finite_rotations = np.isfinite(np.asarray(ik_sequence.local_joint_rotations, dtype=np.float32)).all()
    finite_root = np.isfinite(np.asarray(ik_sequence.root_translation_m, dtype=np.float32)).all()
    finite_offsets = np.isfinite(np.asarray(ik_sequence.joint_offsets_m, dtype=np.float32)).all()
    ik_ok = bool(finite_rotations and finite_root and finite_offsets)
    if not finite_rotations:
        notes.append("ik_local_rotations_contains_nan")
    if not finite_root:
        notes.append("ik_root_translation_contains_nan")
    if not finite_offsets:
        notes.append("ik_joint_offsets_contains_nan")
    if float(reconstruction_error_mean_m) > float(reconstruction_warning_m):
        notes.append(f"ik_reconstruction_mean_above_threshold:{float(reconstruction_error_mean_m):.4f}")

    status = "ok"
    if not ik_ok:
        status = "fail"
    elif float(reconstruction_error_mean_m) > float(reconstruction_warning_m):
        status = "warning"

    return {
        "clip_id": str(ik_sequence.clip_id),
        "status": str(status),
        "fps": None if ik_sequence.fps is None else float(ik_sequence.fps),
        "fps_original": None if ik_sequence.fps_original is None else float(ik_sequence.fps_original),
        "num_frames": int(ik_sequence.num_frames),
        "num_joints": int(ik_sequence.num_joints),
        "joint_names": list(ik_sequence.joint_names_3d),
        "rotation_representation": str(ik_sequence.rotation_representation),
        "ik_ok": bool(ik_ok),
        "reconstruction_error_mean_m": float(reconstruction_error_mean_m),
        "reconstruction_error_max_m": float(reconstruction_error_max_m),
        "assumptions": [
            "tree_skeleton_with_single_root",
            "positions_are_pseudo_global_metric",
            "local_rotations_estimated_from_best_fit_bone_alignment"
        ],
        "limitations": [
            "rest_offsets_are_estimated_from_motion",
            "leaf_joint_rotations_can_be_ambiguous"
        ],
        "notes": list(dict.fromkeys(notes))
    }
