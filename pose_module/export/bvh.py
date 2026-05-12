"""BVH export helpers for interactive 3D inspection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from pose_module.interfaces import PoseSequence3D

_DEFAULT_FPS = 20.0
_EPSILON = 1e-6


def export_pose_sequence3d_to_bvh(sequence: PoseSequence3D, output_path: str | Path, *, ground_to_floor: bool = True) -> Dict[str, Any]:
    """Export an IMUGPT22 pose sequence to BVH for interactive viewing."""

    joint_names, parents, root_index = _validate_bvh_export_sequence(sequence)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joint_positions_xyz = np.asarray(sequence.joint_positions_xyz, dtype=np.float32).copy()
    if bool(ground_to_floor):
        ground_offset_m = float(np.min(joint_positions_xyz[..., 1]))
        joint_positions_xyz[..., 1] -= np.float32(ground_offset_m)
    else:
        ground_offset_m = 0.0

    children = _build_children_list(parents)
    offsets = _estimate_rest_offsets(joint_positions_xyz, parents)
    traversal_order = _collect_depth_first_order(children, root_index=root_index)
    local_rotations_zyx = _estimate_local_euler_rotations_zyx(
        joint_positions_xyz,
        offsets,
        parents,
        children,
        traversal_order,
        root_index=root_index
    )
    fps = _resolve_sequence_fps(sequence)
    _write_bvh_file(
        output_path=output_path,
        joint_names=joint_names,
        parents=parents,
        children=children,
        offsets=offsets,
        root_positions=joint_positions_xyz[:, root_index, :],
        local_rotations_zyx=local_rotations_zyx,
        traversal_order=traversal_order,
        root_index=root_index,
        frame_time_sec=1.0 / float(fps)
    )

    return {
        "pose3d_bvh_path": str(output_path.resolve()),
        "bvh_fps": float(fps),
        "bvh_frame_time_sec": float(1.0 / float(fps)),
        "ground_to_floor": bool(ground_to_floor),
        "ground_offset_m": float(ground_offset_m),
        "joint_format": list(joint_names),
        "coordinate_space": str(sequence.coordinate_space)
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a PoseSequence3D NPZ to BVH.")
    parser.add_argument(
        "--pose3d-npz",
        type=Path,
        required=True,
        help="Path to the input pose3d.npz file."
    )
    parser.add_argument(
        "--output-bvh",
        type=Path,
        required=True,
        help="Path to the output BVH file."
    )
    parser.add_argument(
        "--no-ground-to-floor",
        action="store_true",
        help="Disable vertical offset normalization that places the lowest point on the floor."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    with np.load(args.pose3d_npz, allow_pickle=False) as payload:
        sequence = PoseSequence3D.from_npz_payload(payload)

    artifacts = export_pose_sequence3d_to_bvh(sequence, args.output_bvh, ground_to_floor=bool(not args.no_ground_to_floor))
    print(json.dumps(artifacts, indent=2, ensure_ascii=True))
    return 0


def _validate_bvh_export_sequence(sequence: PoseSequence3D) -> tuple[list[str], list[int], int]:
    joint_names = [str(name) for name in sequence.joint_names_3d]
    num_joints = len(joint_names)
    if num_joints == 0:
        raise ValueError("BVH export expects at least one joint in joint_names_3d.")

    parents = [int(parent) for parent in sequence.skeleton_parents]
    if len(parents) != num_joints:
        raise ValueError(
            "BVH export expects skeleton_parents and joint_names_3d to have the same length: "
            f"got {len(parents)} and {num_joints}."
        )

    invalid_parent_indices = [parent for parent in parents if parent < -1 or parent >= num_joints]
    if len(invalid_parent_indices) > 0:
        raise ValueError(
            "BVH export expects each skeleton parent index to be -1 or in [0, num_joints)."
        )

    root_indices = [index for index, parent in enumerate(parents) if parent == -1]
    if len(root_indices) != 1:
        raise ValueError(
            f"BVH export expects exactly one skeleton root (parent -1), got {len(root_indices)}."
        )
    root_index = int(root_indices[0])

    children = _build_children_list(parents)
    traversal = _collect_depth_first_order(children, root_index=root_index)
    if len(traversal) != num_joints:
        raise ValueError(
            "BVH export expects a connected tree skeleton rooted at the unique parent=-1 joint."
        )

    joint_positions_xyz = np.asarray(sequence.joint_positions_xyz, dtype=np.float32)
    if joint_positions_xyz.ndim != 3 or joint_positions_xyz.shape[1:] != (num_joints, 3):
        raise ValueError(f"BVH export expects joint_positions_xyz with shape [T, {num_joints}, 3].")
    if joint_positions_xyz.shape[0] == 0:
        raise ValueError("BVH export expects at least one frame.")
    if not np.isfinite(joint_positions_xyz).all():
        raise ValueError("BVH export expects finite joint_positions_xyz without NaN.")

    return joint_names, parents, root_index


def _build_children_list(parents: Sequence[int]) -> List[List[int]]:
    children = [[] for _ in range(len(parents))]
    for joint_index, parent_index in enumerate(parents):
        if parent_index >= 0:
            children[parent_index].append(joint_index)
    return children


def _collect_depth_first_order(children: Sequence[Sequence[int]], *, root_index: int) -> List[int]:
    traversal_order: List[int] = []

    def _visit(joint_index: int) -> None:
        traversal_order.append(joint_index)
        for child_index in children[joint_index]:
            _visit(int(child_index))

    _visit(int(root_index))
    return traversal_order


def _estimate_rest_offsets(joint_positions_xyz: np.ndarray, parents: Sequence[int]) -> np.ndarray:
    offsets = np.zeros((joint_positions_xyz.shape[1], 3), dtype=np.float32)
    for joint_index in range(joint_positions_xyz.shape[1]):
        parent_index = int(parents[joint_index])
        if parent_index < 0:
            continue
        segment_vectors = joint_positions_xyz[:, joint_index, :] - joint_positions_xyz[:, parent_index, :]
        lengths = np.linalg.norm(segment_vectors, axis=1)
        valid_mask = np.isfinite(lengths) & (lengths > _EPSILON)
        if not np.any(valid_mask):
            raise ValueError(f"BVH export cannot estimate rest offset for joint index {joint_index}.")
        valid_vectors = segment_vectors[valid_mask]
        valid_lengths = lengths[valid_mask]
        directions = valid_vectors / valid_lengths[:, None]
        mean_direction = np.mean(directions, axis=0)
        mean_direction_norm = float(np.linalg.norm(mean_direction))
        if mean_direction_norm <= _EPSILON:
            mean_direction = directions[0]
            mean_direction_norm = float(np.linalg.norm(mean_direction))
        offsets[joint_index] = (
            mean_direction / np.float32(mean_direction_norm) * np.float32(np.median(valid_lengths))
        ).astype(np.float32, copy=False)
    return offsets


def _estimate_local_euler_rotations_zyx(
    joint_positions_xyz: np.ndarray,
    offsets: np.ndarray,
    parents: Sequence[int],
    children: Sequence[Sequence[int]],
    traversal_order: Sequence[int],
    *,
    root_index: int,
) -> np.ndarray:
    num_frames = int(joint_positions_xyz.shape[0])
    num_joints = int(joint_positions_xyz.shape[1])
    global_rotations = np.repeat(np.eye(3, dtype=np.float32)[None, None, :, :], num_frames, axis=0)
    global_rotations = np.repeat(global_rotations, num_joints, axis=1)
    local_rotations_zyx = np.zeros((num_frames, num_joints, 3), dtype=np.float32)

    for frame_index in range(num_frames):
        for joint_index in traversal_order:
            child_indices = [int(child_index) for child_index in children[joint_index]]
            if joint_index == root_index:
                parent_global_rotation = np.eye(3, dtype=np.float32)
            else:
                parent_global_rotation = global_rotations[frame_index, int(parents[joint_index])]

            global_rotation = _estimate_joint_global_rotation(
                frame_positions=joint_positions_xyz[frame_index],
                offsets=offsets,
                joint_index=int(joint_index),
                child_indices=child_indices,
                fallback_rotation=parent_global_rotation,
            )
            global_rotations[frame_index, joint_index] = global_rotation
            local_rotation = parent_global_rotation.T @ global_rotation
            local_rotation = _orthonormalize_rotation(local_rotation)
            local_rotations_zyx[frame_index, joint_index] = Rotation.from_matrix(
                local_rotation.astype(np.float64, copy=False)
            ).as_euler("zyx", degrees=True).astype(np.float32, copy=False)

    return local_rotations_zyx


def _estimate_joint_global_rotation(
    *,
    frame_positions: np.ndarray,
    offsets: np.ndarray,
    joint_index: int,
    child_indices: Sequence[int],
    fallback_rotation: np.ndarray,
) -> np.ndarray:
    if len(child_indices) == 0:
        return fallback_rotation.astype(np.float32, copy=False)

    rest_vectors = offsets[np.asarray(child_indices, dtype=np.int32)]
    target_vectors = frame_positions[np.asarray(child_indices, dtype=np.int32)] - frame_positions[joint_index]
    rotation = _best_fit_rotation(rest_vectors, target_vectors)
    if rotation is None:
        return fallback_rotation.astype(np.float32, copy=False)
    return rotation.astype(np.float32, copy=False)


def _best_fit_rotation(rest_vectors: np.ndarray, target_vectors: np.ndarray) -> np.ndarray | None:
    rest_vectors = np.asarray(rest_vectors, dtype=np.float64)
    target_vectors = np.asarray(target_vectors, dtype=np.float64)
    rest_norms = np.linalg.norm(rest_vectors, axis=1)
    target_norms = np.linalg.norm(target_vectors, axis=1)
    valid_mask = (
        np.isfinite(rest_norms)
        & np.isfinite(target_norms)
        & (rest_norms > _EPSILON)
        & (target_norms > _EPSILON)
    )
    if not np.any(valid_mask):
        return None

    normalized_rest = rest_vectors[valid_mask] / rest_norms[valid_mask][:, None]
    normalized_target = target_vectors[valid_mask] / target_norms[valid_mask][:, None]
    if normalized_rest.shape[0] == 1:
        return _align_single_vector(normalized_rest[0], normalized_target[0])

    covariance = normalized_rest.T @ normalized_target
    left_u, _, right_vt = np.linalg.svd(covariance)
    rotation = right_vt.T @ left_u.T
    if np.linalg.det(rotation) < 0.0:
        right_vt[-1, :] *= -1.0
        rotation = right_vt.T @ left_u.T
    return _orthonormalize_rotation(rotation)


def _align_single_vector(source_vector: np.ndarray, target_vector: np.ndarray) -> np.ndarray:
    source_unit = source_vector / np.linalg.norm(source_vector)
    target_unit = target_vector / np.linalg.norm(target_vector)
    cross_product = np.cross(source_unit, target_unit)
    cross_norm = float(np.linalg.norm(cross_product))
    dot_product = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))

    if cross_norm <= _EPSILON:
        if dot_product > 0.0:
            return np.eye(3, dtype=np.float64)
        orthogonal = _find_orthogonal_unit_vector(source_unit)
        return Rotation.from_rotvec(math.pi * orthogonal).as_matrix()

    skew = np.asarray(
        [
            [0.0, -cross_product[2], cross_product[1]],
            [cross_product[2], 0.0, -cross_product[0]],
            [-cross_product[1], cross_product[0], 0.0]
        ],
        dtype=np.float64,
    )
    rotation = np.eye(3, dtype=np.float64) + skew + (skew @ skew) * ((1.0 - dot_product) / (cross_norm**2))
    return _orthonormalize_rotation(rotation)


def _find_orthogonal_unit_vector(vector: np.ndarray) -> np.ndarray:
    axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(axis, vector))) > 0.9:
        axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    orthogonal = np.cross(vector, axis)
    orthogonal_norm = float(np.linalg.norm(orthogonal))
    if orthogonal_norm <= _EPSILON:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return orthogonal / orthogonal_norm


def _orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    left_u, _, right_vt = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    orthonormal = left_u @ right_vt
    if np.linalg.det(orthonormal) < 0.0:
        right_vt[-1, :] *= -1.0
        orthonormal = left_u @ right_vt
    return orthonormal.astype(np.float64, copy=False)


def _write_bvh_file(
    *,
    output_path: Path,
    joint_names: Sequence[str],
    parents: Sequence[int],
    children: Sequence[Sequence[int]],
    offsets: np.ndarray,
    root_positions: np.ndarray,
    local_rotations_zyx: np.ndarray,
    traversal_order: Sequence[int],
    root_index: int,
    frame_time_sec: float
) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("HIERARCHY\n")
        _write_joint_hierarchy(
            handle=handle,
            joint_index=int(root_index),
            joint_names=joint_names,
            parents=parents,
            children=children,
            offsets=offsets,
            indent=""
        )
        handle.write("MOTION\n")
        handle.write(f"Frames: {int(root_positions.shape[0])}\n")
        handle.write(f"Frame Time: {frame_time_sec:.8f}\n")
        for frame_index in range(root_positions.shape[0]):
            frame_values: List[float] = []
            for joint_index in traversal_order:
                if int(joint_index) == int(root_index):
                    frame_values.extend(float(value) for value in root_positions[frame_index])
                frame_values.extend(float(value) for value in local_rotations_zyx[frame_index, joint_index])
            handle.write(" ".join(f"{value:.6f}" for value in frame_values) + "\n")


def _write_joint_hierarchy(
    *,
    handle,
    joint_index: int,
    joint_names: Sequence[str],
    parents: Sequence[int],
    children: Sequence[Sequence[int]],
    offsets: np.ndarray,
    indent: str
) -> None:
    joint_label = "ROOT" if int(parents[joint_index]) == -1 else "JOINT"
    handle.write(f"{indent}{joint_label} {joint_names[joint_index]}\n")
    handle.write(f"{indent}{{\n")
    next_indent = indent + "\t"
    offset_vector = offsets[joint_index]
    handle.write(
        f"{next_indent}OFFSET {float(offset_vector[0]):.6f} {float(offset_vector[1]):.6f} {float(offset_vector[2]):.6f}\n"
    )
    if int(parents[joint_index]) == -1:
        handle.write(f"{next_indent}CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n")
    else:
        handle.write(f"{next_indent}CHANNELS 3 Zrotation Yrotation Xrotation\n")

    child_indices = [int(child_index) for child_index in children[joint_index]]
    if len(child_indices) == 0:
        end_offset = _build_end_site_offset(offsets, joint_index, parents)
        handle.write(f"{next_indent}End Site\n")
        handle.write(f"{next_indent}{{\n")
        handle.write(
            f"{next_indent}\tOFFSET {float(end_offset[0]):.6f} {float(end_offset[1]):.6f} {float(end_offset[2]):.6f}\n"
        )
        handle.write(f"{next_indent}}}\n")
    else:
        for child_index in child_indices:
            _write_joint_hierarchy(
                handle=handle,
                joint_index=child_index,
                joint_names=joint_names,
                parents=parents,
                children=children,
                offsets=offsets,
                indent=next_indent,
            )
    handle.write(f"{indent}}}\n")


def _build_end_site_offset(offsets: np.ndarray, joint_index: int, parents: Sequence[int]) -> np.ndarray:
    if int(parents[joint_index]) >= 0:
        parent_offset = offsets[joint_index]
        parent_norm = float(np.linalg.norm(parent_offset))
        if parent_norm > _EPSILON:
            return (parent_offset / np.float32(parent_norm) * np.float32(parent_norm * 0.35)).astype(np.float32, copy=False)
    return np.asarray([0.0, 0.05, 0.0], dtype=np.float32)


def _resolve_sequence_fps(sequence: PoseSequence3D) -> float:
    if sequence.fps is not None and float(sequence.fps) > 0.0:
        return float(sequence.fps)
    timestamps_sec = np.asarray(sequence.timestamps_sec, dtype=np.float32)
    if timestamps_sec.size >= 2:
        deltas = np.diff(timestamps_sec)
        finite_positive = deltas[np.isfinite(deltas) & (deltas > 0.0)]
        if finite_positive.size > 0:
            return float(1.0 / float(np.median(finite_positive)))
    return float(_DEFAULT_FPS)


if __name__ == "__main__":
    raise SystemExit(main())
