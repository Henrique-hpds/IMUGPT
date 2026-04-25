"""
Monkey-patch para kimodo.constraints: adiciona suporte a dicts de constraint que
carregam global_joints_positions sem global_joints_rots, e persiste posições globais
no get_save_info do FullBodyConstraintSet.

Chamar apply_patch() uma vez, após kimodo estar importável (após _import_kimodo_package()).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kimodo.skeleton import SkeletonBase


# ── Helper functions ───────────────────────────────────────────────────────────

def _build_children_list_from_parents(parents: list[int]) -> list[list[int]]:
    children = [[] for _ in range(len(parents))]
    for joint_index, parent_index in enumerate(parents):
        if int(parent_index) >= 0:
            children[int(parent_index)].append(int(joint_index))
    return children


def _collect_depth_first_order(children: list[list[int]], root_index: int) -> list[int]:
    traversal_order: list[int] = []

    def _visit(joint_index: int) -> None:
        traversal_order.append(int(joint_index))
        for child_index in children[int(joint_index)]:
            _visit(int(child_index))

    _visit(int(root_index))
    return traversal_order


def _orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    left_u, _, right_vt = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    orthonormal = left_u @ right_vt
    if np.linalg.det(orthonormal) < 0.0:
        right_vt[-1, :] *= -1.0
        orthonormal = left_u @ right_vt
    return orthonormal.astype(np.float64, copy=False)


def _find_orthogonal_unit_vector(vector: np.ndarray) -> np.ndarray:
    axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(axis, vector))) > 0.9:
        axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    orthogonal = np.cross(vector, axis)
    orthogonal_norm = float(np.linalg.norm(orthogonal))
    if orthogonal_norm <= 1e-8:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return orthogonal / orthogonal_norm


def _align_single_vector(source_vector: np.ndarray, target_vector: np.ndarray) -> np.ndarray:
    from kimodo.geometry import axis_angle_to_matrix

    source_unit = source_vector / np.linalg.norm(source_vector)
    target_unit = target_vector / np.linalg.norm(target_vector)
    cross_product = np.cross(source_unit, target_unit)
    cross_norm = float(np.linalg.norm(cross_product))
    dot_product = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))

    if cross_norm <= 1e-8:
        if dot_product > 0.0:
            return np.eye(3, dtype=np.float64)
        orthogonal = _find_orthogonal_unit_vector(source_unit)
        return axis_angle_to_matrix(
            torch.tensor((math.pi * orthogonal)[None], dtype=torch.float64)
        )[0].detach().cpu().numpy().astype(np.float64, copy=False)

    skew = np.asarray(
        [
            [0.0, -cross_product[2], cross_product[1]],
            [cross_product[2], 0.0, -cross_product[0]],
            [-cross_product[1], cross_product[0], 0.0],
        ],
        dtype=np.float64,
    )
    rotation = np.eye(3, dtype=np.float64) + skew + (skew @ skew) * ((1.0 - dot_product) / (cross_norm**2))
    return _orthonormalize_rotation(rotation)


def _best_fit_rotation(rest_vectors: np.ndarray, target_vectors: np.ndarray) -> np.ndarray | None:
    rest_vectors = np.asarray(rest_vectors, dtype=np.float64)
    target_vectors = np.asarray(target_vectors, dtype=np.float64)
    rest_norms = np.linalg.norm(rest_vectors, axis=1)
    target_norms = np.linalg.norm(target_vectors, axis=1)
    valid_mask = (
        np.isfinite(rest_norms)
        & np.isfinite(target_norms)
        & (rest_norms > 1e-8)
        & (target_norms > 1e-8)
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


def _estimate_global_rotations_from_positions(global_joints_positions: Tensor, skeleton: "SkeletonBase") -> Tensor:
    positions = global_joints_positions.detach().cpu().numpy().astype(np.float64, copy=False)
    neutral_joints = skeleton.neutral_joints.detach().cpu().numpy().astype(np.float64, copy=False)
    root_index = int(skeleton.root_idx)
    neutral_joints = neutral_joints - neutral_joints[root_index]
    parents = skeleton.joint_parents.detach().cpu().numpy().astype(np.int32).tolist()
    children = _build_children_list_from_parents(parents)
    traversal_order = _collect_depth_first_order(children, root_index)

    offsets = np.zeros_like(neutral_joints, dtype=np.float64)
    for joint_index, parent_index in enumerate(parents):
        if int(parent_index) < 0:
            continue
        offsets[joint_index] = neutral_joints[joint_index] - neutral_joints[int(parent_index)]

    num_frames, num_joints = positions.shape[:2]
    global_rotations = np.repeat(np.eye(3, dtype=np.float64)[None, None, :, :], num_frames, axis=0)
    global_rotations = np.repeat(global_rotations, num_joints, axis=1)

    for frame_index in range(num_frames):
        for joint_index in traversal_order:
            parent_index = int(parents[joint_index])
            if parent_index < 0:
                parent_global_rotation = np.eye(3, dtype=np.float64)
            else:
                parent_global_rotation = global_rotations[frame_index, parent_index]

            child_indices = children[joint_index]
            if not child_indices:
                global_rotations[frame_index, joint_index] = parent_global_rotation
                continue

            rest_vectors = offsets[np.asarray(child_indices, dtype=np.int32)]
            target_vectors = positions[frame_index, np.asarray(child_indices, dtype=np.int32)] - positions[frame_index, joint_index]
            best_fit = _best_fit_rotation(rest_vectors, target_vectors)
            if best_fit is None:
                best_fit = parent_global_rotation
            global_rotations[frame_index, joint_index] = best_fit

    return torch.tensor(
        global_rotations.astype(np.float32, copy=False),
        device=global_joints_positions.device,
        dtype=global_joints_positions.dtype,
    )


# ── Patched from_dict implementations ─────────────────────────────────────────

def _patched_fullbody_get_save_info(self) -> dict:
    """Return a dict for JSON save.

    The saved payload keeps global positions as the authoritative full-body pose. Local rotations
    are included as auxiliary data for compatibility with existing tooling, but loading should
    prefer `global_joints_positions` when present.
    """
    from kimodo.skeleton import SOMASkeleton30

    local_joints_rot = self.skeleton.global_rots_to_local_rots(self.global_joints_rots)
    if isinstance(self.skeleton, SOMASkeleton30):
        local_joints_rot = self.skeleton.to_SOMASkeleton77(local_joints_rot)
    root_positions = self.global_joints_positions[:, self.skeleton.root_idx, :]
    return {
        "type": self.name,
        "frame_indices": self.frame_indices,
        "local_joints_rot": local_joints_rot,
        "root_positions": root_positions,
        "global_joints_positions": self.global_joints_positions,
        "global_joints_rots": self.global_joints_rots,
        "smooth_root_2d": self.smooth_root_2d,
    }


def _patched_fullbody_from_dict(cls, skeleton: "SkeletonBase", dico: dict):
    """Build a FullBodyConstraintSet from a dict (e.g. loaded from JSON)."""
    from kimodo.geometry import axis_angle_to_matrix
    from kimodo.constraints import _convert_constraint_local_rots_to_skeleton

    frame_indices = torch.tensor(dico["frame_indices"])
    device = skeleton.device if hasattr(skeleton, "device") else "cpu"
    smooth_root_2d = None
    if "smooth_root_2d" in dico:
        smooth_root_2d = torch.tensor(dico["smooth_root_2d"], device=device)

    global_joints_positions = None
    if "global_joints_positions" in dico:
        global_joints_positions = torch.tensor(dico["global_joints_positions"], device=device)
        if global_joints_positions.shape[-2] != skeleton.nbjoints:
            raise ValueError(
                f"Constraint joint count ({global_joints_positions.shape[-2]}) does not match skeleton joint count "
                f"({skeleton.nbjoints}) for global_joints_positions."
            )

    global_joints_rots = None
    if "global_joints_rots" in dico:
        global_joints_rots = torch.tensor(dico["global_joints_rots"], device=device)
        if global_joints_rots.shape[-3] != skeleton.nbjoints:
            raise ValueError(
                f"Constraint joint count ({global_joints_rots.shape[-3]}) does not match skeleton joint count "
                f"({skeleton.nbjoints}) for global_joints_rots."
            )

    if global_joints_positions is None and "local_joints_rot" in dico:
        local_rot = torch.tensor(dico["local_joints_rot"], device=device)
        local_rot_mats = axis_angle_to_matrix(local_rot)
        local_rot_mats = _convert_constraint_local_rots_to_skeleton(local_rot_mats, skeleton)
        global_joints_rots, global_joints_positions, _ = skeleton.fk(
            local_rot_mats,
            torch.tensor(dico["root_positions"], device=device),
        )

    if global_joints_positions is None:
        raise ValueError("FullBodyConstraintSet requires local_joints_rot/root_positions or global_joints_positions.")

    if global_joints_rots is None:
        global_joints_rots = _estimate_global_rotations_from_positions(global_joints_positions, skeleton)

    return cls(
        skeleton,
        frame_indices=frame_indices,
        global_joints_positions=global_joints_positions,
        global_joints_rots=global_joints_rots,
        smooth_root_2d=smooth_root_2d,
    )


def _patched_endeffector_from_dict(cls, skeleton: "SkeletonBase", dico: dict):
    """Build an EndEffectorConstraintSet from a dict (e.g. loaded from JSON).

    Accepts either:
    - ``global_joints_positions`` (K, J, 3): positions used directly, avoiding FK.
      ``global_joints_rots`` is estimated from positions when not provided.
    - ``local_joints_rot`` (K, J, 3) + ``root_positions`` (K, 3): reconstructed via FK.
    """
    from kimodo.geometry import axis_angle_to_matrix
    from kimodo.constraints import _convert_constraint_local_rots_to_skeleton

    frame_indices = torch.tensor(dico["frame_indices"])
    device = skeleton.device if hasattr(skeleton, "device") else "cpu"
    smooth_root_2d = None
    if "smooth_root_2d" in dico:
        smooth_root_2d = torch.tensor(dico["smooth_root_2d"], device=device)

    kwargs = {}
    if not hasattr(cls, "joint_names"):
        kwargs["joint_names"] = dico["joint_names"]

    if "global_joints_positions" in dico:
        global_joints_positions = torch.tensor(dico["global_joints_positions"], device=device)
        if "global_joints_rots" in dico:
            global_joints_rots = torch.tensor(dico["global_joints_rots"], device=device)
        else:
            global_joints_rots = _estimate_global_rotations_from_positions(global_joints_positions, skeleton)
        return cls(
            skeleton,
            frame_indices=frame_indices,
            global_joints_positions=global_joints_positions,
            global_joints_rots=global_joints_rots,
            smooth_root_2d=smooth_root_2d,
            **kwargs,
        )

    local_rot = torch.tensor(dico["local_joints_rot"], device=device)
    local_rot_mats = axis_angle_to_matrix(local_rot)
    local_rot_mats = _convert_constraint_local_rots_to_skeleton(local_rot_mats, skeleton)
    global_joints_rots, global_joints_positions, _ = skeleton.fk(
        local_rot_mats,
        torch.tensor(dico["root_positions"], device=device),
    )
    return cls(
        skeleton,
        frame_indices=frame_indices,
        global_joints_positions=global_joints_positions,
        global_joints_rots=global_joints_rots,
        smooth_root_2d=smooth_root_2d,
        **kwargs,
    )


# ── apply_patch() ──────────────────────────────────────────────────────────────

_patch_applied = False


def apply_patch() -> None:
    """Substitui from_dict e get_save_info nas classes de constraint do kimodo.

    Seguro chamar múltiplas vezes; aplica apenas uma vez.
    Deve ser chamado após kimodo estar importável (após _import_kimodo_package()).
    """
    global _patch_applied
    if _patch_applied:
        return
    from kimodo.constraints import FullBodyConstraintSet, EndEffectorConstraintSet
    FullBodyConstraintSet.get_save_info = _patched_fullbody_get_save_info
    FullBodyConstraintSet.from_dict = classmethod(_patched_fullbody_from_dict)
    EndEffectorConstraintSet.from_dict = classmethod(_patched_endeffector_from_dict)
    _patch_applied = True
