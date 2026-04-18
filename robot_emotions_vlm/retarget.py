"""Retarget IMUGPT22 joint positions to the SMPLX22 bone-length space.

The IMUGPT22 and SMPLX22 skeletons share identical topology (same 22 joints,
same parent indices, same joint order).  The only difference is bone lengths:
MotionBERT produces positions scaled to the subject's actual body, while the
Kimodo SMPLX22 neutral skeleton has fixed canonical bone lengths.

Passing raw IMUGPT positions to Kimodo's _estimate_global_rotations_from_positions
causes ~180-degree errors on the pelvis because the pelvis→hip bone points in
opposite Y directions between the two rest poses.

The fix is a single pass that rescales each bone in the observed skeleton to
match the corresponding bone length in the SMPLX22 neutral pose, preserving
direction.  After rescaling every bone length matches the SMPLX22 rest pose,
so the SVD-based retarget produces correct rotations.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

# SMPLX22 neutral bone lengths are loaded once and cached.
_SMPLX22_ASSETS_PATH = Path(__file__).resolve().parents[1] / "kimodo" / "kimodo" / "assets" / "skeletons" / "smplx22" / "joints.p"

# Canonical parent indices for SMPLX22 / IMUGPT22 (identical topology).
_PARENTS: tuple[int, ...] = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


@lru_cache(maxsize=1)
def _load_smplx22_neutral() -> np.ndarray:
    """Load and cache the SMPLX22 neutral joint positions, shape (22, 3), pelvis-centered."""
    if not _SMPLX22_ASSETS_PATH.exists():
        raise FileNotFoundError(
            f"SMPLX22 neutral joints not found at {_SMPLX22_ASSETS_PATH}. "
            "Ensure the kimodo submodule is initialised."
        )
    import torch
    neutral = torch.load(str(_SMPLX22_ASSETS_PATH), map_location="cpu", weights_only=False)
    neutral = np.asarray(neutral, dtype=np.float64)
    neutral = neutral - neutral[0]  # pelvis-center
    return neutral.astype(np.float32, copy=False)


def _compute_bone_lengths(neutral: np.ndarray, parents: tuple[int, ...]) -> np.ndarray:
    """Return canonical SMPLX22 bone lengths, shape (22,).  Root bone length is 0."""
    lengths = np.zeros(len(parents), dtype=np.float32)
    for i, p in enumerate(parents):
        if p >= 0:
            lengths[i] = float(np.linalg.norm(neutral[i] - neutral[p]))
    return lengths


@lru_cache(maxsize=1)
def _smplx22_bone_lengths() -> np.ndarray:
    return _compute_bone_lengths(_load_smplx22_neutral(), _PARENTS)


def rescale_positions_to_smplx22(
    positions: np.ndarray,
    parents: tuple[int, ...] = _PARENTS,
) -> np.ndarray:
    """Rescale IMUGPT22 joint positions to SMPLX22 bone lengths.

    Each bone direction is preserved; only its length is changed to match the
    SMPLX22 neutral skeleton.  The pelvis position is preserved as-is (it is
    the root and has no parent bone).

    Args:
        positions: float32 array of shape (T, 22, 3) in any coordinate system.
        parents: parent index tuple matching the joint order of ``positions``.

    Returns:
        float32 array of shape (T, 22, 3) with SMPLX22-compatible bone lengths.
    """
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim == 2:
        positions = positions[None]
        squeeze = True
    else:
        squeeze = False

    if positions.shape[1] != 22 or positions.shape[2] != 3:
        raise ValueError(
            f"Expected positions of shape (T, 22, 3); got {positions.shape}."
        )

    target_lengths = _smplx22_bone_lengths()
    out = positions.copy()

    # BFS/topological order — parents are always before children in _PARENTS.
    for i, p in enumerate(parents):
        if p < 0:
            continue
        # Use original positions to get observed bone direction,
        # then place child relative to its already-rescaled parent.
        bone_vec = positions[:, i, :] - positions[:, p, :]  # (T, 3)
        bone_norms = np.linalg.norm(bone_vec, axis=1, keepdims=True)  # (T, 1)

        target_len = float(target_lengths[i])
        if target_len < 1e-6:
            # Zero-length canonical bone — keep original position.
            continue

        # Where the observed bone is degenerate (zero length), fall back to
        # the canonical rest-pose offset so the child is at a valid position.
        degenerate = (bone_norms[:, 0] < 1e-6)
        neutral = _load_smplx22_neutral()
        rest_dir = (neutral[i] - neutral[p]).astype(np.float32)
        rest_dir_norm = float(np.linalg.norm(rest_dir))
        if rest_dir_norm > 1e-6:
            rest_dir = rest_dir / rest_dir_norm
        else:
            rest_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        bone_dirs = np.where(
            degenerate[:, None],
            rest_dir[None, :],
            bone_vec / np.maximum(bone_norms, 1e-6),
        )  # (T, 3)

        out[:, i, :] = out[:, p, :] + bone_dirs * target_len

    if squeeze:
        out = out[0]
    return out


def retarget_positions_to_smplx22_space(
    positions: np.ndarray,
    *,
    pelvis_index: int = 0,
) -> np.ndarray:
    """Full retarget pipeline: pelvis-center → rescale → restore pelvis.

    The Kimodo FK expects positions where the pelvis is the root and the rest
    of the skeleton is expressed relative to it.  This function:

    1. Subtracts pelvis so all positions are pelvis-relative.
    2. Rescales each bone to SMPLX22 canonical length.
    3. Re-adds the original pelvis translation so global positions are preserved.

    Args:
        positions: float32 (T, 22, 3) in IMUGPT pseudo-global space.
        pelvis_index: index of the pelvis joint (0 for both IMUGPT22 and SMPLX22).

    Returns:
        float32 (T, 22, 3) with SMPLX22-compatible bone lengths, same pelvis trajectory.
    """
    positions = np.asarray(positions, dtype=np.float32)
    pelvis = positions[:, pelvis_index : pelvis_index + 1, :].copy()  # (T,1,3)
    centered = positions - pelvis
    rescaled = rescale_positions_to_smplx22(centered)
    return (rescaled + pelvis).astype(np.float32, copy=False)


def compute_local_axis_angle_from_positions_robust(
    positions_smplx_space: np.ndarray,
) -> np.ndarray:
    """Convert SMPLX22-space global positions to local axis-angle rotations.

    Uses Kimodo's own _estimate_global_rotations_from_positions + FK inversion.
    Assumes positions have already been rescaled to SMPLX22 bone lengths via
    retarget_positions_to_smplx22_space.

    Args:
        positions_smplx_space: float32 (K, 22, 3), pelvis at its global position.

    Returns:
        float32 (K, 22, 3) local axis-angle rotations.
    """
    import sys
    import torch

    _ensure_kimodo_on_syspath()
    from kimodo.constraints import _estimate_global_rotations_from_positions
    from kimodo.geometry import matrix_to_axis_angle
    from kimodo.skeleton import build_skeleton, global_rots_to_local_rots

    skeleton = build_skeleton(22)

    # Pelvis-center before passing to retarget (Kimodo FK expects this).
    pelvis = positions_smplx_space[:, 0:1, :].copy()
    centered = positions_smplx_space - pelvis

    positions_tensor = torch.from_numpy(
        np.ascontiguousarray(centered, dtype=np.float32)
    )
    global_rot_mats = _estimate_global_rotations_from_positions(positions_tensor, skeleton)

    dets = torch.linalg.det(global_rot_mats)
    if bool((dets <= 0.0).any()):
        raise ValueError("Retarget produced non-rotation matrices (det <= 0).")

    local_rot_mats = global_rots_to_local_rots(global_rot_mats, skeleton)
    local_aa = matrix_to_axis_angle(local_rot_mats)
    result = local_aa.detach().cpu().numpy().astype(np.float32, copy=False)

    if not np.isfinite(result).all():
        raise ValueError("Retarget produced NaN/Inf axis-angle values.")
    return result


def _ensure_kimodo_on_syspath() -> None:
    import sys
    kimodo_repo = Path(__file__).resolve().parents[1] / "kimodo"
    kimodo_str = str(kimodo_repo)
    if kimodo_repo.exists() and kimodo_str not in sys.path:
        sys.path.insert(0, kimodo_str)
