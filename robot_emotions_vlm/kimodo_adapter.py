"""Adapter that converts Kimodo motion.npz output into PoseSequence3D."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from pose_module.interfaces import (
    IMUGPT_22_JOINT_NAMES,
    IMUGPT_22_PARENT_INDICES,
    PoseSequence3D,
)

KIMODO_COORDINATE_SPACE = "kimodo_world_y_up"


def load_kimodo_pose_sequence3d(
    *,
    kimodo_npz_path: str | Path,
    clip_id: str,
    generation_config_path: str | Path | None = None,
    fps: Optional[float] = None,
) -> PoseSequence3D:
    """Build a PoseSequence3D from a Kimodo motion.npz file.

    The Kimodo SMPL-X 22-joint skeleton is structurally identical to IMUGPT22
    (same joint order and hierarchy), so no remapping is needed.
    """
    data = np.load(str(kimodo_npz_path))

    posed_joints = np.asarray(data["posed_joints"], dtype=np.float32)
    if posed_joints.ndim != 3 or posed_joints.shape[1:] != (len(IMUGPT_22_JOINT_NAMES), 3):
        raise ValueError(
            f"Kimodo posed_joints must have shape (T, 22, 3), got {posed_joints.shape}"
        )
    if not np.isfinite(posed_joints).all():
        raise ValueError("Kimodo posed_joints contains NaN or infinite values.")

    resolved_fps = _resolve_fps(fps=fps, generation_config_path=generation_config_path, data=data)

    num_frames = int(posed_joints.shape[0])
    frame_indices = np.arange(num_frames, dtype=np.int32)
    timestamps_sec = (frame_indices.astype(np.float32)) / np.float32(resolved_fps)

    return PoseSequence3D(
        clip_id=str(clip_id),
        fps=resolved_fps,
        fps_original=resolved_fps,
        joint_names_3d=list(IMUGPT_22_JOINT_NAMES),
        joint_positions_xyz=posed_joints,
        joint_confidence=np.ones((num_frames, len(IMUGPT_22_JOINT_NAMES)), dtype=np.float32),
        skeleton_parents=list(IMUGPT_22_PARENT_INDICES),
        frame_indices=frame_indices,
        timestamps_sec=timestamps_sec,
        source="kimodo_smplx",
        coordinate_space=KIMODO_COORDINATE_SPACE,
        observed_mask=np.ones((num_frames, len(IMUGPT_22_JOINT_NAMES)), dtype=bool),
        imputed_mask=np.zeros((num_frames, len(IMUGPT_22_JOINT_NAMES)), dtype=bool),
    )


def build_kimodo_adapter_quality_report(pose_sequence: PoseSequence3D) -> Dict[str, Any]:
    positions = np.asarray(pose_sequence.joint_positions_xyz, dtype=np.float32)
    finite = bool(np.isfinite(positions).all())
    parents_ok = list(pose_sequence.skeleton_parents) == list(IMUGPT_22_PARENT_INDICES)
    contract_ok = list(pose_sequence.joint_names_3d) == list(IMUGPT_22_JOINT_NAMES)
    status = "ok" if finite and parents_ok and contract_ok else "fail"
    notes: list[str] = []
    if not finite:
        notes.append("kimodo_adapter_output_contains_nan")
    if not parents_ok:
        notes.append("kimodo_adapter_parent_contract_mismatch")
    if not contract_ok:
        notes.append("kimodo_adapter_joint_contract_mismatch")
    return {
        "clip_id": str(pose_sequence.clip_id),
        "status": str(status),
        "generation_backend": "kimodo",
        "num_frames": int(pose_sequence.num_frames),
        "num_joints": int(pose_sequence.num_joints),
        "coordinate_space": str(pose_sequence.coordinate_space),
        "notes": list(notes),
    }


def _resolve_fps(
    *,
    fps: Optional[float],
    generation_config_path: str | Path | None,
    data: Any,
) -> float:
    if fps is not None:
        return float(fps)
    if generation_config_path is not None:
        config_path = Path(generation_config_path)
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if "fps" in config:
                return float(config["fps"])
    # Kimodo saves fps inside the npz when available
    if "fps" in data:
        return float(data["fps"])
    return 30.0
