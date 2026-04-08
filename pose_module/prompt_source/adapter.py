"""Adapters that turn prompt-motion backend outputs into PoseSequence3D."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from pose_module.interfaces import (
    IMUGPT_22_JOINT_NAMES,
    IMUGPT_22_PARENT_INDICES,
    PoseSequence3D,
)


PROMPT_RAW_COORDINATE_SPACE = "t2mgpt_pseudo_global_model"


def canonicalize_prompt_joint_positions_xyz(joint_positions_xyz: Any) -> np.ndarray:
    positions = np.asarray(joint_positions_xyz, dtype=np.float32)
    if positions.ndim == 4:
        if positions.shape[0] != 1:
            raise ValueError("Prompt backend output must have shape [T, 22, 3] or [1, T, 22, 3].")
        positions = positions[0]
    if positions.ndim != 3 or positions.shape[1:] != (len(IMUGPT_22_JOINT_NAMES), 3):
        raise ValueError(
            "Prompt backend output must have shape [T, 22, 3] ordered as IMUGPT_22_JOINT_NAMES."
        )
    if positions.shape[0] <= 0:
        raise ValueError("Prompt backend output must contain at least one frame.")
    if not np.isfinite(positions).all():
        raise ValueError("Prompt backend output contains NaN or infinite values.")
    return positions.astype(np.float32, copy=False)


def build_prompt_pose_sequence3d(
    *,
    prompt_id: str,
    joint_positions_xyz: Any,
    fps: float,
    source: str = "t2mgpt_prompt",
    coordinate_space: str = PROMPT_RAW_COORDINATE_SPACE,
    clip_id: Optional[str] = None,
) -> PoseSequence3D:
    positions = canonicalize_prompt_joint_positions_xyz(joint_positions_xyz)
    num_frames = int(positions.shape[0])
    fps = float(fps)
    if fps <= 0.0:
        raise ValueError("Prompt pose adapter requires fps > 0.")
    frame_indices = np.arange(num_frames, dtype=np.int32)
    timestamps_sec = frame_indices.astype(np.float32) / np.float32(fps)
    return PoseSequence3D(
        clip_id=str(prompt_id if clip_id is None else clip_id),
        fps=fps,
        fps_original=fps,
        joint_names_3d=list(IMUGPT_22_JOINT_NAMES),
        joint_positions_xyz=positions,
        joint_confidence=np.ones((num_frames, len(IMUGPT_22_JOINT_NAMES)), dtype=np.float32),
        skeleton_parents=list(IMUGPT_22_PARENT_INDICES),
        frame_indices=frame_indices,
        timestamps_sec=timestamps_sec,
        source=str(source),
        coordinate_space=str(coordinate_space),
        observed_mask=np.ones((num_frames, len(IMUGPT_22_JOINT_NAMES)), dtype=bool),
        imputed_mask=np.zeros((num_frames, len(IMUGPT_22_JOINT_NAMES)), dtype=bool),
    )


def build_prompt_metadata(
    *,
    sample_id: str,
    prompt_id: str,
    prompt_text: str,
    labels: Mapping[str, Any],
    seed: int,
    fps: float,
    num_generated_frames: int,
    generation_backend: str,
    action_detail: str | None = None,
    stimulus_type: str | None = None,
    reference_clip_id: str | None = None,
    duration_hint_sec: float | None = None,
    notes: str | None = None,
    group_id: str | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "sample_id": str(sample_id),
        "prompt_id": str(prompt_id),
        "prompt_text": str(prompt_text),
        "labels": dict(labels),
        "seed": int(seed),
        "fps": float(fps),
        "num_generated_frames": int(num_generated_frames),
        "generation_backend": str(generation_backend),
        "source_kind": "prompt",
        "modality_domain": "synthetic",
    }
    if action_detail is not None:
        payload["action_detail"] = str(action_detail)
    if stimulus_type is not None:
        payload["stimulus_type"] = str(stimulus_type)
    if reference_clip_id is not None:
        payload["reference_clip_id"] = str(reference_clip_id)
    if duration_hint_sec is not None:
        payload["duration_hint_sec"] = float(duration_hint_sec)
    if notes is not None:
        payload["notes"] = str(notes)
    if group_id is not None:
        payload["group_id"] = str(group_id)
    if source_metadata:
        payload["source_metadata"] = dict(source_metadata)
    return payload


def build_prompt_adapter_quality_report(
    *,
    pose_sequence: PoseSequence3D,
    generation_backend: str,
) -> Dict[str, Any]:
    positions = np.asarray(pose_sequence.joint_positions_xyz, dtype=np.float32)
    finite = bool(np.isfinite(positions).all())
    parents_ok = [int(parent) for parent in pose_sequence.skeleton_parents] == list(IMUGPT_22_PARENT_INDICES)
    contract_ok = list(pose_sequence.joint_names_3d) == list(IMUGPT_22_JOINT_NAMES)
    confidence = np.asarray(pose_sequence.joint_confidence, dtype=np.float32)
    confidence_ok = confidence.shape == positions.shape[:2] and np.allclose(confidence, 1.0)
    status = "ok" if finite and parents_ok and contract_ok and confidence_ok else "fail"
    notes: list[str] = []
    if not finite:
        notes.append("prompt_adapter_output_contains_nan")
    if not parents_ok:
        notes.append("prompt_adapter_parent_contract_mismatch")
    if not contract_ok:
        notes.append("prompt_adapter_joint_contract_mismatch")
    if not confidence_ok:
        notes.append("prompt_adapter_confidence_contract_mismatch")
    return {
        "clip_id": str(pose_sequence.clip_id),
        "status": str(status),
        "generation_backend": str(generation_backend),
        "num_frames": int(pose_sequence.num_frames),
        "num_joints": int(pose_sequence.num_joints),
        "coordinate_space": str(pose_sequence.coordinate_space),
        "joint_format": list(pose_sequence.joint_names_3d),
        "notes": list(notes),
    }
