"""Adapters for MotionBERT-compatible 2D/3D tensor contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from pose_module.interfaces import MOTIONBERT_17_JOINT_NAMES, PoseSequence2D, PoseSequence3D


@dataclass(frozen=True)
class MotionBERTWindowBatch:
    inputs: np.ndarray
    frame_index_map: np.ndarray
    valid_mask: np.ndarray
    window_starts: np.ndarray

    @property
    def num_windows(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def window_size(self) -> int:
        return int(self.inputs.shape[1])


def build_motionbert_window_batch(
    sequence: PoseSequence2D,
    *,
    window_size: int,
    window_overlap: float,
    include_confidence: bool = True
) -> MotionBERTWindowBatch:
    _validate_motionbert_sequence(sequence)
    num_frames = int(sequence.num_frames)
    if num_frames <= 0:
        raise ValueError("MotionBERT lifting requires at least one frame.")

    starts = _compute_window_starts(
        num_frames=num_frames,
        window_size=int(window_size),
        window_overlap=float(window_overlap),
    )

    keypoints_xy = np.nan_to_num(np.asarray(sequence.keypoints_xy, dtype=np.float32), nan=0.0)
    confidence = np.asarray(sequence.confidence, dtype=np.float32)
    if include_confidence:
        features = np.concatenate([keypoints_xy, confidence[..., None]], axis=2)
    else:
        features = keypoints_xy

    num_joints = int(features.shape[1])
    num_channels = int(features.shape[2])
    batch_inputs = np.zeros((len(starts), int(window_size), num_joints, num_channels), dtype=np.float32)
    frame_index_map = np.full((len(starts), int(window_size)), -1, dtype=np.int32)
    valid_mask = np.zeros((len(starts), int(window_size)), dtype=bool)

    for window_index, start in enumerate(starts):
        stop = min(int(start) + int(window_size), num_frames)
        valid_length = max(0, stop - int(start))
        if valid_length <= 0:
            continue
        batch_inputs[window_index, :valid_length] = features[int(start):stop]
        frame_index_map[window_index, :valid_length] = np.arange(int(start), stop, dtype=np.int32)
        valid_mask[window_index, :valid_length] = True
        if valid_length < int(window_size):
            batch_inputs[window_index, valid_length:] = features[stop - 1]
            frame_index_map[window_index, valid_length:] = int(stop - 1)

    return MotionBERTWindowBatch(
        inputs=batch_inputs,
        frame_index_map=frame_index_map,
        valid_mask=valid_mask,
        window_starts=np.asarray(starts, dtype=np.int32),
    )


def canonicalize_motionbert_output(
    raw_output: np.ndarray | Mapping[str, Any],
    *,
    expected_batch_size: int,
    expected_window_size: int
) -> np.ndarray:
    joint_names: Optional[Sequence[str]] = None
    if isinstance(raw_output, Mapping):
        if "keypoints_3d" not in raw_output:
            raise ValueError("MotionBERT backend mapping must include 'keypoints_3d'.")
        joint_names = raw_output.get("joint_names")
        raw_output = raw_output["keypoints_3d"]

    predictions = np.asarray(raw_output, dtype=np.float32)
    if predictions.ndim == 3:
        predictions = predictions[None, ...]
    if predictions.ndim != 4 or predictions.shape[-1] < 3:
        raise ValueError(
            "MotionBERT backend output must have shape [B, T, J, 3+] or [T, J, 3+]."
        )
    if int(predictions.shape[0]) != int(expected_batch_size):
        raise ValueError(
            f"Unexpected MotionBERT batch size {predictions.shape[0]} != {expected_batch_size}"
        )
    if int(predictions.shape[1]) != int(expected_window_size):
        raise ValueError(
            f"Unexpected MotionBERT window size {predictions.shape[1]} != {expected_window_size}"
        )

    if joint_names is None:
        if int(predictions.shape[2]) < len(MOTIONBERT_17_JOINT_NAMES):
            raise ValueError("MotionBERT backend output has fewer than 17 joints.")
        return predictions[:, :, : len(MOTIONBERT_17_JOINT_NAMES), :3].astype(np.float32, copy=False)

    joint_index = {str(name): index for index, name in enumerate(joint_names)}
    missing = [name for name in MOTIONBERT_17_JOINT_NAMES if name not in joint_index]
    if missing:
        raise ValueError(f"MotionBERT backend output is missing joints: {missing}")
    ordered_indices = [joint_index[name] for name in MOTIONBERT_17_JOINT_NAMES]
    return predictions[:, :, ordered_indices, :3].astype(np.float32, copy=False)


def merge_motionbert_window_predictions(predictions: np.ndarray, batch: MotionBERTWindowBatch, *, num_frames: int) -> np.ndarray:
    fused = np.zeros((int(num_frames), predictions.shape[2], 3), dtype=np.float32)
    weight_sum = np.zeros((int(num_frames), 1, 1), dtype=np.float32)

    for window_index in range(batch.num_windows):
        valid_positions = np.flatnonzero(batch.valid_mask[window_index])
        if valid_positions.size == 0:
            continue
        weights = _compute_window_weights(int(valid_positions.size))
        for local_offset, position in enumerate(valid_positions.tolist()):
            frame_index = int(batch.frame_index_map[window_index, position])
            if frame_index < 0:
                continue
            weight = float(weights[local_offset])
            fused[frame_index] += weight * predictions[window_index, position]
            weight_sum[frame_index] += weight

    missing_mask = weight_sum[:, 0, 0] <= 0.0
    if np.any(missing_mask):
        raise ValueError("MotionBERT window fusion left uncovered frames in the sequence.")
    fused /= weight_sum
    return fused


def write_pose_sequence3d_npz(sequence: PoseSequence3D, pose3d_npz_path: str | Path) -> None:
    pose3d_npz_path = Path(pose3d_npz_path)
    pose3d_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pose3d_npz_path, **sequence.to_npz_payload())


def _validate_motionbert_sequence(sequence: PoseSequence2D) -> None:
    joint_names = [str(name) for name in sequence.joint_names_2d]
    if joint_names != list(MOTIONBERT_17_JOINT_NAMES):
        raise ValueError("MotionBERT lifting expects output ordered as MOTIONBERT_17_JOINT_NAMES.")
    points = np.asarray(sequence.keypoints_xy, dtype=np.float32)
    confidence = np.asarray(sequence.confidence, dtype=np.float32)
    if points.ndim != 3 or points.shape[1:] != (len(MOTIONBERT_17_JOINT_NAMES), 2):
        raise ValueError("MotionBERT lifting expects keypoints_xy with shape [T, 17, 2].")
    if confidence.shape != points.shape[:2]:
        raise ValueError("MotionBERT lifting expects confidence with shape [T, 17].")


def _compute_window_starts(*, num_frames: int, window_size: int, window_overlap: float) -> list[int]:
    if window_size <= 0:
        raise ValueError("MotionBERT window_size must be positive.")
    if not 0.0 <= float(window_overlap) < 1.0:
        raise ValueError("MotionBERT window_overlap must be in the [0.0, 1.0) range.")
    if num_frames <= window_size:
        return [0]

    step = max(1, int(round(float(window_size) * (1.0 - float(window_overlap)))))
    last_start = max(0, int(num_frames) - int(window_size))
    starts = list(range(0, last_start + 1, step))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _compute_window_weights(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones((max(length, 1),), dtype=np.float32)
    center = 0.5 * float(length - 1)
    offsets = np.abs(np.arange(length, dtype=np.float32) - center)
    normalized = offsets / max(center, 1.0)
    return (1.0 - (0.5 * normalized)).astype(np.float32, copy=False)
