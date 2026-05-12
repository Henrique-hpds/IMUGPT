"""Application utilities for learned IMU alignment transforms."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .fit import AlignmentFittingError
from .interfaces import AlignmentConfig, AlignmentResult, IMUSequence, SensorSubjectTransform
from .metrics import summarize_alignment_metrics
from .rotation import apply_rotation
from .temporal import align_streams_with_lag, estimate_time_lag, prepare_sequences_for_alignment


def apply_sensor_subject_transform(
    real_seq: IMUSequence,
    virt_seq: IMUSequence,
    transforms: Dict[Tuple[str, str], SensorSubjectTransform],
    config: AlignmentConfig,
) -> List[AlignmentResult]:
    """Apply learned per-sensor transforms to a capture and evaluate the result."""

    config.validate()
    if str(real_seq.subject_id) != str(virt_seq.subject_id) or str(real_seq.capture_id) != str(virt_seq.capture_id):
        raise AlignmentFittingError(
            "real_seq and virt_seq must refer to the same (subject_id, capture_id) pair."
        )
    if set(real_seq.sensor_names) != set(virt_seq.sensor_names):
        raise AlignmentFittingError("real_seq and virt_seq must expose the same sensor names.")

    aligned_real_seq, aligned_virt_seq, timebase_summary = prepare_sequences_for_alignment(real_seq, virt_seq)
    results: list[AlignmentResult] = []
    for sensor_name in aligned_real_seq.sensor_names:
        transform_key = (str(aligned_real_seq.subject_id), str(sensor_name))
        if transform_key not in transforms:
            raise AlignmentFittingError(
                f"Missing transform for subject '{aligned_real_seq.subject_id}' sensor '{sensor_name}'."
            )
        transform = transforms[transform_key]
        real_index = _sensor_index(aligned_real_seq.sensor_names, sensor_name)
        virt_index = _sensor_index(aligned_virt_seq.sensor_names, sensor_name)
        assert real_index is not None and virt_index is not None

        real_acc = np.asarray(aligned_real_seq.acc[:, real_index, :], dtype=np.float32)
        real_gyro = np.asarray(aligned_real_seq.gyro[:, real_index, :], dtype=np.float32)
        virt_acc = np.asarray(aligned_virt_seq.acc[:, virt_index, :], dtype=np.float32)
        virt_gyro = np.asarray(aligned_virt_seq.gyro[:, virt_index, :], dtype=np.float32)
        lag_samples = _estimate_capture_lag(
            real_acc=real_acc,
            virt_acc=virt_acc,
            real_gyro=real_gyro,
            virt_gyro=virt_gyro,
            config=config
        )

        aligned_real_acc, aligned_virt_acc = align_streams_with_lag(real_acc, virt_acc, lag_samples)
        aligned_real_gyro, aligned_virt_gyro = align_streams_with_lag(real_gyro, virt_gyro, lag_samples)
        metrics_before = summarize_alignment_metrics(
            real_acc=aligned_real_acc,
            estimate_acc=aligned_virt_acc,
            real_gyro=aligned_real_gyro,
            estimate_gyro=aligned_virt_gyro
        )
        metrics_before["timebase_summary"] = dict(timebase_summary)

        aligned_acc = _apply_transform_to_values(
            aligned_virt_acc,
            rotation=transform.rotation,
            scale=transform.acc_scale,
            bias=transform.acc_bias
        )
        aligned_gyro = _apply_transform_to_values(
            aligned_virt_gyro,
            rotation=transform.rotation,
            scale=transform.gyro_scale,
            bias=transform.gyro_bias
        )
        metrics_after = summarize_alignment_metrics(
            real_acc=aligned_real_acc,
            estimate_acc=aligned_acc,
            real_gyro=aligned_real_gyro,
            estimate_gyro=aligned_gyro
        )
        metrics_after["timebase_summary"] = dict(timebase_summary)
        results.append(
            AlignmentResult(
                subject_id=str(aligned_real_seq.subject_id),
                capture_id=str(aligned_real_seq.capture_id),
                sensor_name=str(sensor_name),
                lag_samples=int(lag_samples),
                acc_aligned=aligned_acc,
                gyro_aligned=aligned_gyro,
                metrics_before=metrics_before,
                metrics_after=metrics_after
            )
        )
    return results


def apply_transforms_to_imu_sequence(virt_seq: IMUSequence, transforms: Dict[Tuple[str, str], SensorSubjectTransform]) -> IMUSequence:
    """Apply learned spatial transforms to the full virtual sequence.

    Temporal lag is intentionally not baked into the returned sequence because lag
    is a capture-level evaluation adjustment rather than a trained geometric
    parameter shared across captures.
    """

    transformed_acc = np.asarray(virt_seq.acc, dtype=np.float32).copy()
    transformed_gyro = np.asarray(virt_seq.gyro, dtype=np.float32).copy()
    for sensor_index, sensor_name in enumerate(virt_seq.sensor_names):
        transform_key = (str(virt_seq.subject_id), str(sensor_name))
        if transform_key not in transforms:
            raise AlignmentFittingError(f"Missing transform for subject '{virt_seq.subject_id}' sensor '{sensor_name}'.")
        transform = transforms[transform_key]
        transformed_acc[:, sensor_index, :] = _apply_transform_to_values(
            transformed_acc[:, sensor_index, :],
            rotation=transform.rotation,
            scale=transform.acc_scale,
            bias=transform.acc_bias
        )
        transformed_gyro[:, sensor_index, :] = _apply_transform_to_values(
            transformed_gyro[:, sensor_index, :],
            rotation=transform.rotation,
            scale=transform.gyro_scale,
            bias=transform.gyro_bias
        )
    return IMUSequence(
        subject_id=str(virt_seq.subject_id),
        capture_id=str(virt_seq.capture_id),
        sensor_names=list(virt_seq.sensor_names),
        fps=virt_seq.fps,
        timestamps=np.asarray(virt_seq.timestamps, dtype=np.float32),
        acc=transformed_acc,
        gyro=transformed_gyro
    )


def _estimate_capture_lag(
    *,
    real_acc: np.ndarray,
    virt_acc: np.ndarray,
    real_gyro: np.ndarray,
    virt_gyro: np.ndarray,
    config: AlignmentConfig,
) -> int:
    if not bool(config.estimate_time_lag):
        return 0
    if str(config.lag_signal) == "acc_norm":
        return estimate_time_lag(real_acc, virt_acc, int(config.max_lag_samples), str(config.lag_signal))
    return estimate_time_lag(real_gyro, virt_gyro, int(config.max_lag_samples), str(config.lag_signal))


def _apply_transform_to_values(values_xyz: np.ndarray, *, rotation: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    rotated = apply_rotation(values_xyz, rotation)
    return (rotated * np.asarray(scale, dtype=np.float32)[None, :] + np.asarray(bias, dtype=np.float32)[None, :]).astype(np.float32, copy=False)


def _sensor_index(sensor_names: Sequence[str], sensor_name: str) -> int | None:
    for index, candidate in enumerate(sensor_names):
        if str(candidate) == str(sensor_name):
            return int(index)
    return None
