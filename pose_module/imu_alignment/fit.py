"""Training routines for per-subject, per-sensor IMU geometric alignment."""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .interfaces import AlignmentConfig, IMUSequence, SensorSubjectTransform
from .metrics import summarize_alignment_metrics
from .rotation import apply_rotation, estimate_rotation_procrustes
from .temporal import align_streams_with_lag, estimate_time_lag, prepare_sequences_for_alignment

_MIN_REQUIRED_SAMPLES = 3
_LOW_SAMPLE_WARNING = 25
_EPSILON = 1e-8


class AlignmentFittingError(ValueError):
    """Raised when a transform cannot be fit cleanly from the provided data."""


def fit_sensor_subject_transforms(
    real_sequences: List[IMUSequence],
    virtual_sequences: List[IMUSequence],
    config: AlignmentConfig,
) -> Dict[Tuple[str, str], SensorSubjectTransform]:
    """Fit one rigid transform per ``(subject_id, sensor_name)`` pair."""

    config.validate()
    paired_sequences = _pair_sequences(real_sequences, virtual_sequences)

    grouped_pairs: dict[str, list[tuple[IMUSequence, IMUSequence]]] = defaultdict(list)
    for real_sequence, virtual_sequence in paired_sequences:
        grouped_pairs[str(real_sequence.subject_id)].append((real_sequence, virtual_sequence))

    transforms: Dict[Tuple[str, str], SensorSubjectTransform] = {}
    for subject_id, subject_pairs in grouped_pairs.items():
        sensor_names = _collect_common_sensor_names(subject_pairs)
        for sensor_name in sensor_names:
            transform = _fit_single_sensor_subject_transform(
                subject_id=str(subject_id),
                sensor_name=str(sensor_name),
                paired_sequences=subject_pairs,
                config=config
            )
            transforms[(str(subject_id), str(sensor_name))] = transform
    return transforms


def _fit_single_sensor_subject_transform(
    *,
    subject_id: str,
    sensor_name: str,
    paired_sequences: Sequence[tuple[IMUSequence, IMUSequence]],
    config: AlignmentConfig
) -> SensorSubjectTransform:
    rotation_virtual_blocks: list[np.ndarray] = []
    rotation_real_blocks: list[np.ndarray] = []
    residual_acc_virtual_blocks: list[np.ndarray] = []
    residual_acc_real_blocks: list[np.ndarray] = []
    residual_gyro_virtual_blocks: list[np.ndarray] = []
    residual_gyro_real_blocks: list[np.ndarray] = []
    capture_diagnostics = []
    fitted_capture_ids = []

    total_acc_samples = 0
    total_gyro_samples = 0

    for real_sequence, virtual_sequence in paired_sequences:
        aligned_real_sequence, aligned_virtual_sequence, timebase_summary = prepare_sequences_for_alignment(real_sequence, virtual_sequence)
        real_index = _sensor_index(aligned_real_sequence.sensor_names, sensor_name)
        virtual_index = _sensor_index(aligned_virtual_sequence.sensor_names, sensor_name)
        if real_index is None or virtual_index is None:
            raise AlignmentFittingError(
                f"Sensor '{sensor_name}' is missing in capture '{real_sequence.capture_id}' for subject '{subject_id}'."
            )

        lag_samples = _estimate_capture_lag(
            real_acc=np.asarray(aligned_real_sequence.acc[:, real_index, :], dtype=np.float32),
            virt_acc=np.asarray(aligned_virtual_sequence.acc[:, virtual_index, :], dtype=np.float32),
            real_gyro=np.asarray(aligned_real_sequence.gyro[:, real_index, :], dtype=np.float32),
            virt_gyro=np.asarray(aligned_virtual_sequence.gyro[:, virtual_index, :], dtype=np.float32),
            config=config
        )
        aligned_real_acc, aligned_virt_acc = align_streams_with_lag(
            np.asarray(aligned_real_sequence.acc[:, real_index, :], dtype=np.float32),
            np.asarray(aligned_virtual_sequence.acc[:, virtual_index, :], dtype=np.float32),
            lag_samples
        )
        aligned_real_gyro, aligned_virt_gyro = align_streams_with_lag(
            np.asarray(aligned_real_sequence.gyro[:, real_index, :], dtype=np.float32),
            np.asarray(aligned_virtual_sequence.gyro[:, virtual_index, :], dtype=np.float32),
            lag_samples
        )

        valid_acc_mask = _finite_mask(aligned_real_acc, aligned_virt_acc)
        valid_gyro_mask = _finite_mask(aligned_real_gyro, aligned_virt_gyro)
        acc_mask = _select_acc_samples(
            real_acc=aligned_real_acc,
            virt_acc=aligned_virt_acc,
            real_gyro=aligned_real_gyro,
            virt_gyro=aligned_virt_gyro,
            valid_mask=valid_acc_mask,
            config=config
        )
        gyro_mask = _select_gyro_samples(
            real_gyro=aligned_real_gyro,
            virt_gyro=aligned_virt_gyro,
            valid_mask=valid_gyro_mask,
            config=config
        )

        if bool(config.use_acc) and int(np.count_nonzero(acc_mask)) > 0:
            rotation_real_blocks.append(aligned_real_acc[acc_mask])
            rotation_virtual_blocks.append(aligned_virt_acc[acc_mask])
            total_acc_samples += int(np.count_nonzero(acc_mask))
        if bool(config.use_gyro) and int(np.count_nonzero(gyro_mask)) > 0:
            rotation_real_blocks.append(aligned_real_gyro[gyro_mask])
            rotation_virtual_blocks.append(aligned_virt_gyro[gyro_mask])
            total_gyro_samples += int(np.count_nonzero(gyro_mask))

        residual_acc_mask = _finite_mask(aligned_real_acc, aligned_virt_acc)
        residual_gyro_mask = _finite_mask(aligned_real_gyro, aligned_virt_gyro)
        if int(np.count_nonzero(residual_acc_mask)) > 0:
            residual_acc_real_blocks.append(aligned_real_acc[residual_acc_mask])
            residual_acc_virtual_blocks.append(aligned_virt_acc[residual_acc_mask])
        if int(np.count_nonzero(residual_gyro_mask)) > 0:
            residual_gyro_real_blocks.append(aligned_real_gyro[residual_gyro_mask])
            residual_gyro_virtual_blocks.append(aligned_virt_gyro[residual_gyro_mask])

        capture_diagnostics.append(
            {
                "capture_id": str(real_sequence.capture_id),
                "lag_samples": int(lag_samples),
                "num_acc_samples": int(np.count_nonzero(acc_mask)),
                "num_gyro_samples": int(np.count_nonzero(gyro_mask)),
                "num_acc_valid_frames": int(np.count_nonzero(residual_acc_mask)),
                "num_gyro_valid_frames": int(np.count_nonzero(residual_gyro_mask)),
                "timebase_summary": dict(timebase_summary)
            }
        )
        fitted_capture_ids.append(str(real_sequence.capture_id))

    if len(rotation_virtual_blocks) == 0 or len(rotation_real_blocks) == 0:
        raise AlignmentFittingError(
            f"No informative samples were found to fit subject '{subject_id}' sensor '{sensor_name}'."
        )

    virtual_vectors = np.concatenate(rotation_virtual_blocks, axis=0).astype(np.float32, copy=False)
    real_vectors = np.concatenate(rotation_real_blocks, axis=0).astype(np.float32, copy=False)
    if virtual_vectors.shape[0] < _MIN_REQUIRED_SAMPLES:
        raise AlignmentFittingError(
            f"Need at least {_MIN_REQUIRED_SAMPLES} informative samples to fit subject '{subject_id}' sensor "
            f"'{sensor_name}', but found {virtual_vectors.shape[0]}."
        )
    if virtual_vectors.shape[0] < _LOW_SAMPLE_WARNING:
        warnings.warn(
            f"Low sample count while fitting subject '{subject_id}' sensor '{sensor_name}': "
            f"{virtual_vectors.shape[0]} samples.",
            RuntimeWarning,
            stacklevel=2
        )

    rotation = estimate_rotation_procrustes(virtual_vectors, real_vectors)

    acc_scale = np.ones(3, dtype=np.float32)
    acc_bias = np.zeros(3, dtype=np.float32)
    gyro_scale = np.ones(3, dtype=np.float32)
    gyro_bias = np.zeros(3, dtype=np.float32)

    calibration_metrics_before = {}
    calibration_metrics_after = {}

    if len(residual_acc_real_blocks) > 0:
        residual_real_acc = np.concatenate(residual_acc_real_blocks, axis=0).astype(np.float32, copy=False)
        residual_virtual_acc = np.concatenate(residual_acc_virtual_blocks, axis=0).astype(np.float32, copy=False)
    else:
        residual_real_acc = None
        residual_virtual_acc = None
    if len(residual_gyro_real_blocks) > 0:
        residual_real_gyro = np.concatenate(residual_gyro_real_blocks, axis=0).astype(np.float32, copy=False)
        residual_virtual_gyro = np.concatenate(residual_gyro_virtual_blocks, axis=0).astype(np.float32, copy=False)
    else:
        residual_real_gyro = None
        residual_virtual_gyro = None

    if residual_real_acc is not None and residual_virtual_acc is not None and (
        bool(config.use_bias) or bool(config.use_scale)
    ):
        rotated_virtual_acc = apply_rotation(residual_virtual_acc, rotation)
        acc_scale, acc_bias = _fit_axiswise_scale_bias(
            rotated_virtual_acc,
            residual_real_acc,
            use_bias=bool(config.use_bias),
            use_scale=bool(config.use_scale),
            ridge_lambda=float(config.ridge_lambda)
        )
    if residual_real_gyro is not None and residual_virtual_gyro is not None and (
        bool(config.use_bias) or bool(config.use_scale)
    ):
        rotated_virtual_gyro = apply_rotation(residual_virtual_gyro, rotation)
        gyro_scale, gyro_bias = _fit_axiswise_scale_bias(
            rotated_virtual_gyro,
            residual_real_gyro,
            use_bias=bool(config.use_bias),
            use_scale=bool(config.use_scale),
            ridge_lambda=float(config.ridge_lambda)
        )

    if residual_real_acc is not None or residual_real_gyro is not None:
        before_acc = None if residual_virtual_acc is None else residual_virtual_acc
        before_gyro = None if residual_virtual_gyro is None else residual_virtual_gyro
        after_acc = None
        after_gyro = None
        if residual_virtual_acc is not None:
            after_acc = _apply_scale_bias(apply_rotation(residual_virtual_acc, rotation), acc_scale, acc_bias)
        if residual_virtual_gyro is not None:
            after_gyro = _apply_scale_bias(apply_rotation(residual_virtual_gyro, rotation), gyro_scale, gyro_bias)
        calibration_metrics_before = summarize_alignment_metrics(
            real_acc=residual_real_acc,
            estimate_acc=before_acc,
            real_gyro=residual_real_gyro,
            estimate_gyro=before_gyro
        )
        calibration_metrics_after = summarize_alignment_metrics(
            real_acc=residual_real_acc,
            estimate_acc=after_acc,
            real_gyro=residual_real_gyro,
            estimate_gyro=after_gyro
        )

    singular_values = np.linalg.svd(real_vectors.T @ virtual_vectors, compute_uv=False)
    condition_number = np.inf
    if singular_values.size > 0 and float(singular_values[-1]) > _EPSILON:
        condition_number = float(singular_values[0] / singular_values[-1])

    diagnostics = {
        "num_samples_used": int(virtual_vectors.shape[0]),
        "num_acc_samples": int(total_acc_samples),
        "num_gyro_samples": int(total_gyro_samples),
        "num_calibration_captures": int(len(fitted_capture_ids)),
        "capture_diagnostics": capture_diagnostics,
        "condition_number": None if not np.isfinite(condition_number) else float(condition_number),
        "det_rotation": float(np.linalg.det(rotation)),
        "calibration_metrics_before": calibration_metrics_before,
        "calibration_metrics_after": calibration_metrics_after
    }
    return SensorSubjectTransform(
        subject_id=str(subject_id),
        sensor_name=str(sensor_name),
        rotation=rotation,
        acc_bias=acc_bias,
        gyro_bias=gyro_bias,
        acc_scale=acc_scale,
        gyro_scale=gyro_scale,
        fitted_capture_ids=fitted_capture_ids,
        diagnostics=diagnostics
    )


def _pair_sequences(real_sequences: Sequence[IMUSequence], virtual_sequences: Sequence[IMUSequence]) -> list[tuple[IMUSequence, IMUSequence]]:
    if len(real_sequences) == 0 or len(virtual_sequences) == 0:
        raise AlignmentFittingError("real_sequences and virtual_sequences must both be non-empty.")

    real_by_key = _index_sequences(real_sequences, "real")
    virtual_by_key = _index_sequences(virtual_sequences, "virtual")
    real_keys = set(real_by_key.keys())
    virtual_keys = set(virtual_by_key.keys())
    if real_keys != virtual_keys:
        missing_in_virtual = sorted(real_keys.difference(virtual_keys))
        missing_in_real = sorted(virtual_keys.difference(real_keys))
        raise AlignmentFittingError(
            "Real and virtual sequences must be paired by (subject_id, capture_id). "
            f"Missing in virtual: {missing_in_virtual}. Missing in real: {missing_in_real}."
        )

    pairs = []
    for key in sorted(real_keys):
        real_sequence = real_by_key[key]
        virtual_sequence = virtual_by_key[key]
        if set(real_sequence.sensor_names) != set(virtual_sequence.sensor_names):
            raise AlignmentFittingError(
                f"Real and virtual sensor layouts do not match for pair {key}: "
                f"{sorted(set(real_sequence.sensor_names))} != {sorted(set(virtual_sequence.sensor_names))}."
            )
        pairs.append((real_sequence, virtual_sequence))
    return pairs


def _index_sequences(sequences: Sequence[IMUSequence], label: str) -> dict[tuple[str, str], IMUSequence]:
    indexed: dict[tuple[str, str], IMUSequence] = {}
    for sequence in sequences:
        key = (str(sequence.subject_id), str(sequence.capture_id))
        if key in indexed:
            raise AlignmentFittingError(f"Duplicate {label} sequence for key {key}.")
        indexed[key] = sequence
    return indexed


def _collect_common_sensor_names(paired_sequences: Sequence[tuple[IMUSequence, IMUSequence]]) -> list[str]:
    sensor_sets = [set(real_sequence.sensor_names) for real_sequence, _ in paired_sequences]
    common_names = set.intersection(*sensor_sets)
    return sorted(str(name) for name in common_names)


def _sensor_index(sensor_names: Sequence[str], sensor_name: str) -> int | None:
    for index, candidate in enumerate(sensor_names):
        if str(candidate) == str(sensor_name):
            return int(index)
    return None


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


def _select_gyro_samples(*, real_gyro: np.ndarray, virt_gyro: np.ndarray, valid_mask: np.ndarray, config: AlignmentConfig) -> np.ndarray:
    if not bool(config.use_gyro):
        return np.zeros_like(valid_mask, dtype=bool)
    real_norm = np.linalg.norm(real_gyro, axis=1)
    virt_norm = np.linalg.norm(virt_gyro, axis=1)
    informative = (real_norm >= float(config.min_motion_norm_gyro)) & (virt_norm >= float(config.min_motion_norm_gyro))
    return np.asarray(valid_mask & informative, dtype=bool)


def _select_acc_samples(
    *,
    real_acc: np.ndarray,
    virt_acc: np.ndarray,
    real_gyro: np.ndarray,
    virt_gyro: np.ndarray,
    valid_mask: np.ndarray,
    config: AlignmentConfig,
) -> np.ndarray:
    if not bool(config.use_acc):
        return np.zeros_like(valid_mask, dtype=bool)
    if bool(config.gravity_window_only):
        real_gyro_norm = np.linalg.norm(real_gyro, axis=1)
        virt_gyro_norm = np.linalg.norm(virt_gyro, axis=1)
        informative = (real_gyro_norm <= float(config.min_motion_norm_gyro)) & (
            virt_gyro_norm <= float(config.min_motion_norm_gyro)
        )
        return np.asarray(valid_mask & informative, dtype=bool)
    real_norm = np.linalg.norm(real_acc, axis=1)
    virt_norm = np.linalg.norm(virt_acc, axis=1)
    informative = (real_norm >= float(config.min_motion_norm_acc)) & (
        virt_norm >= float(config.min_motion_norm_acc)
    )
    return np.asarray(valid_mask & informative, dtype=bool)


def _finite_mask(reference_xyz: np.ndarray, estimate_xyz: np.ndarray) -> np.ndarray:
    return np.isfinite(reference_xyz).all(axis=1) & np.isfinite(estimate_xyz).all(axis=1)


def _fit_axiswise_scale_bias(
    predictor_xyz: np.ndarray,
    target_xyz: np.ndarray,
    *,
    use_bias: bool,
    use_scale: bool,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    predictor = np.asarray(predictor_xyz, dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)
    scale = np.ones(3, dtype=np.float32)
    bias = np.zeros(3, dtype=np.float32)
    for axis_index in range(3):
        x_axis = predictor[:, axis_index]
        y_axis = target[:, axis_index]
        if not use_scale and not use_bias:
            continue
        if use_scale and use_bias:
            design = np.stack([x_axis, np.ones_like(x_axis)], axis=1)
            lhs = design.T @ design + float(ridge_lambda) * np.eye(2, dtype=np.float64)
            rhs = design.T @ y_axis
            coefficients = np.linalg.solve(lhs, rhs)
            scale[axis_index] = float(coefficients[0])
            bias[axis_index] = float(coefficients[1])
        elif use_scale:
            denominator = float(np.dot(x_axis, x_axis) + float(ridge_lambda))
            scale[axis_index] = 1.0 if denominator <= 0.0 else float(np.dot(x_axis, y_axis) / denominator)
        elif use_bias:
            bias[axis_index] = float(np.mean(y_axis - x_axis))
    return scale, bias


def _apply_scale_bias(values_xyz: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (
        np.asarray(values_xyz, dtype=np.float32) * np.asarray(scale, dtype=np.float32)[None, :]
        + np.asarray(bias, dtype=np.float32)[None, :]
    ).astype(np.float32, copy=False)
