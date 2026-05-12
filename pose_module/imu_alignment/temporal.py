"""Temporal alignment helpers for IMU signals."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from pose_module.processing.frequency_alignment import (
    estimate_sampling_frequency_hz,
    undersample_signal_to_reference,
)

from .interfaces import IMUSequence

_TIMESTAMP_ATOL = 1e-5
_TIMESTAMP_RTOL = 1e-4


def prepare_sequences_for_alignment(real_sequence: IMUSequence, virt_sequence: IMUSequence):
    """Project the real sequence onto the virtual timestamps when needed.

    The geometric alignment logic operates on paired sample grids. In the real
    RobotEmotions exports the reference IMU is sampled more densely than the
    virtual IMU, so we first resample the real stream onto the virtual
    timestamps using nearest-neighbor matching.
    """

    if _timestamps_match(real_sequence.timestamps, virt_sequence.timestamps):
        return real_sequence, virt_sequence, {
            "resampled": False,
            "real_original_frames": int(real_sequence.num_frames),
            "real_aligned_frames": int(real_sequence.num_frames),
            "virtual_frames": int(virt_sequence.num_frames),
            "real_original_frequency_hz": _estimate_frequency_or_none(real_sequence.timestamps),
            "real_aligned_frequency_hz": _estimate_frequency_or_none(real_sequence.timestamps),
            "virtual_frequency_hz": _estimate_frequency_or_none(virt_sequence.timestamps),
            "mean_time_error_ms": 0.0,
            "max_time_error_ms": 0.0,
        }

    aligned_acc = undersample_signal_to_reference(
        source_timestamps_sec=real_sequence.timestamps,
        source_values=real_sequence.acc,
        reference_timestamps_sec=virt_sequence.timestamps,
    )
    aligned_gyro = undersample_signal_to_reference(
        source_timestamps_sec=real_sequence.timestamps,
        source_values=real_sequence.gyro,
        reference_timestamps_sec=virt_sequence.timestamps,
    )
    time_error_ms = np.abs(np.asarray(aligned_acc["time_error_sec"], dtype=np.float64)) * 1000.0
    aligned_real_sequence = IMUSequence(
        subject_id=real_sequence.subject_id,
        capture_id=real_sequence.capture_id,
        sensor_names=list(real_sequence.sensor_names),
        fps=virt_sequence.fps,
        timestamps=np.asarray(virt_sequence.timestamps, dtype=np.float32),
        acc=np.asarray(aligned_acc["values"], dtype=np.float32),
        gyro=np.asarray(aligned_gyro["values"], dtype=np.float32),
    )
    return aligned_real_sequence, virt_sequence, {
        "resampled": True,
        "real_original_frames": int(real_sequence.num_frames),
        "real_aligned_frames": int(aligned_real_sequence.num_frames),
        "virtual_frames": int(virt_sequence.num_frames),
        "real_original_frequency_hz": _estimate_frequency_or_none(real_sequence.timestamps),
        "real_aligned_frequency_hz": _estimate_frequency_or_none(aligned_real_sequence.timestamps),
        "virtual_frequency_hz": _estimate_frequency_or_none(virt_sequence.timestamps),
        "mean_time_error_ms": None if time_error_ms.size == 0 else float(np.mean(time_error_ms)),
        "max_time_error_ms": None if time_error_ms.size == 0 else float(np.max(time_error_ms)),
    }


def estimate_time_lag(real_xyz: np.ndarray, virt_xyz: np.ndarray, max_lag: int, mode: str) -> int:
    """Estimate the sample lag that best aligns real and virtual signal norms.

    The returned lag follows the convention used by :func:`align_streams_with_lag`.
    Positive lag means the real stream is delayed relative to the virtual stream,
    so the aligned overlap is ``real[lag:]`` against ``virt[:-lag]``.
    """

    if mode not in {"gyro_norm", "acc_norm"}:
        raise ValueError("estimate_time_lag mode must be 'gyro_norm' or 'acc_norm'.")

    real = _validate_xyz_signal(real_xyz, "real_xyz")
    virt = _validate_xyz_signal(virt_xyz, "virt_xyz")
    max_lag = int(max_lag)
    if max_lag < 0:
        raise ValueError("estimate_time_lag max_lag must be non-negative.")

    real_norm = np.linalg.norm(real, axis=1)
    virt_norm = np.linalg.norm(virt, axis=1)

    best_lag = 0
    best_score = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        aligned_real, aligned_virt = align_streams_with_lag(real_norm, virt_norm, lag)
        finite_mask = np.isfinite(aligned_real) & np.isfinite(aligned_virt)
        if int(np.count_nonzero(finite_mask)) < 3:
            continue
        score = _centered_correlation(aligned_real[finite_mask], aligned_virt[finite_mask])
        if score > best_score:
            best_score = score
            best_lag = lag
    return int(best_lag)


def align_streams_with_lag(real_values: np.ndarray, virt_values: np.ndarray, lag_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return the overlapping slices implied by a discrete lag."""

    real = np.asarray(real_values)
    virt = np.asarray(virt_values)
    lag_samples = int(lag_samples)
    if real.shape[0] == 0 or virt.shape[0] == 0:
        return real[:0], virt[:0]
    if lag_samples > 0:
        if lag_samples >= min(real.shape[0], virt.shape[0]):
            return real[:0], virt[:0]
        return real[lag_samples:], virt[:-lag_samples]
    if lag_samples < 0:
        lag = abs(lag_samples)
        if lag >= min(real.shape[0], virt.shape[0]):
            return real[:0], virt[:0]
        return real[:-lag], virt[lag:]
    return real, virt


def _validate_xyz_signal(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape [T, 3].")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    return array


def _centered_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    y_centered = np.asarray(y, dtype=np.float64) - float(np.mean(y))
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator <= 0.0:
        return -np.inf
    return float(np.dot(x_centered, y_centered) / denominator)


def _timestamps_match(real_timestamps: np.ndarray, virt_timestamps: np.ndarray) -> bool:
    real = np.asarray(real_timestamps, dtype=np.float64)
    virt = np.asarray(virt_timestamps, dtype=np.float64)
    return real.shape == virt.shape and np.allclose(real, virt, atol=_TIMESTAMP_ATOL, rtol=_TIMESTAMP_RTOL)


def _estimate_frequency_or_none(timestamps: np.ndarray) -> float | None:
    values = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] < 2:
        return None
    try:
        return float(estimate_sampling_frequency_hz(values))
    except ValueError:
        return None
