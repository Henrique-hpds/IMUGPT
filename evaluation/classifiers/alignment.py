from __future__ import annotations

from typing import Any

import numpy as np
try:
    from scipy.signal import correlate, correlation_lags
except ImportError:  # pragma: no cover - exercised when scipy is unavailable
    def correlate(
        in1: np.ndarray,
        in2: np.ndarray,
        *,
        mode: str = "full",
        method: str = "auto",
    ) -> np.ndarray:
        if mode != "full":
            raise ValueError("NumPy fallback only supports mode='full'.")
        del method
        return np.correlate(in1, in2, mode="full")

    def correlation_lags(
        in1_len: int,
        in2_len: int,
        *,
        mode: str = "full",
    ) -> np.ndarray:
        if mode != "full":
            raise ValueError("NumPy fallback only supports mode='full'.")
        return np.arange(-(int(in2_len) - 1), int(in1_len), dtype=np.int64)

from pose_module.processing.frequency_alignment import estimate_sampling_frequency_hz

_EPSILON = 1e-8


def _as_time_series(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim < 1:
        raise ValueError(f"{name} must expose at least one temporal axis.")
    if array.shape[0] < 2:
        raise ValueError(f"{name} must contain at least 2 frames.")
    return array


def _as_timestamps(timestamps_sec: np.ndarray, *, name: str) -> np.ndarray:
    timestamps = np.asarray(timestamps_sec, dtype=np.float64)
    if timestamps.ndim != 1:
        raise ValueError(f"{name} must be a 1D timestamp array.")
    if timestamps.shape[0] < 2:
        raise ValueError(f"{name} must contain at least 2 samples.")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError(f"{name} contains non-finite values.")
    if np.any(np.diff(timestamps) < 0.0):
        raise ValueError(f"{name} must be sorted in ascending order.")
    return timestamps


def _flatten_channels(values: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    block = np.asarray(values, dtype=np.float32)
    trailing_shape = tuple(block.shape[1:])
    return block.reshape(block.shape[0], -1), trailing_shape


def resample_values_to_reference(
    source_timestamps_sec: np.ndarray,
    source_values: np.ndarray,
    reference_timestamps_sec: np.ndarray,
    *,
    method: str = "linear",
) -> np.ndarray:
    source_timestamps = _as_timestamps(source_timestamps_sec, name="source_timestamps_sec")
    reference_timestamps = _as_timestamps(reference_timestamps_sec, name="reference_timestamps_sec")
    values = _as_time_series(source_values, name="source_values")
    if values.shape[0] != source_timestamps.shape[0]:
        raise ValueError("source_values and source_timestamps_sec must share the same frame count.")

    if source_timestamps.shape == reference_timestamps.shape and np.allclose(source_timestamps, reference_timestamps):
        return np.asarray(values, dtype=np.float32).copy()

    normalized_method = str(method).strip().lower()
    if normalized_method not in {"linear", "nearest"}:
        raise ValueError("method must be either 'linear' or 'nearest'.")

    flat_values, trailing_shape = _flatten_channels(values)
    resampled_flat = np.empty((reference_timestamps.shape[0], flat_values.shape[1]), dtype=np.float32)

    if normalized_method == "nearest":
        right_indices = np.searchsorted(source_timestamps, reference_timestamps, side="left")
        right_indices = np.clip(right_indices, 0, source_timestamps.shape[0] - 1)
        left_indices = np.clip(right_indices - 1, 0, source_timestamps.shape[0] - 1)
        right_distance = np.abs(source_timestamps[right_indices] - reference_timestamps)
        left_distance = np.abs(source_timestamps[left_indices] - reference_timestamps)
        nearest_indices = np.where(right_distance < left_distance, right_indices, left_indices)
        resampled_flat[:] = flat_values[nearest_indices]
    else:
        for channel_index in range(flat_values.shape[1]):
            channel_values = flat_values[:, channel_index].astype(np.float64, copy=False)
            resampled_flat[:, channel_index] = np.interp(
                reference_timestamps,
                source_timestamps,
                channel_values,
            ).astype(np.float32, copy=False)

    return resampled_flat.reshape((reference_timestamps.shape[0], *trailing_shape)).astype(np.float32, copy=False)


def collapse_alignment_signal(values: np.ndarray) -> np.ndarray:
    block = _as_time_series(values, name="values").astype(np.float64, copy=False)
    flat_block = block.reshape(block.shape[0], -1)
    if flat_block.shape[1] == 1:
        return flat_block[:, 0].astype(np.float32, copy=False)

    centered = flat_block - np.nanmean(flat_block, axis=0, keepdims=True)
    std = np.nanstd(centered, axis=0, keepdims=True)
    standardized = centered / np.maximum(std, _EPSILON)
    summary = np.nanmean(np.abs(standardized), axis=1)
    return np.asarray(summary, dtype=np.float32)


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    valid_mask = np.isfinite(x_array) & np.isfinite(y_array)
    if int(np.count_nonzero(valid_mask)) < 3:
        return None
    x_centered = x_array[valid_mask] - float(np.mean(x_array[valid_mask]))
    y_centered = y_array[valid_mask] - float(np.mean(y_array[valid_mask]))
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator <= _EPSILON:
        return None
    return float(np.dot(x_centered, y_centered) / denominator)


def shift_values_with_nan(values: np.ndarray, lag_samples: int) -> np.ndarray:
    block = np.asarray(values, dtype=np.float32)
    shifted = np.full_like(block, np.nan, dtype=np.float32)
    lag = int(lag_samples)
    if lag == 0:
        shifted[:] = block
        return shifted

    if abs(lag) >= block.shape[0]:
        return shifted

    if lag > 0:
        shifted[lag:] = block[:-lag]
    else:
        lag_abs = abs(lag)
        shifted[:-lag_abs] = block[lag_abs:]
    return shifted


def estimate_lag_cross_correlation(
    reference_signal: np.ndarray,
    target_signal: np.ndarray,
    *,
    max_lag_samples: int = 20,
) -> dict[str, Any]:
    reference = np.asarray(reference_signal, dtype=np.float64).reshape(-1)
    target = np.asarray(target_signal, dtype=np.float64).reshape(-1)
    if reference.shape != target.shape:
        raise ValueError("reference_signal and target_signal must have the same shape.")
    if reference.shape[0] < 3:
        raise ValueError("At least 3 samples are required to estimate cross-correlation lag.")

    valid_mask = np.isfinite(reference) & np.isfinite(target)
    if int(np.count_nonzero(valid_mask)) < 3:
        return {
            "lag_samples": 0,
            "correlation": None,
            "num_valid_samples": int(np.count_nonzero(valid_mask)),
        }

    reference_valid = reference.copy()
    target_valid = target.copy()
    reference_valid[~valid_mask] = float(np.nanmean(reference[valid_mask]))
    target_valid[~valid_mask] = float(np.nanmean(target[valid_mask]))

    reference_centered = reference_valid - float(np.mean(reference_valid))
    target_centered = target_valid - float(np.mean(target_valid))
    denom = float(np.linalg.norm(reference_centered) * np.linalg.norm(target_centered))
    if denom <= _EPSILON:
        return {
            "lag_samples": 0,
            "correlation": None,
            "num_valid_samples": int(np.count_nonzero(valid_mask)),
        }

    full_corr = correlate(reference_centered, target_centered, mode="full", method="auto")
    full_lags = correlation_lags(reference_centered.size, target_centered.size, mode="full")
    bounded_mask = np.abs(full_lags) <= int(max_lag_samples)
    bounded_corr = full_corr[bounded_mask] / denom
    bounded_lags = full_lags[bounded_mask]
    best_index = int(np.argmax(bounded_corr))

    return {
        "lag_samples": int(bounded_lags[best_index]),
        "correlation": float(bounded_corr[best_index]),
        "num_valid_samples": int(np.count_nonzero(valid_mask)),
    }


def compute_constrained_dtw(
    reference_signal: np.ndarray,
    target_signal: np.ndarray,
    *,
    radius: int | None = None,
) -> dict[str, Any]:
    reference = np.asarray(reference_signal, dtype=np.float64).reshape(-1)
    target = np.asarray(target_signal, dtype=np.float64).reshape(-1)
    if reference.size == 0 or target.size == 0:
        raise ValueError("DTW requires non-empty input signals.")

    radius_value = max(reference.size, target.size) if radius is None else max(1, int(radius))
    cost = np.full((reference.size + 1, target.size + 1), np.inf, dtype=np.float64)
    predecessor = np.full((reference.size + 1, target.size + 1, 2), -1, dtype=np.int32)
    cost[0, 0] = 0.0

    for ref_index in range(1, reference.size + 1):
        start = max(1, ref_index - radius_value)
        end = min(target.size, ref_index + radius_value)
        for target_index in range(start, end + 1):
            local_cost = abs(reference[ref_index - 1] - target[target_index - 1])
            candidate_offsets = (
                (ref_index - 1, target_index),
                (ref_index, target_index - 1),
                (ref_index - 1, target_index - 1),
            )
            prev_costs = [cost[i, j] for i, j in candidate_offsets]
            best_prev_offset = int(np.argmin(prev_costs))
            prev_i, prev_j = candidate_offsets[best_prev_offset]
            cost[ref_index, target_index] = local_cost + cost[prev_i, prev_j]
            predecessor[ref_index, target_index] = (prev_i, prev_j)

    if not np.isfinite(cost[reference.size, target.size]):
        raise ValueError("DTW failed to find a valid path for the requested radius.")

    path_pairs: list[tuple[int, int]] = []
    current_i = reference.size
    current_j = target.size
    while current_i > 0 and current_j > 0:
        path_pairs.append((current_i - 1, current_j - 1))
        current_i, current_j = predecessor[current_i, current_j]
        if current_i < 0 or current_j < 0:
            break

    path_pairs.reverse()
    path = np.asarray(path_pairs, dtype=np.int32)
    path_length = max(1, path.shape[0])
    return {
        "path": path,
        "path_length": int(path_length),
        "distance": float(cost[reference.size, target.size]),
        "normalized_distance": float(cost[reference.size, target.size] / path_length),
    }


def warp_target_to_reference(values: np.ndarray, dtw_path: np.ndarray, reference_length: int) -> np.ndarray:
    block = np.asarray(values, dtype=np.float32)
    if block.shape[0] == 0:
        raise ValueError("values must not be empty.")
    path = np.asarray(dtw_path, dtype=np.int32)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("dtw_path must have shape [path_length, 2].")
    if reference_length <= 0:
        raise ValueError("reference_length must be greater than 0.")

    flat_block, trailing_shape = _flatten_channels(block)
    warped = np.zeros((int(reference_length), flat_block.shape[1]), dtype=np.float32)
    counts = np.zeros(int(reference_length), dtype=np.float32)

    for ref_index, target_index in path.tolist():
        warped[int(ref_index)] += flat_block[int(target_index)]
        counts[int(ref_index)] += 1.0

    valid_indices = counts > 0.0
    if not np.all(valid_indices):
        raise ValueError("DTW path did not cover every reference frame.")

    warped = warped / counts[:, None]
    return warped.reshape((int(reference_length), *trailing_shape)).astype(np.float32, copy=False)


def align_target_to_reference(
    reference_timestamps_sec: np.ndarray,
    reference_values: np.ndarray,
    target_timestamps_sec: np.ndarray,
    target_values: np.ndarray,
    *,
    reference_summary: np.ndarray | None = None,
    target_summary: np.ndarray | None = None,
    resample_method: str = "linear",
    max_lag_samples: int = 20,
    dtw_radius: int = 20,
) -> dict[str, Any]:
    reference_timestamps = _as_timestamps(reference_timestamps_sec, name="reference_timestamps_sec")
    target_timestamps = _as_timestamps(target_timestamps_sec, name="target_timestamps_sec")
    reference_block = _as_time_series(reference_values, name="reference_values")
    target_block = _as_time_series(target_values, name="target_values")
    if reference_block.shape[0] != reference_timestamps.shape[0]:
        raise ValueError("reference_values and reference_timestamps_sec must share the same frame count.")
    if target_block.shape[0] != target_timestamps.shape[0]:
        raise ValueError("target_values and target_timestamps_sec must share the same frame count.")

    overlap_start_sec = float(max(reference_timestamps[0], target_timestamps[0]))
    overlap_end_sec = float(min(reference_timestamps[-1], target_timestamps[-1]))
    if overlap_end_sec <= overlap_start_sec:
        raise ValueError("There is no temporal overlap between the reference and target signals.")

    reference_overlap_mask = (reference_timestamps >= overlap_start_sec) & (reference_timestamps <= overlap_end_sec)
    target_overlap_mask = (target_timestamps >= overlap_start_sec) & (target_timestamps <= overlap_end_sec)
    reference_overlap_timestamps = reference_timestamps[reference_overlap_mask]
    reference_overlap_values = reference_block[reference_overlap_mask]
    target_overlap_timestamps = target_timestamps[target_overlap_mask]
    target_overlap_values = target_block[target_overlap_mask]

    if reference_overlap_timestamps.shape[0] < 3 or target_overlap_timestamps.shape[0] < 3:
        raise ValueError("At least 3 overlapping samples are required to align reference and target.")

    target_resampled = resample_values_to_reference(
        target_overlap_timestamps,
        target_overlap_values,
        reference_overlap_timestamps,
        method=resample_method,
    )
    reference_summary_overlap = (
        collapse_alignment_signal(reference_overlap_values)
        if reference_summary is None
        else np.asarray(reference_summary, dtype=np.float32)[reference_overlap_mask]
    )
    target_summary_overlap = (
        collapse_alignment_signal(target_overlap_values)
        if target_summary is None
        else np.asarray(target_summary, dtype=np.float32)[target_overlap_mask]
    )
    target_summary_resampled = resample_values_to_reference(
        target_overlap_timestamps,
        target_summary_overlap[:, None],
        reference_overlap_timestamps,
        method=resample_method,
    )[:, 0]

    lag_report = estimate_lag_cross_correlation(
        reference_summary_overlap,
        target_summary_resampled,
        max_lag_samples=max_lag_samples,
    )
    lag_samples = int(lag_report["lag_samples"])

    target_shifted = shift_values_with_nan(target_resampled, lag_samples)
    target_summary_shifted = shift_values_with_nan(target_summary_resampled[:, None], lag_samples)[:, 0]

    valid_mask = np.isfinite(reference_summary_overlap) & np.isfinite(target_summary_shifted)
    if int(np.count_nonzero(valid_mask)) < 3:
        raise ValueError("The overlap after correlation-based lag alignment is too small.")

    valid_indices = np.flatnonzero(valid_mask)
    start_index = int(valid_indices[0])
    end_index = int(valid_indices[-1]) + 1

    reference_trimmed = np.asarray(reference_overlap_values[start_index:end_index], dtype=np.float32)
    target_trimmed = np.asarray(target_shifted[start_index:end_index], dtype=np.float32)
    reference_summary_trimmed = np.asarray(reference_summary_overlap[start_index:end_index], dtype=np.float32)
    target_summary_trimmed = np.asarray(target_summary_shifted[start_index:end_index], dtype=np.float32)
    trimmed_timestamps = np.asarray(reference_overlap_timestamps[start_index:end_index], dtype=np.float32)

    dtw_report = compute_constrained_dtw(
        reference_summary_trimmed,
        target_summary_trimmed,
        radius=dtw_radius,
    )
    target_warped = warp_target_to_reference(
        target_trimmed,
        dtw_report["path"],
        reference_trimmed.shape[0],
    )
    target_summary_warped = warp_target_to_reference(
        target_summary_trimmed[:, None],
        dtw_report["path"],
        reference_trimmed.shape[0],
    )[:, 0]

    aligned_frequency_hz = estimate_sampling_frequency_hz(trimmed_timestamps)
    lag_seconds = 0.0 if aligned_frequency_hz <= _EPSILON else float(lag_samples / aligned_frequency_hz)

    return {
        "timestamps_sec": trimmed_timestamps,
        "reference_values": reference_trimmed,
        "aligned_target_values": target_warped,
        "reference_indices": np.flatnonzero(reference_overlap_mask)[start_index:end_index].astype(np.int32),
        "lag_samples": lag_samples,
        "lag_seconds": lag_seconds,
        "correlation_before_dtw": pearson_correlation(reference_summary_trimmed, target_summary_trimmed),
        "correlation_after_dtw": pearson_correlation(reference_summary_trimmed, target_summary_warped),
        "dtw_distance": float(dtw_report["distance"]),
        "dtw_normalized_distance": float(dtw_report["normalized_distance"]),
        "dtw_path_length": int(dtw_report["path_length"]),
        "dtw_path": np.asarray(dtw_report["path"], dtype=np.int32),
        "aligned_frequency_hz": float(aligned_frequency_hz),
        "reference_summary": reference_summary_trimmed,
        "target_summary_resampled": target_summary_resampled[start_index:end_index].astype(np.float32, copy=False),
        "target_summary_shifted": target_summary_trimmed,
        "target_summary_aligned": target_summary_warped.astype(np.float32, copy=False),
        "diagnostics": {
            "resample_method": str(resample_method),
            "overlap_start_sec": overlap_start_sec,
            "overlap_end_sec": overlap_end_sec,
            "overlap_frames_before_trim": int(reference_overlap_values.shape[0]),
            "overlap_frames_after_trim": int(reference_trimmed.shape[0]),
            "num_valid_frames_after_lag": int(np.count_nonzero(valid_mask)),
            "lag_correlation": lag_report["correlation"],
            "max_lag_samples": int(max_lag_samples),
            "dtw_radius": int(dtw_radius),
        },
    }
