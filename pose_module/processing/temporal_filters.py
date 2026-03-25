"""Temporal interpolation and smoothing helpers for stage 5.4."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def interpolate_short_gaps(
    values: np.ndarray,
    valid_mask: np.ndarray,
    *,
    max_gap: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate interior gaps up to ``max_gap`` samples."""

    array = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    original_shape = array.shape
    if array.ndim == 1:
        array = array[:, None]
    if valid.ndim == 1:
        valid = valid[:, None]
    if array.shape != valid.shape:
        raise ValueError("values and valid_mask must have the same shape")

    interpolated = np.zeros_like(valid, dtype=bool)
    output = array.copy()
    num_frames, num_dims = output.shape

    if num_frames == 0 or int(max_gap) <= 0:
        return output.reshape(original_shape), interpolated.reshape(original_shape)

    for dim_index in range(num_dims):
        dim_valid = valid[:, dim_index]
        frame_index = 0
        while frame_index < num_frames:
            if dim_valid[frame_index]:
                frame_index += 1
                continue

            gap_start = frame_index
            while frame_index < num_frames and not dim_valid[frame_index]:
                frame_index += 1
            gap_end = frame_index
            gap_length = int(gap_end - gap_start)

            left_index = gap_start - 1
            right_index = gap_end
            if (
                gap_length <= int(max_gap)
                and left_index >= 0
                and right_index < num_frames
                and dim_valid[left_index]
                and dim_valid[right_index]
            ):
                start_value = float(output[left_index, dim_index])
                end_value = float(output[right_index, dim_index])
                for offset in range(gap_length):
                    alpha = float(offset + 1) / float(gap_length + 1)
                    output[gap_start + offset, dim_index] = ((1.0 - alpha) * start_value) + (
                        alpha * end_value
                    )
                    interpolated[gap_start + offset, dim_index] = True

    return output.reshape(original_shape), interpolated.reshape(original_shape)


def savgol_smooth(
    values: np.ndarray,
    *,
    window_length: int,
    polyorder: int,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing with a small numpy-only implementation."""

    array = np.asarray(values, dtype=np.float32)
    original_shape = array.shape
    if array.ndim == 1:
        array = array[:, None]

    if valid_mask is None:
        valid = np.isfinite(array)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.ndim == 1:
            valid = valid[:, None]
        if valid.shape != array.shape:
            raise ValueError("valid_mask must match values shape")

    num_frames = int(array.shape[0])
    adjusted_window = _adjust_window_length(
        requested=int(window_length),
        num_frames=num_frames,
        polyorder=int(polyorder),
    )
    if adjusted_window is None:
        return array.reshape(original_shape)

    coeffs = _compute_savgol_coeffs(adjusted_window, int(polyorder))
    filled = _fill_invalid_by_linear_interpolation(array, valid)
    if filled.size == 0:
        return array.reshape(original_shape)

    pad = adjusted_window // 2
    padded = np.pad(filled, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.empty_like(filled)
    for frame_index in range(num_frames):
        window_slice = padded[frame_index : frame_index + adjusted_window]
        smoothed[frame_index] = np.tensordot(coeffs, window_slice, axes=(0, 0))

    return smoothed.reshape(original_shape)


def _adjust_window_length(
    *,
    requested: int,
    num_frames: int,
    polyorder: int,
) -> int | None:
    if num_frames <= int(polyorder):
        return None

    window_length = max(int(requested), int(polyorder) + 1, 3)
    if window_length % 2 == 0:
        window_length += 1
    if window_length > num_frames:
        window_length = num_frames if num_frames % 2 == 1 else num_frames - 1
    if window_length <= int(polyorder):
        return None
    return int(window_length)


def _compute_savgol_coeffs(window_length: int, polyorder: int) -> np.ndarray:
    half_window = window_length // 2
    offsets = np.arange(-half_window, half_window + 1, dtype=np.float64)
    design = np.vander(offsets, int(polyorder) + 1, increasing=True)
    return np.linalg.pinv(design)[0].astype(np.float32)


def _fill_invalid_by_linear_interpolation(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    output = values.copy()
    num_frames, num_dims = output.shape
    frame_axis = np.arange(num_frames, dtype=np.float32)
    for dim_index in range(num_dims):
        dim_valid = valid_mask[:, dim_index] & np.isfinite(output[:, dim_index])
        if not np.any(dim_valid):
            output[:, dim_index] = 0.0
            continue
        valid_indices = frame_axis[dim_valid]
        valid_values = output[dim_valid, dim_index]
        output[:, dim_index] = np.interp(frame_axis, valid_indices, valid_values)
    return output
