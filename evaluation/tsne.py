from __future__ import annotations

from fractions import Fraction
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.fft import rfft, rfftfreq
from scipy.signal import resample, resample_poly
from sklearn.manifold import TSNE

from evaluation.utils import (
    load_real_capture,
    load_virtual_capture,
    select_capture_row,
)
from pose_module.processing.frequency_alignment import estimate_sampling_frequency_hz

AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
SIGNAL_GROUP_TO_REAL_SLICE = {"acc": slice(0, 3), "gyro": slice(3, 6)}
SIGNAL_GROUP_TO_VIRTUAL_KEY = {"acc": "acc", "gyro": "gyro"}
DOMAIN_TO_LABEL = {"real": 0, "synthetic": 1}
LABEL_TO_DOMAIN = {0: "real", 1: "synthetic"}
DOMAIN_COLORS = {"real": "tab:blue", "synthetic": "tab:orange"}
CLASS_HIGHLIGHT_DOMAIN_COLORS = {"real": "tab:blue", "synthetic": "tab:red"}
CLASS_HIGHLIGHT_BACKGROUND_COLORS = {"real": "#9ecae1", "synthetic": "#fcbba1"}
CLASS_HIGHLIGHT_MARKER = "o"


def _normalize_string_list(values: Iterable[str]) -> list[str]:
    return [str(value).strip() for value in values]


def _validate_signal_groups(signal_groups: Sequence[str]) -> list[str]:
    normalized = [str(group).strip().lower() for group in signal_groups]
    invalid = [group for group in normalized if group not in SIGNAL_GROUP_TO_REAL_SLICE]
    if invalid:
        raise ValueError(f"Invalid signal_groups: {invalid}. Use 'acc' and/or 'gyro'.")
    if len(normalized) == 0:
        raise ValueError("signal_groups cannot be empty.")
    return normalized


def _validate_axes(selected_axes: Sequence[str]) -> list[str]:
    normalized = [str(axis).strip().lower() for axis in selected_axes]
    invalid = [axis for axis in normalized if axis not in AXIS_TO_INDEX]
    if invalid:
        raise ValueError(f"Invalid selected_axes: {invalid}. Use 'x', 'y', and/or 'z'.")
    if len(normalized) == 0:
        raise ValueError("selected_axes cannot be empty.")
    return normalized


def _resolve_selected_sensors(
    requested_sensors: Sequence[str] | None,
    real_sensor_names: Sequence[str],
    synthetic_sensor_names: Sequence[str],
) -> list[str]:
    common_sensors = [sensor for sensor in real_sensor_names if sensor in set(synthetic_sensor_names)]
    if len(common_sensors) == 0:
        raise ValueError("There are no sensors shared by the real and synthetic signals.")

    if requested_sensors is None:
        return common_sensors

    selected_sensors = _normalize_string_list(requested_sensors)
    missing = [sensor for sensor in selected_sensors if sensor not in set(common_sensors)]
    if missing:
        raise ValueError(f"Sensors not found in both domains: {missing}")
    return selected_sensors


def _flatten_for_interpolation(values: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    if values.ndim < 2:
        raise ValueError("values must include a time axis and at least one channel axis.")
    trailing_shape = tuple(values.shape[1:])
    return values.reshape(values.shape[0], -1), trailing_shape


def _interpolate_multichannel(
    source_timestamps_sec: np.ndarray,
    source_values: np.ndarray,
    target_timestamps_sec: np.ndarray,
) -> np.ndarray:
    flat_values, trailing_shape = _flatten_for_interpolation(source_values)
    interpolated = np.empty((target_timestamps_sec.shape[0], flat_values.shape[1]), dtype=np.float64)

    for channel_index in range(flat_values.shape[1]):
        interpolated[:, channel_index] = np.interp(
            target_timestamps_sec,
            source_timestamps_sec,
            flat_values[:, channel_index],
        )

    return interpolated.reshape((target_timestamps_sec.shape[0], *trailing_shape)).astype(np.float32)


def load_capture_pair(
    captures_df: pd.DataFrame,
    *,
    domain: str,
    user_id: int,
    tag_number: int,
    take_id: str | None = None,
    synthetic_filename: str = "virtual_imu.npz",
) -> dict[str, Any]:
    capture_row = select_capture_row(
        captures_df,
        domain=domain,
        user_id=user_id,
        tag_number=tag_number,
        take_id=take_id,
    )

    real_capture = load_real_capture(capture_row["clip_dir"])
    synthetic_capture = load_virtual_capture(capture_row["pose_dir"], filename=synthetic_filename)

    return {
        "capture_row": capture_row,
        "real": real_capture,
        "synthetic": synthetic_capture,
    }


def load_capture_pair_from_row(
    capture_row: pd.Series | dict[str, Any],
    *,
    synthetic_filename: str = "virtual_imu.npz",
) -> dict[str, Any]:
    row = pd.Series(capture_row).copy()
    real_capture = load_real_capture(row["clip_dir"])
    synthetic_capture = load_virtual_capture(row["pose_dir"], filename=synthetic_filename)
    return {
        "capture_row": row,
        "real": real_capture,
        "synthetic": synthetic_capture,
    }


def extract_selected_modalities(
    capture_pair: dict[str, Any],
    *,
    signal_groups: Sequence[str] = ("acc",),
    selected_sensors: Sequence[str] | None = None,
    selected_axes: Sequence[str] = ("x", "y", "z"),
) -> dict[str, Any]:
    real_capture = dict(capture_pair["real"])
    synthetic_capture = dict(capture_pair["synthetic"])

    normalized_groups = _validate_signal_groups(signal_groups)
    normalized_axes = _validate_axes(selected_axes)
    resolved_sensors = _resolve_selected_sensors(
        requested_sensors=selected_sensors,
        real_sensor_names=real_capture["sensor_names"],
        synthetic_sensor_names=synthetic_capture["sensor_names"],
    )

    real_sensor_indices = [real_capture["sensor_names"].index(sensor) for sensor in resolved_sensors]
    synthetic_sensor_indices = [synthetic_capture["sensor_names"].index(sensor) for sensor in resolved_sensors]
    axis_indices = [AXIS_TO_INDEX[axis] for axis in normalized_axes]

    real_blocks = []
    synthetic_blocks = []
    channel_labels: list[str] = []

    for signal_group in normalized_groups:
        real_block = real_capture["imu"][:, real_sensor_indices, SIGNAL_GROUP_TO_REAL_SLICE[signal_group]]
        synthetic_block = synthetic_capture[SIGNAL_GROUP_TO_VIRTUAL_KEY[signal_group]][:, synthetic_sensor_indices, :]

        real_blocks.append(real_block[:, :, axis_indices])
        synthetic_blocks.append(synthetic_block[:, :, axis_indices])

        for axis_name in normalized_axes:
            channel_labels.append(f"{signal_group}_{axis_name}")

    real_values = np.concatenate(real_blocks, axis=2).astype(np.float32)
    synthetic_values = np.concatenate(synthetic_blocks, axis=2).astype(np.float32)

    if real_values.shape[1:] != synthetic_values.shape[1:]:
        raise ValueError(
            "Sensor/channel dimensions do not match between real and synthetic: "
            f"{real_values.shape[1:]} vs {synthetic_values.shape[1:]}"
        )

    return {
        "capture_row": capture_pair["capture_row"],
        "real_timestamps_sec": np.asarray(real_capture["timestamps_sec"], dtype=np.float64),
        "real_values": real_values,
        "real_sensor_names": list(resolved_sensors),
        "synthetic_timestamps_sec": np.asarray(synthetic_capture["timestamps_sec"], dtype=np.float64),
        "synthetic_values": synthetic_values,
        "synthetic_sensor_names": list(resolved_sensors),
        "selected_sensors": list(resolved_sensors),
        "selected_axes": normalized_axes,
        "signal_groups": normalized_groups,
        "channel_labels": channel_labels,
    }


def summarize_selected_signals(signal_bundle: dict[str, Any]) -> pd.DataFrame:
    real_values = np.asarray(signal_bundle["real_values"])
    synthetic_values = np.asarray(signal_bundle["synthetic_values"])

    rows = [
        {
            "domain": "real",
            "num_frames": int(real_values.shape[0]),
            "num_sensors": int(real_values.shape[1]),
            "num_channels_per_sensor": int(real_values.shape[2]),
            "sampling_frequency_hz": float(estimate_sampling_frequency_hz(signal_bundle["real_timestamps_sec"])),
            "selected_sensors": ", ".join(signal_bundle["selected_sensors"]),
            "selected_axes": ", ".join(signal_bundle["selected_axes"]),
            "signal_groups": ", ".join(signal_bundle["signal_groups"]),
        },
        {
            "domain": "synthetic",
            "num_frames": int(synthetic_values.shape[0]),
            "num_sensors": int(synthetic_values.shape[1]),
            "num_channels_per_sensor": int(synthetic_values.shape[2]),
            "sampling_frequency_hz": float(estimate_sampling_frequency_hz(signal_bundle["synthetic_timestamps_sec"])),
            "selected_sensors": ", ".join(signal_bundle["selected_sensors"]),
            "selected_axes": ", ".join(signal_bundle["selected_axes"]),
            "signal_groups": ", ".join(signal_bundle["signal_groups"]),
        },
    ]
    return pd.DataFrame(rows)


def resample_real_to_synthetic_rate(
    *,
    real_timestamps_sec: np.ndarray,
    real_values: np.ndarray,
    synthetic_timestamps_sec: np.ndarray,
    synthetic_values: np.ndarray,
    method: str = "resample_poly",
    max_denominator: int = 1000,
) -> dict[str, Any]:
    real_timestamps = np.asarray(real_timestamps_sec, dtype=np.float64)
    synthetic_timestamps = np.asarray(synthetic_timestamps_sec, dtype=np.float64)
    real_block = np.asarray(real_values, dtype=np.float32)
    synthetic_block = np.asarray(synthetic_values, dtype=np.float32)

    if real_block.shape[0] != real_timestamps.shape[0]:
        raise ValueError("real_values and real_timestamps_sec must have the same number of frames.")
    if synthetic_block.shape[0] != synthetic_timestamps.shape[0]:
        raise ValueError("synthetic_values and synthetic_timestamps_sec must have the same number of frames.")
    if real_block.shape[1:] != synthetic_block.shape[1:]:
        raise ValueError(
            "real_values and synthetic_values must have the same sensors/channels after selection."
        )

    overlap_start_sec = float(max(real_timestamps[0], synthetic_timestamps[0]))
    overlap_end_sec = float(min(real_timestamps[-1], synthetic_timestamps[-1]))
    if overlap_end_sec <= overlap_start_sec:
        raise ValueError("There is no temporal overlap between the real and synthetic signals.")

    real_mask = (real_timestamps >= overlap_start_sec) & (real_timestamps <= overlap_end_sec)
    synthetic_mask = (synthetic_timestamps >= overlap_start_sec) & (synthetic_timestamps <= overlap_end_sec)
    real_overlap_timestamps = real_timestamps[real_mask]
    real_overlap_values = real_block[real_mask]
    synthetic_overlap_timestamps = synthetic_timestamps[synthetic_mask]
    synthetic_overlap_values = synthetic_block[synthetic_mask]

    if real_overlap_timestamps.size < 2 or synthetic_overlap_timestamps.size < 2:
        raise ValueError("At least 2 samples are required in each domain within the overlap range.")

    real_frequency_hz = float(estimate_sampling_frequency_hz(real_overlap_timestamps))
    synthetic_frequency_hz = float(estimate_sampling_frequency_hz(synthetic_overlap_timestamps))

    normalized_method = str(method).strip().lower()
    if normalized_method not in {"resample", "resample_poly"}:
        raise ValueError("method must be either 'resample' or 'resample_poly'.")

    if normalized_method == "resample":
        resampled_real_values = resample(real_overlap_values, num=synthetic_overlap_timestamps.size, axis=0).astype(
            np.float32
        )
        effective_method = "resample"
    else:
        ratio = Fraction(synthetic_frequency_hz / real_frequency_hz).limit_denominator(max_denominator)
        resampled_uniform_values = resample_poly(
            real_overlap_values,
            up=ratio.numerator,
            down=ratio.denominator,
            axis=0,
        ).astype(np.float32)
        uniform_timestamps_sec = overlap_start_sec + (
            np.arange(resampled_uniform_values.shape[0], dtype=np.float64) / synthetic_frequency_hz
        )
        valid_mask = uniform_timestamps_sec <= (overlap_end_sec + (0.5 / synthetic_frequency_hz))
        uniform_timestamps_sec = uniform_timestamps_sec[valid_mask]
        resampled_uniform_values = resampled_uniform_values[valid_mask]

        if uniform_timestamps_sec.size < 2:
            raise ValueError("resample_poly produced too few usable samples.")

        resampled_real_values = _interpolate_multichannel(
            source_timestamps_sec=uniform_timestamps_sec,
            source_values=resampled_uniform_values,
            target_timestamps_sec=synthetic_overlap_timestamps,
        )
        effective_method = f"resample_poly(up={ratio.numerator}, down={ratio.denominator})"

    if resampled_real_values.shape != synthetic_overlap_values.shape:
        raise ValueError(
            "Aligned signals must have the same shape. "
            f"Obtido: {resampled_real_values.shape} vs {synthetic_overlap_values.shape}"
        )

    return {
        "timestamps_sec": synthetic_overlap_timestamps.astype(np.float32),
        "real_resampled_values": resampled_real_values,
        "synthetic_values": synthetic_overlap_values.astype(np.float32),
        "real_frequency_hz": real_frequency_hz,
        "synthetic_frequency_hz": synthetic_frequency_hz,
        "overlap_start_sec": overlap_start_sec,
        "overlap_end_sec": overlap_end_sec,
        "resample_method": effective_method,
    }


def summarize_alignment(aligned_bundle: dict[str, Any]) -> pd.DataFrame:
    aligned_timestamps = np.asarray(aligned_bundle["timestamps_sec"], dtype=np.float64)
    return pd.DataFrame(
        [
            {
                "aligned_frames": int(aligned_timestamps.size),
                "aligned_sampling_frequency_hz": float(estimate_sampling_frequency_hz(aligned_timestamps)),
                "real_frequency_hz_before": float(aligned_bundle["real_frequency_hz"]),
                "synthetic_frequency_hz": float(aligned_bundle["synthetic_frequency_hz"]),
                "overlap_start_sec": float(aligned_bundle["overlap_start_sec"]),
                "overlap_end_sec": float(aligned_bundle["overlap_end_sec"]),
                "overlap_duration_sec": float(aligned_bundle["overlap_end_sec"] - aligned_bundle["overlap_start_sec"]),
                "resample_method": str(aligned_bundle["resample_method"]),
            }
        ]
    )


def resolve_window_spec(
    *,
    sampling_frequency_hz: float | None = None,
    window_type: str | None = None,
    window_size: int | None = None,
    window_duration_sec: float | None = None,
) -> dict[str, Any]:
    normalized_window_type = None if window_type is None else str(window_type).strip().lower()
    if normalized_window_type not in {None, "n_samples", "seconds"}:
        raise ValueError("window_type must be either 'n_samples' or 'seconds'.")

    if normalized_window_type == "n_samples":
        if window_size is None or window_duration_sec is not None:
            raise ValueError("For window_type='n_samples', use window_size only.")
    if normalized_window_type == "seconds":
        if window_duration_sec is None or window_size is not None:
            raise ValueError("For window_type='seconds', use window_duration_sec only.")

    if window_size is not None and window_duration_sec is not None:
        raise ValueError("Use either window_size or window_duration_sec, not both.")
    if window_size is None and window_duration_sec is None:
        raise ValueError("You must provide window_size or window_duration_sec.")

    if window_duration_sec is not None:
        if sampling_frequency_hz is None:
            raise ValueError("sampling_frequency_hz is required when window_duration_sec is used.")
        duration_sec = float(window_duration_sec)
        if duration_sec <= 0.0:
            raise ValueError("window_duration_sec must be greater than 0.")

        window_size_samples = int(round(duration_sec * float(sampling_frequency_hz)))
        if window_size_samples <= 1:
            raise ValueError(
                "window_duration_sec is too short for the current sampling frequency. "
                f"Resolved samples: {window_size_samples}"
            )

        return {
            "window_type": "seconds",
            "window_size_samples": window_size_samples,
            "window_duration_sec": duration_sec,
        }

    if int(window_size) <= 1:
        raise ValueError("window_size must be greater than 1.")

    duration_sec = None if sampling_frequency_hz is None else float(int(window_size) / float(sampling_frequency_hz))
    return {
        "window_type": "n_samples",
        "window_size_samples": int(window_size),
        "window_duration_sec": duration_sec,
    }


def resolve_stride_or_overlap_spec(
    *,
    window_type: str,
    window_size_samples: int,
    sampling_frequency_hz: float | None = None,
    stride_or_overlap_mode: str | None = None,
    step_value: float | None = None,
    overlap: float = 0.5,
    stride: int | None = None,
    stride_sec: float | None = None,
    stride_or_overlap: float | None = None,
) -> dict[str, Any]:
    normalized_mode = None if stride_or_overlap_mode is None else str(stride_or_overlap_mode).strip().lower()
    if normalized_mode not in {None, "stride", "overlap"}:
        raise ValueError("stride_or_overlap_mode must be either 'stride' or 'overlap'.")

    explicit_count = sum(value is not None for value in (stride, stride_sec, stride_or_overlap, step_value))
    if explicit_count > 1:
        raise ValueError("Use only one of step_value, stride, stride_sec, or stride_or_overlap.")

    if normalized_mode is not None:
        if normalized_mode == "overlap":
            value = float(overlap if step_value is None else step_value)
            if not 0.0 < value < 1.0:
                raise ValueError("For overlap mode, overlap must be in the (0, 1) interval.")
            step_size_samples = max(1, int(round(window_size_samples * (1.0 - value))))
            step_duration_sec = None if sampling_frequency_hz is None else float(step_size_samples / sampling_frequency_hz)
            return {
                "step_mode": "overlap",
                "step_size_samples": step_size_samples,
                "step_duration_sec": step_duration_sec,
                "overlap_ratio": float(value),
                "stride_value": None,
            }

        if step_value is not None:
            value = float(step_value)
        elif stride is not None:
            value = float(int(stride))
        elif stride_sec is not None:
            value = float(stride_sec)
        elif stride_or_overlap is not None:
            value = float(stride_or_overlap)
        else:
            raise ValueError(
                "When stride_or_overlap_mode='stride', provide step_value, stride, stride_sec, or stride_or_overlap."
            )

        if value <= 0.0:
            raise ValueError("For stride mode, stride must be greater than 0.")

        if stride_sec is not None or (step_value is not None and window_type == "seconds"):
            if sampling_frequency_hz is None:
                raise ValueError("sampling_frequency_hz is required when a time stride is used.")
            step_size_samples = int(round(value * float(sampling_frequency_hz)))
            if step_size_samples <= 0:
                raise ValueError("Resolved stride is too small.")
            overlap_ratio = None if step_size_samples > window_size_samples else float(1.0 - (step_size_samples / window_size_samples))
            return {
                "step_mode": "stride",
                "step_size_samples": step_size_samples,
                "step_duration_sec": float(value),
                "overlap_ratio": overlap_ratio,
                "stride_value": float(value),
            }

        step_size_samples = int(round(value))
        if step_size_samples <= 0:
            raise ValueError("Resolved stride from step_value must be greater than 0.")
        overlap_ratio = None if step_size_samples > window_size_samples else float(1.0 - (step_size_samples / window_size_samples))
        return {
            "step_mode": "stride",
            "step_size_samples": step_size_samples,
            "step_duration_sec": None if sampling_frequency_hz is None else float(step_size_samples / sampling_frequency_hz),
            "overlap_ratio": overlap_ratio,
            "stride_value": int(step_size_samples),
        }

    if stride_or_overlap is not None:
        value = float(stride_or_overlap)
        if 0.0 < value < 1.0:
            overlap_ratio = value
            step_size_samples = max(1, int(round(window_size_samples * (1.0 - overlap_ratio))))
            step_duration_sec = None if sampling_frequency_hz is None else float(step_size_samples / sampling_frequency_hz)
            return {
                "step_mode": "overlap",
                "step_size_samples": step_size_samples,
                "step_duration_sec": step_duration_sec,
                "overlap_ratio": overlap_ratio,
                "stride_value": None,
            }

        if value <= 0.0:
            raise ValueError("stride_or_overlap must be greater than 0 when used as a stride.")

        if window_type == "seconds":
            if sampling_frequency_hz is None:
                raise ValueError("sampling_frequency_hz is required when stride_or_overlap is a time stride.")
            step_size_samples = int(round(value * float(sampling_frequency_hz)))
            if step_size_samples <= 0:
                raise ValueError("Resolved stride from stride_or_overlap is too small.")
            overlap_ratio = None if step_size_samples > window_size_samples else float(1.0 - (step_size_samples / window_size_samples))
            return {
                "step_mode": "stride",
                "step_size_samples": step_size_samples,
                "step_duration_sec": float(value),
                "overlap_ratio": overlap_ratio,
                "stride_value": float(value),
            }

        step_size_samples = int(round(value))
        if step_size_samples <= 0:
            raise ValueError("Resolved stride from stride_or_overlap must be greater than 0.")
        overlap_ratio = None if step_size_samples > window_size_samples else float(1.0 - (step_size_samples / window_size_samples))
        return {
            "step_mode": "stride",
            "step_size_samples": step_size_samples,
            "step_duration_sec": None if sampling_frequency_hz is None else float(step_size_samples / sampling_frequency_hz),
            "overlap_ratio": overlap_ratio,
            "stride_value": int(step_size_samples),
        }

    if stride is not None:
        step_size_samples = int(stride)
        if step_size_samples <= 0:
            raise ValueError("stride must be greater than 0.")
        overlap_ratio = None if step_size_samples > window_size_samples else float(1.0 - (step_size_samples / window_size_samples))
        return {
            "step_mode": "stride",
            "step_size_samples": step_size_samples,
            "step_duration_sec": None if sampling_frequency_hz is None else float(step_size_samples / sampling_frequency_hz),
            "overlap_ratio": overlap_ratio,
            "stride_value": int(step_size_samples),
        }

    if stride_sec is not None:
        if sampling_frequency_hz is None:
            raise ValueError("sampling_frequency_hz is required when stride_sec is used.")
        stride_sec = float(stride_sec)
        if stride_sec <= 0.0:
            raise ValueError("stride_sec must be greater than 0.")
        step_size_samples = int(round(stride_sec * float(sampling_frequency_hz)))
        if step_size_samples <= 0:
            raise ValueError("Resolved stride from stride_sec is too small.")
        overlap_ratio = None if step_size_samples > window_size_samples else float(1.0 - (step_size_samples / window_size_samples))
        return {
            "step_mode": "stride",
            "step_size_samples": step_size_samples,
            "step_duration_sec": stride_sec,
            "overlap_ratio": overlap_ratio,
            "stride_value": stride_sec,
        }

    if not 0.0 <= float(overlap) < 1.0:
        raise ValueError("overlap must be in the [0, 1) interval.")

    step_size_samples = max(1, int(round(window_size_samples * (1.0 - float(overlap)))))
    step_duration_sec = None if sampling_frequency_hz is None else float(step_size_samples / sampling_frequency_hz)
    return {
        "step_mode": "overlap",
        "step_size_samples": step_size_samples,
        "step_duration_sec": step_duration_sec,
        "overlap_ratio": float(overlap),
        "stride_value": None,
    }


def segment_signal_windows(
    values: np.ndarray,
    *,
    window_type: str | None = None,
    window_size: int | None = None,
    window_duration_sec: float | None = None,
    sampling_frequency_hz: float | None = None,
    stride_or_overlap_mode: str | None = None,
    step_value: float | None = None,
    overlap: float = 0.5,
    stride: int | None = None,
    stride_sec: float | None = None,
    stride_or_overlap: float | None = None,
) -> dict[str, Any]:
    block = np.asarray(values, dtype=np.float32)
    if block.ndim != 3:
        raise ValueError("values must have shape [frames, sensors, channels].")

    window_spec = resolve_window_spec(
        sampling_frequency_hz=sampling_frequency_hz,
        window_type=window_type,
        window_size=window_size,
        window_duration_sec=window_duration_sec,
    )
    window_size = int(window_spec["window_size_samples"])
    step_spec = resolve_stride_or_overlap_spec(
        window_type=str(window_spec["window_type"]),
        window_size_samples=window_size,
        sampling_frequency_hz=sampling_frequency_hz,
        stride_or_overlap_mode=stride_or_overlap_mode,
        step_value=step_value,
        overlap=overlap,
        stride=stride,
        stride_sec=stride_sec,
        stride_or_overlap=stride_or_overlap,
    )
    step_size = int(step_spec["step_size_samples"])
    if block.shape[0] < window_size:
        raise ValueError(
            f"There are not enough frames for window_size={window_size}. "
            f"Available frames: {block.shape[0]}"
        )

    start_indices = np.arange(0, block.shape[0] - window_size + 1, step_size, dtype=np.int32)
    windows = np.stack([block[start : start + window_size] for start in start_indices], axis=0).astype(np.float32)

    return {
        "windows": windows,
        "start_indices": start_indices,
        "step_size": step_size,
        "step_duration_sec": step_spec["step_duration_sec"],
        "step_mode": step_spec["step_mode"],
        "stride_value": step_spec["stride_value"],
        "window_size": window_size,
        "window_type": window_spec["window_type"],
        "window_duration_sec": window_spec["window_duration_sec"],
        "overlap": step_spec["overlap_ratio"],
    }


def transform_windows_to_frequency_domain(
    windows: np.ndarray,
    *,
    sampling_frequency_hz: float,
    normalization: str = "zscore",
    use_log_power: bool = True,
    drop_dc: bool = False,
) -> dict[str, Any]:
    window_block = np.asarray(windows, dtype=np.float32)
    if window_block.ndim != 4:
        raise ValueError("windows must have shape [windows, frames, sensors, channels].")

    spectra = np.abs(rfft(window_block, axis=1)).astype(np.float32)
    frequencies_hz = rfftfreq(window_block.shape[1], d=1.0 / float(sampling_frequency_hz)).astype(np.float32)

    if drop_dc:
        spectra = spectra[:, 1:, :, :]
        frequencies_hz = frequencies_hz[1:]

    if use_log_power:
        spectra = np.log1p(spectra).astype(np.float32)

    features = spectra.reshape(spectra.shape[0], -1).astype(np.float32)
    normalized_mode = str(normalization).strip().lower()

    if normalized_mode == "zscore":
        feature_mean = features.mean(axis=1, keepdims=True)
        feature_std = features.std(axis=1, keepdims=True)
        features = ((features - feature_mean) / np.maximum(feature_std, 1e-8)).astype(np.float32)
    elif normalized_mode == "energy":
        feature_energy = np.sum(np.abs(features), axis=1, keepdims=True)
        features = (features / np.maximum(feature_energy, 1e-8)).astype(np.float32)
    else:
        raise ValueError("normalization must be either 'zscore' or 'energy'.")

    return {
        "features": features,
        "spectra": spectra,
        "frequencies_hz": frequencies_hz,
        "normalization": normalized_mode,
        "use_log_power": bool(use_log_power),
    }


def build_frequency_dataset(
    real_features: np.ndarray,
    synthetic_features: np.ndarray,
) -> dict[str, Any]:
    real_matrix = np.asarray(real_features, dtype=np.float32)
    synthetic_matrix = np.asarray(synthetic_features, dtype=np.float32)

    if real_matrix.ndim != 2 or synthetic_matrix.ndim != 2:
        raise ValueError("Feature matrices must have shape [samples, dimensions].")
    if real_matrix.shape[1] != synthetic_matrix.shape[1]:
        raise ValueError(
            "Real and synthetic must have the same feature dimensionality. "
            f"Obtido: {real_matrix.shape[1]} vs {synthetic_matrix.shape[1]}"
        )

    features = np.concatenate([real_matrix, synthetic_matrix], axis=0).astype(np.float32)
    labels = np.concatenate(
        [
            np.full(real_matrix.shape[0], DOMAIN_TO_LABEL["real"], dtype=np.int32),
            np.full(synthetic_matrix.shape[0], DOMAIN_TO_LABEL["synthetic"], dtype=np.int32),
        ],
        axis=0,
    )
    domains = np.array([LABEL_TO_DOMAIN[label] for label in labels], dtype=object)

    return {
        "features": features,
        "labels": labels,
        "domains": domains,
    }


def fit_tsne_embedding(
    features: np.ndarray,
    *,
    perplexity: float = 30.0,
    init: str = "pca",
    random_state: int = 42,
) -> np.ndarray:
    feature_matrix = np.asarray(features, dtype=np.float32)
    if feature_matrix.ndim != 2:
        raise ValueError("features must have shape [samples, dimensions].")
    if feature_matrix.shape[0] < 3:
        raise ValueError("t-SNE requires at least 3 samples.")

    safe_perplexity = min(float(perplexity), float(max(2, feature_matrix.shape[0] - 1)))
    tsne = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        init=init,
        random_state=int(random_state),
        learning_rate="auto",
    )
    return tsne.fit_transform(feature_matrix).astype(np.float32)


def build_embedding_frame(
    embedding: np.ndarray,
    labels: np.ndarray,
    *,
    window_size: int,
) -> pd.DataFrame:
    embedding_matrix = np.asarray(embedding, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.int32)

    if embedding_matrix.shape[0] != label_array.shape[0]:
        raise ValueError("embedding and labels must have the same number of samples.")

    domain_names = [LABEL_TO_DOMAIN[int(label)] for label in label_array]
    return pd.DataFrame(
        {
            "tsne_1": embedding_matrix[:, 0],
            "tsne_2": embedding_matrix[:, 1],
            "label": label_array,
            "domain": domain_names,
            "window_size": int(window_size),
        }
    )


def plot_tsne_embedding(
    embedding_df: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    point_size: float = 40.0,
    alpha: float = 0.8,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    for domain_name in ("real", "synthetic"):
        domain_frame = embedding_df[embedding_df["domain"] == domain_name]
        ax.scatter(
            domain_frame["tsne_1"],
            domain_frame["tsne_2"],
            label=domain_name,
            c=DOMAIN_COLORS[domain_name],
            alpha=float(alpha),
            s=float(point_size),
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title or "2D t-SNE in the frequency domain")
    ax.legend()
    return ax


def _ordered_non_null_category_values(series: pd.Series) -> list[str]:
    non_null_series = series[series.notna()]
    if non_null_series.empty:
        return []
    return [str(value) for value in pd.unique(non_null_series.astype(str))]


def _resolve_subplot_grid(num_panels: int, *, max_columns: int = 3) -> tuple[int, int]:
    if int(num_panels) <= 0:
        raise ValueError("num_panels must be greater than 0.")
    if int(max_columns) <= 0:
        raise ValueError("max_columns must be greater than 0.")
    num_columns = min(int(max_columns), int(num_panels))
    num_rows = int(np.ceil(int(num_panels) / num_columns))
    return num_rows, num_columns


def plot_tsne_class_highlight_grid(
    embedding_df: pd.DataFrame,
    *,
    category: str,
    title: str | None = None,
    point_size: float = 18.0,
    background_alpha: float = 0.18,
    highlight_alpha: float = 0.9,
    max_columns: int = 3,
    subplot_size: tuple[float, float] = (4.8, 4.1),
) -> tuple[plt.Figure, np.ndarray]:
    if category not in embedding_df.columns:
        raise ValueError(f"Category column '{category}' is not available in embedding_df.")
    if "domain" not in embedding_df.columns:
        raise ValueError("embedding_df must include a 'domain' column for class/domain comparison plots.")
    if embedding_df.empty:
        raise ValueError("embedding_df cannot be empty.")

    category_values = _ordered_non_null_category_values(embedding_df[category])
    if len(category_values) == 0:
        raise ValueError(f"Category column '{category}' does not contain any non-null class value.")

    num_rows, num_columns = _resolve_subplot_grid(len(category_values), max_columns=max_columns)
    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(subplot_size[0] * num_columns, subplot_size[1] * num_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes_flat = axes.ravel()

    x_values = embedding_df["tsne_1"].to_numpy(dtype=np.float32)
    y_values = embedding_df["tsne_2"].to_numpy(dtype=np.float32)
    x_padding = max(1e-6, float(x_values.max() - x_values.min()) * 0.05)
    y_padding = max(1e-6, float(y_values.max() - y_values.min()) * 0.05)
    x_limits = (float(x_values.min() - x_padding), float(x_values.max() + x_padding))
    y_limits = (float(y_values.min() - y_padding), float(y_values.max() + y_padding))
    normalized_category = embedding_df[category].astype(object)
    highlightable_mask = normalized_category.notna()
    normalized_category = normalized_category.where(~highlightable_mask, normalized_category.astype(str))
    domain_series = embedding_df["domain"].astype(str)

    for class_index, class_name in enumerate(category_values):
        axis = axes_flat[class_index]
        highlight_mask = highlightable_mask & (normalized_category == class_name)
        background_mask = ~highlight_mask
        real_background_mask = background_mask & (domain_series == "real")
        synthetic_background_mask = background_mask & (domain_series == "synthetic")
        real_highlight_mask = highlight_mask & (domain_series == "real")
        synthetic_highlight_mask = highlight_mask & (domain_series == "synthetic")

        for domain_name, domain_mask in (
            ("real", real_background_mask),
            ("synthetic", synthetic_background_mask),
        ):
            axis.scatter(
                embedding_df.loc[domain_mask, "tsne_1"],
                embedding_df.loc[domain_mask, "tsne_2"],
                c=CLASS_HIGHLIGHT_BACKGROUND_COLORS[domain_name],
                alpha=float(background_alpha),
                s=float(point_size),
                linewidths=0.0,
                marker=CLASS_HIGHLIGHT_MARKER,
            )

        for domain_name, domain_mask in (
            ("real", real_highlight_mask),
            ("synthetic", synthetic_highlight_mask),
        ):
            axis.scatter(
                embedding_df.loc[domain_mask, "tsne_1"],
                embedding_df.loc[domain_mask, "tsne_2"],
                c=CLASS_HIGHLIGHT_DOMAIN_COLORS[domain_name],
                alpha=float(highlight_alpha),
                s=float(point_size),
                linewidths=0.0,
                marker=CLASS_HIGHLIGHT_MARKER,
            )

        axis.set_title(
            f"{class_name} (real={int(real_highlight_mask.sum())}, "
            f"synthetic={int(synthetic_highlight_mask.sum())})"
        )
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_xlabel("t-SNE 1")
        axis.set_ylabel("t-SNE 2")

    for axis in axes_flat[len(category_values) :]:
        axis.set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=CLASS_HIGHLIGHT_MARKER,
            linestyle="",
            markerfacecolor=CLASS_HIGHLIGHT_DOMAIN_COLORS["real"],
            markeredgecolor=CLASS_HIGHLIGHT_DOMAIN_COLORS["real"],
            markersize=8,
            label="real",
        ),
        Line2D(
            [0],
            [0],
            marker=CLASS_HIGHLIGHT_MARKER,
            linestyle="",
            markerfacecolor=CLASS_HIGHLIGHT_DOMAIN_COLORS["synthetic"],
            markeredgecolor=CLASS_HIGHLIGHT_DOMAIN_COLORS["synthetic"],
            markersize=8,
            label="synthetic",
        ),
    ]

    figure.suptitle(title or f"2D t-SNE real vs synthetic by {category}")
    figure.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.97))
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return figure, axes


def summarize_windows(
    *,
    window_type: str,
    real_windows: np.ndarray,
    synthetic_windows: np.ndarray,
    window_size: int,
    window_duration_sec: float | None,
    step_mode: str,
    step_size_samples: int,
    step_duration_sec: float | None,
    overlap_ratio: float | None,
    overlap: float,
    feature_dim: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "window_type": str(window_type),
                "window_size_samples": int(window_size),
                "window_duration_sec": None if window_duration_sec is None else float(window_duration_sec),
                "step_mode": str(step_mode),
                "step_size_samples": int(step_size_samples),
                "step_duration_sec": None if step_duration_sec is None else float(step_duration_sec),
                "overlap_ratio": None if overlap_ratio is None else float(overlap_ratio),
                "overlap": None if overlap is None else float(overlap),
                "real_num_windows": int(real_windows.shape[0]),
                "synthetic_num_windows": int(synthetic_windows.shape[0]),
                "feature_dim": int(feature_dim),
            }
        ]
    )


def _prepare_frequency_domain_tsne_capture(
    capture_pair: dict[str, Any],
    *,
    signal_groups: Sequence[str] = ("acc",),
    selected_sensors: Sequence[str] | None = None,
    selected_axes: Sequence[str] = ("x", "y", "z"),
    resample_method: str = "resample_poly",
    window_type: str | None = None,
    window_size: int | None = 128,
    window_duration_sec: float | None = None,
    stride_or_overlap_mode: str | None = None,
    step_value: float | None = None,
    stride_or_overlap: float | None = None,
    overlap: float = 0.5,
    stride: int | None = None,
    stride_sec: float | None = None,
    normalization: str = "zscore",
    use_log_power: bool = True,
    drop_dc: bool = False,
) -> dict[str, Any]:
    selected_bundle = extract_selected_modalities(
        capture_pair,
        signal_groups=signal_groups,
        selected_sensors=selected_sensors,
        selected_axes=selected_axes,
    )
    selected_summary_df = summarize_selected_signals(selected_bundle)

    aligned_bundle = resample_real_to_synthetic_rate(
        real_timestamps_sec=selected_bundle["real_timestamps_sec"],
        real_values=selected_bundle["real_values"],
        synthetic_timestamps_sec=selected_bundle["synthetic_timestamps_sec"],
        synthetic_values=selected_bundle["synthetic_values"],
        method=resample_method,
    )
    alignment_summary_df = summarize_alignment(aligned_bundle)
    aligned_frequency_hz = float(alignment_summary_df.loc[0, "aligned_sampling_frequency_hz"])
    resolved_window_spec = resolve_window_spec(
        sampling_frequency_hz=aligned_frequency_hz,
        window_type=window_type,
        window_size=window_size,
        window_duration_sec=window_duration_sec,
    )
    resolved_window_size = int(resolved_window_spec["window_size_samples"])
    resolved_window_duration_sec = resolved_window_spec["window_duration_sec"]
    resolved_window_type = str(resolved_window_spec["window_type"])

    real_windows_bundle = segment_signal_windows(
        aligned_bundle["real_resampled_values"],
        window_type="n_samples",
        window_size=resolved_window_size,
        window_duration_sec=None,
        sampling_frequency_hz=aligned_frequency_hz,
        stride_or_overlap_mode=stride_or_overlap_mode,
        step_value=step_value,
        stride_or_overlap=stride_or_overlap,
        overlap=overlap,
        stride=stride,
        stride_sec=stride_sec,
    )
    synthetic_windows_bundle = segment_signal_windows(
        aligned_bundle["synthetic_values"],
        window_type="n_samples",
        window_size=resolved_window_size,
        window_duration_sec=None,
        sampling_frequency_hz=aligned_frequency_hz,
        stride_or_overlap_mode=stride_or_overlap_mode,
        step_value=step_value,
        stride_or_overlap=stride_or_overlap,
        overlap=overlap,
        stride=stride,
        stride_sec=stride_sec,
    )

    if real_windows_bundle["windows"].shape != synthetic_windows_bundle["windows"].shape:
        raise ValueError(
            "Both domains must produce the same window shape after alignment. "
            f"Received: {real_windows_bundle['windows'].shape} vs {synthetic_windows_bundle['windows'].shape}"
        )

    real_frequency_bundle = transform_windows_to_frequency_domain(
        real_windows_bundle["windows"],
        sampling_frequency_hz=aligned_frequency_hz,
        normalization=normalization,
        use_log_power=use_log_power,
        drop_dc=drop_dc,
    )
    synthetic_frequency_bundle = transform_windows_to_frequency_domain(
        synthetic_windows_bundle["windows"],
        sampling_frequency_hz=aligned_frequency_hz,
        normalization=normalization,
        use_log_power=use_log_power,
        drop_dc=drop_dc,
    )

    window_summary_df = summarize_windows(
        window_type=resolved_window_type,
        real_windows=real_windows_bundle["windows"],
        synthetic_windows=synthetic_windows_bundle["windows"],
        window_size=resolved_window_size,
        window_duration_sec=resolved_window_duration_sec,
        step_mode=str(real_windows_bundle["step_mode"]),
        step_size_samples=int(real_windows_bundle["step_size"]),
        step_duration_sec=real_windows_bundle["step_duration_sec"],
        overlap_ratio=real_windows_bundle["overlap"],
        overlap=real_windows_bundle["overlap"],
        feature_dim=real_frequency_bundle["features"].shape[1],
    )

    return {
        "capture_row": selected_bundle["capture_row"],
        "selected_summary_df": selected_summary_df,
        "alignment_summary_df": alignment_summary_df,
        "window_summary_df": window_summary_df,
        "selected_bundle": selected_bundle,
        "aligned_bundle": aligned_bundle,
        "real_windows_bundle": real_windows_bundle,
        "synthetic_windows_bundle": synthetic_windows_bundle,
        "real_frequency_bundle": real_frequency_bundle,
        "synthetic_frequency_bundle": synthetic_frequency_bundle,
        "window_type": resolved_window_type,
        "window_size": resolved_window_size,
        "window_duration_sec": resolved_window_duration_sec,
        "frequencies_hz": real_frequency_bundle["frequencies_hz"],
    }


def _sample_feature_indices(
    num_rows: int,
    *,
    max_samples: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_samples is None or int(max_samples) >= int(num_rows):
        return np.arange(num_rows, dtype=np.int32)
    return np.sort(rng.choice(num_rows, size=int(max_samples), replace=False)).astype(np.int32)


def run_frequency_domain_tsne_all_captures(
    captures_df: pd.DataFrame,
    *,
    signal_groups: Sequence[str] = ("acc",),
    selected_sensors: Sequence[str] | None = None,
    selected_axes: Sequence[str] = ("x", "y", "z"),
    synthetic_filename: str = "virtual_imu.npz",
    resample_method: str = "resample_poly",
    window_type: str | None = None,
    window_size: int | None = 128,
    window_duration_sec: float | None = None,
    stride_or_overlap_mode: str | None = None,
    step_value: float | None = None,
    stride_or_overlap: float | None = None,
    overlap: float = 0.5,
    stride: int | None = None,
    stride_sec: float | None = None,
    normalization: str = "zscore",
    perplexity: float = 30.0,
    init: str = "pca",
    random_state: int = 42,
    use_log_power: bool = True,
    drop_dc: bool = False,
    max_windows_per_capture_per_domain: int | None = 128,
    skip_invalid_captures: bool = False,
    figsize: tuple[float, float] = (9.5, 7.0),
    title: str | None = None,
    show: bool = True,
) -> dict[str, Any]:
    capture_frame = captures_df.copy().reset_index(drop=True)
    if capture_frame.empty:
        raise ValueError("captures_df cannot be empty for the aggregated t-SNE.")
    if max_windows_per_capture_per_domain is not None and int(max_windows_per_capture_per_domain) <= 0:
        raise ValueError("max_windows_per_capture_per_domain must be greater than 0 when provided.")

    rng = np.random.default_rng(int(random_state))
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    metadata_frames: list[pd.DataFrame] = []
    capture_summary_rows: list[dict[str, Any]] = []
    failed_capture_rows: list[dict[str, Any]] = []

    reference_feature_dim: int | None = None
    reference_window_size: int | None = None
    reference_window_duration_sec: float | None = None
    reference_selected_bundle: dict[str, Any] | None = None
    reference_frequencies_hz: np.ndarray | None = None

    for capture_index, (_, capture_row) in enumerate(capture_frame.iterrows()):
        try:
            capture_pair = load_capture_pair_from_row(capture_row, synthetic_filename=synthetic_filename)
            capture_result = _prepare_frequency_domain_tsne_capture(
                capture_pair,
                signal_groups=signal_groups,
                selected_sensors=selected_sensors,
                selected_axes=selected_axes,
                resample_method=resample_method,
                window_type=window_type,
                window_size=window_size,
                window_duration_sec=window_duration_sec,
                stride_or_overlap_mode=stride_or_overlap_mode,
                step_value=step_value,
                stride_or_overlap=stride_or_overlap,
                overlap=overlap,
                stride=stride,
                stride_sec=stride_sec,
                normalization=normalization,
                use_log_power=use_log_power,
                drop_dc=drop_dc,
            )
        except Exception as exc:
            failed_row = pd.Series(capture_row).to_dict()
            failed_row["error"] = str(exc)
            failed_capture_rows.append(failed_row)
            if not skip_invalid_captures:
                raise
            continue

        real_features = np.asarray(capture_result["real_frequency_bundle"]["features"], dtype=np.float32)
        synthetic_features = np.asarray(capture_result["synthetic_frequency_bundle"]["features"], dtype=np.float32)
        current_feature_dim = int(real_features.shape[1])
        current_window_size = int(capture_result["window_size"])
        current_window_duration_sec = capture_result["window_duration_sec"]

        if reference_feature_dim is None:
            reference_feature_dim = current_feature_dim
            reference_window_size = current_window_size
            reference_window_duration_sec = current_window_duration_sec
            reference_selected_bundle = capture_result["selected_bundle"]
            reference_frequencies_hz = np.asarray(capture_result["frequencies_hz"], dtype=np.float32)
        elif current_feature_dim != reference_feature_dim:
            raise ValueError(
                "All captures must resolve the same feature dimensionality for the aggregated t-SNE. "
                f"Received {current_feature_dim} vs {reference_feature_dim}."
            )
        elif current_window_size != reference_window_size:
            raise ValueError(
                "All captures must resolve the same window size in samples for the aggregated t-SNE. "
                f"Received {current_window_size} vs {reference_window_size}."
            )

        sampled_real_indices = _sample_feature_indices(
            real_features.shape[0],
            max_samples=max_windows_per_capture_per_domain,
            rng=rng,
        )
        sampled_synthetic_indices = _sample_feature_indices(
            synthetic_features.shape[0],
            max_samples=max_windows_per_capture_per_domain,
            rng=rng,
        )

        feature_blocks.extend(
            [
                real_features[sampled_real_indices],
                synthetic_features[sampled_synthetic_indices],
            ]
        )
        label_blocks.extend(
            [
                np.full(sampled_real_indices.size, DOMAIN_TO_LABEL["real"], dtype=np.int32),
                np.full(sampled_synthetic_indices.size, DOMAIN_TO_LABEL["synthetic"], dtype=np.int32),
            ]
        )

        base_metadata = {
            "capture_index": int(capture_index),
            "capture_domain": str(capture_result["capture_row"]["domain"]),
            "user_id": int(capture_result["capture_row"]["user_id"]),
            "tag_number": int(capture_result["capture_row"]["tag_number"]),
            "take_id": None
            if pd.isna(capture_result["capture_row"].get("take_id"))
            else str(capture_result["capture_row"].get("take_id")),
            "clip_id": str(capture_result["capture_row"]["clip_id"]),
            "emotion": None
            if pd.isna(capture_result["capture_row"].get("emotion"))
            else str(capture_result["capture_row"].get("emotion")),
            "modality": None
            if pd.isna(capture_result["capture_row"].get("modality"))
            else str(capture_result["capture_row"].get("modality")),
            "stimulus": None
            if pd.isna(capture_result["capture_row"].get("stimulus"))
            else str(capture_result["capture_row"].get("stimulus")),
        }
        metadata_frames.extend(
            [
                pd.DataFrame(
                    {
                        **base_metadata,
                        "sample_index_within_capture": sampled_real_indices.astype(np.int32),
                        "sampled_domain": "real",
                    }
                ),
                pd.DataFrame(
                    {
                        **base_metadata,
                        "sample_index_within_capture": sampled_synthetic_indices.astype(np.int32),
                        "sampled_domain": "synthetic",
                    }
                ),
            ]
        )

        capture_summary_rows.append(
            {
                **base_metadata,
                "aligned_sampling_frequency_hz": float(
                    capture_result["alignment_summary_df"].loc[0, "aligned_sampling_frequency_hz"]
                ),
                "window_size_samples": int(
                    capture_result["window_summary_df"].loc[0, "window_size_samples"]
                ),
                "window_duration_sec": capture_result["window_summary_df"].loc[0, "window_duration_sec"],
                "real_num_windows_total": int(real_features.shape[0]),
                "synthetic_num_windows_total": int(synthetic_features.shape[0]),
                "sampled_real_num_windows": int(sampled_real_indices.size),
                "sampled_synthetic_num_windows": int(sampled_synthetic_indices.size),
                "feature_dim": current_feature_dim,
            }
        )

    if len(feature_blocks) == 0:
        raise ValueError("No capture could be processed for the aggregated t-SNE.")

    dataset_bundle = {
        "features": np.concatenate(feature_blocks, axis=0).astype(np.float32),
        "labels": np.concatenate(label_blocks, axis=0).astype(np.int32),
    }
    embedding = fit_tsne_embedding(
        dataset_bundle["features"],
        perplexity=perplexity,
        init=init,
        random_state=random_state,
    )
    embedding_df = build_embedding_frame(
        embedding,
        dataset_bundle["labels"],
        window_size=int(reference_window_size),
    )
    embedding_metadata_df = pd.concat(metadata_frames, axis=0, ignore_index=True)
    embedding_df = pd.concat([embedding_df, embedding_metadata_df], axis=1)

    figure, axis = plt.subplots(figsize=figsize)
    auto_title = (
        f"2D t-SNE | captures={len(capture_summary_rows)} | "
        f"sensors={', '.join(reference_selected_bundle['selected_sensors'])} | "
        f"groups={', '.join(reference_selected_bundle['signal_groups'])} | "
        f"window={int(reference_window_size)} samples"
    )
    if reference_window_duration_sec is not None:
        auto_title += f" ({float(reference_window_duration_sec):.3f} s)"
    if max_windows_per_capture_per_domain is not None:
        auto_title += f" | max/capture/domain={int(max_windows_per_capture_per_domain)}"
    plot_tsne_embedding(
        embedding_df,
        ax=axis,
        title=title or auto_title,
        point_size=18.0,
        alpha=0.45,
    )
    figure.tight_layout()
    if show:
        plt.show()

    class_highlight_figures: dict[str, plt.Figure] = {}
    class_highlight_axes: dict[str, np.ndarray] = {}
    for category_name in ("emotion", "modality", "stimulus"):
        if category_name not in embedding_df.columns:
            continue
        category_values = _ordered_non_null_category_values(embedding_df[category_name])
        if len(category_values) == 0:
            continue

        class_figure, class_axes = plot_tsne_class_highlight_grid(
            embedding_df,
            category=category_name,
            title=f"2D t-SNE real vs synthetic by {category_name}",
            point_size=18.0,
        )
        class_highlight_figures[category_name] = class_figure
        class_highlight_axes[category_name] = class_axes
        if show:
            plt.show()

    capture_summary_df = pd.DataFrame(capture_summary_rows)
    failed_capture_df = pd.DataFrame(failed_capture_rows)
    aggregate_summary_df = pd.DataFrame(
        [
            {
                "input_num_captures": int(capture_frame.shape[0]),
                "processed_num_captures": int(capture_summary_df.shape[0]),
                "failed_num_captures": int(failed_capture_df.shape[0]),
                "num_points_total": int(embedding_df.shape[0]),
                "num_real_points": int((embedding_df["domain"] == "real").sum()),
                "num_synthetic_points": int((embedding_df["domain"] == "synthetic").sum()),
                "max_windows_per_capture_per_domain": (
                    None
                    if max_windows_per_capture_per_domain is None
                    else int(max_windows_per_capture_per_domain)
                ),
                "window_size_samples": int(reference_window_size),
                "window_duration_sec": reference_window_duration_sec,
                "feature_dim": int(reference_feature_dim),
            }
        ]
    )

    return {
        "aggregate_summary_df": aggregate_summary_df,
        "capture_summary_df": capture_summary_df,
        "failed_capture_df": failed_capture_df,
        "embedding_df": embedding_df,
        "feature_matrix": dataset_bundle["features"],
        "labels": dataset_bundle["labels"],
        "frequencies_hz": reference_frequencies_hz,
        "figure": figure,
        "axis": axis,
        "class_highlight_figures": class_highlight_figures,
        "class_highlight_axes": class_highlight_axes,
        "selected_capture_df": capture_frame,
    }


def run_frequency_domain_tsne(
    captures_df: pd.DataFrame,
    *,
    domain: str,
    user_id: int,
    tag_number: int,
    take_id: str | None = None,
    signal_groups: Sequence[str] = ("acc",),
    selected_sensors: Sequence[str] | None = None,
    selected_axes: Sequence[str] = ("x", "y", "z"),
    synthetic_filename: str = "virtual_imu.npz",
    resample_method: str = "resample_poly",
    window_type: str | None = None,
    window_size: int | None = 128,
    window_duration_sec: float | None = None,
    stride_or_overlap_mode: str | None = None,
    step_value: float | None = None,
    stride_or_overlap: float | None = None,
    overlap: float = 0.5,
    stride: int | None = None,
    stride_sec: float | None = None,
    normalization: str = "zscore",
    perplexity: float = 30.0,
    init: str = "pca",
    random_state: int = 42,
    use_log_power: bool = True,
    drop_dc: bool = False,
    figsize: tuple[float, float] = (8.5, 6.5),
    title: str | None = None,
    show: bool = True,
) -> dict[str, Any]:
    capture_pair = load_capture_pair(
        captures_df,
        domain=domain,
        user_id=user_id,
        tag_number=tag_number,
        take_id=take_id,
        synthetic_filename=synthetic_filename,
    )
    capture_result = _prepare_frequency_domain_tsne_capture(
        capture_pair,
        signal_groups=signal_groups,
        selected_sensors=selected_sensors,
        selected_axes=selected_axes,
        resample_method=resample_method,
        window_type=window_type,
        window_size=window_size,
        window_duration_sec=window_duration_sec,
        stride_or_overlap_mode=stride_or_overlap_mode,
        step_value=step_value,
        stride_or_overlap=stride_or_overlap,
        overlap=overlap,
        stride=stride,
        stride_sec=stride_sec,
        normalization=normalization,
        use_log_power=use_log_power,
        drop_dc=drop_dc,
    )

    dataset_bundle = build_frequency_dataset(
        real_features=capture_result["real_frequency_bundle"]["features"],
        synthetic_features=capture_result["synthetic_frequency_bundle"]["features"],
    )
    embedding = fit_tsne_embedding(
        dataset_bundle["features"],
        perplexity=perplexity,
        init=init,
        random_state=random_state,
    )
    embedding_df = build_embedding_frame(
        embedding,
        dataset_bundle["labels"],
        window_size=int(capture_result["window_size"]),
    )

    figure, axis = plt.subplots(figsize=figsize)
    auto_title = (
        f"2D t-SNE | sensors={', '.join(capture_result['selected_bundle']['selected_sensors'])} | "
        f"groups={', '.join(capture_result['selected_bundle']['signal_groups'])} | "
        f"window={int(capture_result['window_size'])} samples"
    )
    if capture_result["window_duration_sec"] is not None:
        auto_title += f" ({float(capture_result['window_duration_sec']):.3f} s)"
    plot_tsne_embedding(embedding_df, ax=axis, title=title or auto_title)
    figure.tight_layout()
    if show:
        plt.show()

    return {
        "capture_row": capture_result["capture_row"],
        "selected_summary_df": capture_result["selected_summary_df"],
        "alignment_summary_df": capture_result["alignment_summary_df"],
        "window_summary_df": capture_result["window_summary_df"],
        "embedding_df": embedding_df,
        "feature_matrix": dataset_bundle["features"],
        "labels": dataset_bundle["labels"],
        "frequencies_hz": capture_result["frequencies_hz"],
        "figure": figure,
        "axis": axis,
        "selected_bundle": capture_result["selected_bundle"],
        "aligned_bundle": capture_result["aligned_bundle"],
    }
