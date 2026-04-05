from __future__ import annotations

from fractions import Fraction
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
            alpha=0.8,
            s=40,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title or "2D t-SNE in the frequency domain")
    ax.legend()
    return ax


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

    dataset_bundle = build_frequency_dataset(
        real_features=real_frequency_bundle["features"],
        synthetic_features=synthetic_frequency_bundle["features"],
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
        window_size=resolved_window_size,
    )

    figure, axis = plt.subplots(figsize=figsize)
    auto_title = (
        f"2D t-SNE | sensors={', '.join(selected_bundle['selected_sensors'])} | "
        f"groups={', '.join(selected_bundle['signal_groups'])} | "
        f"window={resolved_window_size} samples"
    )
    if resolved_window_duration_sec is not None:
        auto_title += f" ({resolved_window_duration_sec:.3f} s)"
    plot_tsne_embedding(embedding_df, ax=axis, title=title or auto_title)
    figure.tight_layout()
    if show:
        plt.show()

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
        feature_dim=dataset_bundle["features"].shape[1],
    )

    return {
        "capture_row": selected_bundle["capture_row"],
        "selected_summary_df": selected_summary_df,
        "alignment_summary_df": alignment_summary_df,
        "window_summary_df": window_summary_df,
        "embedding_df": embedding_df,
        "feature_matrix": dataset_bundle["features"],
        "labels": dataset_bundle["labels"],
        "frequencies_hz": real_frequency_bundle["frequencies_hz"],
        "figure": figure,
        "axis": axis,
        "selected_bundle": selected_bundle,
        "aligned_bundle": aligned_bundle,
    }


def compare_window_sizes(
    captures_df: pd.DataFrame,
    *,
    domain: str,
    user_id: int,
    tag_number: int,
    window_type: str | None = None,
    window_sizes: Sequence[int] | None = None,
    window_durations_sec: Sequence[float] | None = None,
    take_id: str | None = None,
    signal_groups: Sequence[str] = ("acc",),
    selected_sensors: Sequence[str] | None = None,
    selected_axes: Sequence[str] = ("x", "y", "z"),
    synthetic_filename: str = "virtual_imu.npz",
    resample_method: str = "resample_poly",
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
    figsize: tuple[float, float] = (6.5, 5.2),
    show: bool = True,
) -> dict[str, Any]:
    if window_sizes is not None and window_durations_sec is not None:
        raise ValueError("Use either window_sizes or window_durations_sec, not both.")
    if window_sizes is None and window_durations_sec is None:
        raise ValueError("You must provide window_sizes or window_durations_sec.")
    if window_sizes is not None and len(window_sizes) == 0:
        raise ValueError("window_sizes cannot be empty.")
    if window_durations_sec is not None and len(window_durations_sec) == 0:
        raise ValueError("window_durations_sec cannot be empty.")

    comparison_values = window_sizes if window_sizes is not None else window_durations_sec
    comparison_mode = "samples" if window_sizes is not None else "seconds"

    results_by_window_size: dict[int, dict[str, Any]] = {}
    comparison_rows: list[dict[str, Any]] = []
    num_columns = len(comparison_values)
    comparison_figure, comparison_axes = plt.subplots(1, num_columns, figsize=(figsize[0] * num_columns, figsize[1]))

    if num_columns == 1:
        comparison_axes = np.array([comparison_axes])

    for axis, current_value in zip(comparison_axes, comparison_values):
        result = run_frequency_domain_tsne(
            captures_df,
            domain=domain,
            user_id=user_id,
            tag_number=tag_number,
            take_id=take_id,
            signal_groups=signal_groups,
            selected_sensors=selected_sensors,
            selected_axes=selected_axes,
            synthetic_filename=synthetic_filename,
            resample_method=resample_method,
            window_type=("n_samples" if comparison_mode == "samples" else "seconds") if window_type is None else window_type,
            window_size=(int(current_value) if comparison_mode == "samples" else None),
            window_duration_sec=(float(current_value) if comparison_mode == "seconds" else None),
            stride_or_overlap_mode=stride_or_overlap_mode,
            step_value=step_value,
            stride_or_overlap=stride_or_overlap,
            overlap=overlap,
            stride=stride,
            stride_sec=stride_sec,
            normalization=normalization,
            perplexity=perplexity,
            init=init,
            random_state=random_state,
            use_log_power=use_log_power,
            drop_dc=drop_dc,
            show=False,
        )

        plot_tsne_embedding(
            result["embedding_df"],
            ax=axis,
            title=(
                f"Window = {int(current_value)} samples"
                if comparison_mode == "samples"
                else f"Window = {float(current_value):.3f} s"
            ),
        )
        plt.close(result["figure"])

        summary_row = {
            **result["window_summary_df"].iloc[0].to_dict(),
            **result["alignment_summary_df"].iloc[0].to_dict(),
        }
        comparison_rows.append(summary_row)
        results_by_window_size[int(round(float(current_value) * 1000)) if comparison_mode == "seconds" else int(current_value)] = result

    comparison_figure.suptitle(
        "t-SNE comparison by window duration" if comparison_mode == "seconds" else "t-SNE comparison by window size",
        y=1.02,
    )
    comparison_figure.tight_layout()
    if show:
        plt.show()

    return {
        "comparison_df": pd.DataFrame(comparison_rows),
        "figure": comparison_figure,
        "axes": comparison_axes,
        "results_by_window_size": results_by_window_size,
    }
