"""Virtual-to-real IMU calibration using the percentile mapping from the paper code."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from pose_module.interfaces import VirtualIMUSequence

DEFAULT_CALIBRATION_SIGNAL_MODE = "acc"
DEFAULT_CALIBRATION_PERCENTILE_RESOLUTION = 100
_SUPPORTED_SIGNAL_MODES = {"acc", "gyro", "both"}


def calibrate_virtual_imu_sequence(
    imu_sequence: VirtualIMUSequence,
    *,
    real_imu_reference_path: str | Path,
    activity_label: Any = None,
    signal_mode: str = DEFAULT_CALIBRATION_SIGNAL_MODE,
    percentile_resolution: int = DEFAULT_CALIBRATION_PERCENTILE_RESOLUTION,
    per_class: bool = True,
    fallback_to_global: bool = True,
) -> Dict[str, Any]:
    """Calibrate a virtual IMU sequence against a real IMU reference set."""

    resolved_signal_mode = _normalize_signal_mode(signal_mode)
    reference_path = Path(real_imu_reference_path)
    if not reference_path.exists():
        raise FileNotFoundError(f"Real IMU reference not found: {reference_path}")

    with np.load(reference_path, allow_pickle=True) as payload:
        reference = _extract_reference_signal(
            payload=payload,
            target_sensor_names=imu_sequence.sensor_names,
            signal_mode=resolved_signal_mode,
        )

    matched_sensor_indices = [
        int(index)
        for index, sensor_name in enumerate(imu_sequence.sensor_names)
        if str(sensor_name) in set(reference["matched_sensor_names"])
    ]
    target_signal, target_indices = _extract_target_signal(
        imu_sequence,
        signal_mode=resolved_signal_mode,
        sensor_indices=matched_sensor_indices,
    )
    selected_target = np.asarray(target_signal[:, target_indices, :], dtype=np.float32)
    virtual_matrix = selected_target.reshape(int(selected_target.shape[0]), -1).astype(np.float64, copy=False)
    real_matrix = np.asarray(reference["matrix"], dtype=np.float64)
    if virtual_matrix.shape[1] != real_matrix.shape[1]:
        raise ValueError(
            "Real IMU reference channel count does not match the selected virtual IMU channels: "
            f"{real_matrix.shape[1]} != {virtual_matrix.shape[1]}"
        )

    calibration = _percentile_map_virtual_to_real(
        virtual_matrix=virtual_matrix,
        real_matrix=real_matrix,
        virtual_labels=None if activity_label is None else np.full((virtual_matrix.shape[0],), activity_label, dtype=object),
        real_labels=reference.get("labels"),
        percentile_resolution=int(percentile_resolution),
        per_class=bool(per_class),
        fallback_to_global=bool(fallback_to_global),
        activity_label=activity_label,
    )

    transformed_matrix = np.asarray(calibration["transformed_matrix"], dtype=np.float32)
    transformed_signal = transformed_matrix.reshape(selected_target.shape).astype(np.float32, copy=False)
    calibrated_signal = np.asarray(target_signal, dtype=np.float32).copy()
    calibrated_signal[:, target_indices, :] = transformed_signal

    calibrated_sequence = _build_calibrated_sequence(
        imu_sequence=imu_sequence,
        signal_mode=resolved_signal_mode,
        calibrated_signal=calibrated_signal,
    )

    delta = np.abs(transformed_matrix - virtual_matrix.astype(np.float32, copy=False))
    report_notes = []
    report_notes.extend([str(value) for value in reference.get("notes", [])])
    report_notes.extend([str(value) for value in calibration.get("notes", [])])

    status = "ok"
    if len(reference.get("missing_sensor_names", [])) > 0 or calibration.get("status") == "warning":
        status = "warning"

    return {
        "virtual_imu_sequence": calibrated_sequence,
        "calibration_report": {
            "clip_id": str(imu_sequence.clip_id),
            "status": str(status),
            "calibration_applied": True,
            "reference_path": str(reference_path.resolve()),
            "signal_mode": str(resolved_signal_mode),
            "percentile_resolution": int(percentile_resolution),
            "per_class_requested": bool(per_class),
            "per_class_applied": bool(calibration.get("per_class_applied", False)),
            "fallback_to_global": bool(fallback_to_global),
            "activity_label": None if activity_label is None else str(activity_label),
            "reference_num_rows": int(real_matrix.shape[0]),
            "reference_num_channels": int(real_matrix.shape[1]),
            "matched_sensor_names": [str(name) for name in reference["matched_sensor_names"]],
            "missing_sensor_names": [str(name) for name in reference["missing_sensor_names"]],
            "matched_sensor_count": int(len(reference["matched_sensor_names"])),
            "target_num_rows": int(virtual_matrix.shape[0]),
            "target_num_channels": int(virtual_matrix.shape[1]),
            "mean_abs_delta": float(np.mean(delta)) if delta.size > 0 else 0.0,
            "max_abs_delta": float(np.max(delta)) if delta.size > 0 else 0.0,
            "assumptions": [
                "percentile_mapping_matches_article_code",
                "reference_and_virtual_sensor_layouts_are_compatible",
                "calibration_is_applied_per_channel_independently",
            ],
            "limitations": [
                "mapping_is_piecewise_constant_not_interpolated",
                "calibration_preserves_no_cross_channel_correlations",
                "unmatched_sensors_remain_uncalibrated",
            ],
            "notes": list(dict.fromkeys(report_notes)),
        },
        "matched_sensor_indices": [int(index) for index in matched_sensor_indices],
        "matched_sensor_names": [str(name) for name in reference["matched_sensor_names"]],
    }


def _normalize_signal_mode(signal_mode: str) -> str:
    mode = str(signal_mode).strip().lower()
    if mode not in _SUPPORTED_SIGNAL_MODES:
        raise ValueError(f"Unsupported signal_mode '{signal_mode}'. Expected one of {_SUPPORTED_SIGNAL_MODES}.")
    return mode


def _extract_reference_signal(
    *,
    payload: Mapping[str, Any],
    target_sensor_names: Sequence[str],
    signal_mode: str,
) -> Dict[str, Any]:
    labels = _extract_reference_labels(payload)
    sensor_names = _extract_optional_sensor_names(payload)
    notes: list[str] = []

    if "imu" in payload:
        imu = np.asarray(payload["imu"], dtype=np.float32)
        if imu.ndim != 3 or imu.shape[-1] < 6:
            raise ValueError("Expected payload['imu'] with shape [T, S, 6].")
        signal = _select_signal_channels(imu, signal_mode=signal_mode)
        matrix, matched_names, missing_names = _structured_reference_to_matrix(
            signal=signal,
            sensor_names=sensor_names,
            target_sensor_names=target_sensor_names,
        )
    elif "acc" in payload or "gyro" in payload:
        signal = _combine_acc_gyro_payload(payload, signal_mode=signal_mode)
        matrix, matched_names, missing_names = _structured_reference_to_matrix(
            signal=signal,
            sensor_names=sensor_names,
            target_sensor_names=target_sensor_names,
        )
    elif "imu_flat" in payload:
        imu_flat = np.asarray(payload["imu_flat"], dtype=np.float32)
        matrix, matched_names, missing_names = _flat_reference_to_matrix(
            flat=imu_flat,
            sensor_names=sensor_names,
            target_sensor_names=target_sensor_names,
            signal_mode=signal_mode,
        )
    elif "x" in payload or "X" in payload:
        key = "x" if "x" in payload else "X"
        matrix = np.asarray(payload[key], dtype=np.float32)
        matched_names, missing_names = _validate_legacy_reference_shape(
            matrix=matrix,
            target_sensor_names=target_sensor_names,
            signal_mode=signal_mode,
        )
        notes.append(f"legacy_reference_matrix_key:{key}")
    else:
        raise ValueError(
            "Unsupported real IMU reference payload. Expected one of: imu, acc/gyro, imu_flat, x, or X."
        )

    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Real IMU reference matrix must have shape [N, C] with N > 0 and C > 0.")
    if not np.isfinite(matrix).all():
        raise ValueError("Real IMU reference contains NaN or infinite values.")
    if len(matched_names) == 0:
        raise ValueError("No overlapping sensors found between the virtual IMU output and the real IMU reference.")
    if len(missing_names) > 0:
        notes.append("reference_missing_virtual_sensors")

    return {
        "matrix": matrix.astype(np.float32, copy=False),
        "labels": labels,
        "matched_sensor_names": [str(name) for name in matched_names],
        "missing_sensor_names": [str(name) for name in missing_names],
        "notes": list(dict.fromkeys(notes)),
    }


def _extract_reference_labels(payload: Mapping[str, Any]) -> np.ndarray | None:
    for key in ("y", "labels", "activity_labels", "activity_label", "label"):
        if key not in payload:
            continue
        labels = np.asarray(payload[key])
        if labels.ndim == 0:
            return None
        return labels.reshape(-1)
    return None


def _extract_optional_sensor_names(payload: Mapping[str, Any]) -> list[str] | None:
    for key in ("sensor_names", "sensors"):
        if key not in payload:
            continue
        values = np.asarray(payload[key])
        if values.ndim == 0:
            continue
        return [str(value) for value in values.reshape(-1).tolist()]
    return None


def _combine_acc_gyro_payload(payload: Mapping[str, Any], *, signal_mode: str) -> np.ndarray:
    acc = None if "acc" not in payload else np.asarray(payload["acc"], dtype=np.float32)
    gyro = None if "gyro" not in payload else np.asarray(payload["gyro"], dtype=np.float32)
    if signal_mode == "acc":
        if acc is None:
            raise ValueError("Real IMU reference missing accelerometer signal.")
        return acc
    if signal_mode == "gyro":
        if gyro is None:
            raise ValueError("Real IMU reference missing gyroscope signal.")
        return gyro
    if acc is None or gyro is None:
        raise ValueError("Real IMU reference must include both acc and gyro for signal_mode='both'.")
    if acc.shape[:2] != gyro.shape[:2]:
        raise ValueError("Real IMU reference acc/gyro shapes must match on [T, S].")
    return np.concatenate([acc, gyro], axis=2).astype(np.float32, copy=False)


def _select_signal_channels(signal: np.ndarray, *, signal_mode: str) -> np.ndarray:
    if signal_mode == "acc":
        return np.asarray(signal[..., :3], dtype=np.float32)
    if signal_mode == "gyro":
        return np.asarray(signal[..., 3:6], dtype=np.float32)
    return np.asarray(signal[..., :6], dtype=np.float32)


def _structured_reference_to_matrix(
    *,
    signal: np.ndarray,
    sensor_names: Sequence[str] | None,
    target_sensor_names: Sequence[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    array = np.asarray(signal, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("Structured real IMU reference must have shape [T, S, C].")

    if sensor_names is None:
        if int(array.shape[1]) != int(len(target_sensor_names)):
            raise ValueError(
                "Real IMU reference does not define sensor names and sensor count differs from the virtual IMU output."
            )
        matched_names = [str(name) for name in target_sensor_names]
        return array.reshape(int(array.shape[0]), -1), matched_names, []

    name_to_index = {str(name): index for index, name in enumerate(sensor_names)}
    matched_target_indices: list[int] = []
    matched_reference_indices: list[int] = []
    missing_names: list[str] = []
    matched_names: list[str] = []
    for target_index, target_name in enumerate(target_sensor_names):
        key = str(target_name)
        if key not in name_to_index:
            missing_names.append(key)
            continue
        matched_target_indices.append(int(target_index))
        matched_reference_indices.append(int(name_to_index[key]))
        matched_names.append(key)
    if len(matched_reference_indices) == 0:
        raise ValueError("Real IMU reference sensor_names do not overlap with the virtual IMU sensor layout.")
    selected = np.asarray(array[:, matched_reference_indices, :], dtype=np.float32)
    return selected.reshape(int(selected.shape[0]), -1), matched_names, missing_names


def _flat_reference_to_matrix(
    *,
    flat: np.ndarray,
    sensor_names: Sequence[str] | None,
    target_sensor_names: Sequence[str],
    signal_mode: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    array = np.asarray(flat, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Flat real IMU reference must have shape [T, C].")

    num_channels_per_sensor = 6
    if array.shape[1] % num_channels_per_sensor != 0:
        matched_names, missing_names = _validate_legacy_reference_shape(
            matrix=array,
            target_sensor_names=target_sensor_names,
            signal_mode=signal_mode,
        )
        return array, matched_names, missing_names

    num_sensors = int(array.shape[1] // num_channels_per_sensor)
    reshaped = array.reshape(int(array.shape[0]), num_sensors, num_channels_per_sensor)
    selected = _select_signal_channels(reshaped, signal_mode=signal_mode)
    return _structured_reference_to_matrix(
        signal=selected,
        sensor_names=sensor_names,
        target_sensor_names=target_sensor_names,
    )


def _validate_legacy_reference_shape(
    *,
    matrix: np.ndarray,
    target_sensor_names: Sequence[str],
    signal_mode: str,
) -> tuple[list[str], list[str]]:
    channels_per_sensor = 6 if signal_mode == "both" else 3
    expected_channels = int(len(target_sensor_names)) * channels_per_sensor
    if int(matrix.shape[1]) != expected_channels:
        raise ValueError(
            "Legacy real IMU reference matrix does not match the virtual IMU channel count: "
            f"{matrix.shape[1]} != {expected_channels}"
        )
    return [str(name) for name in target_sensor_names], []


def _extract_target_signal(
    imu_sequence: VirtualIMUSequence,
    *,
    signal_mode: str,
    sensor_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, list[int]]:
    if signal_mode == "acc":
        signal = np.asarray(imu_sequence.acc, dtype=np.float32)
    elif signal_mode == "gyro":
        signal = np.asarray(imu_sequence.gyro, dtype=np.float32)
    else:
        signal = np.concatenate(
            [
                np.asarray(imu_sequence.acc, dtype=np.float32),
                np.asarray(imu_sequence.gyro, dtype=np.float32),
            ],
            axis=2,
        ).astype(np.float32, copy=False)
    if sensor_indices is None:
        indices = list(range(int(signal.shape[1])))
    else:
        indices = [int(index) for index in sensor_indices]
    return signal, indices


def _build_calibrated_sequence(
    *,
    imu_sequence: VirtualIMUSequence,
    signal_mode: str,
    calibrated_signal: np.ndarray,
) -> VirtualIMUSequence:
    if signal_mode == "acc":
        acc = np.asarray(calibrated_signal, dtype=np.float32)
        gyro = np.asarray(imu_sequence.gyro, dtype=np.float32)
    elif signal_mode == "gyro":
        acc = np.asarray(imu_sequence.acc, dtype=np.float32)
        gyro = np.asarray(calibrated_signal, dtype=np.float32)
    else:
        acc = np.asarray(calibrated_signal[..., :3], dtype=np.float32)
        gyro = np.asarray(calibrated_signal[..., 3:6], dtype=np.float32)
    return VirtualIMUSequence(
        clip_id=str(imu_sequence.clip_id),
        fps=None if imu_sequence.fps is None else float(imu_sequence.fps),
        sensor_names=list(imu_sequence.sensor_names),
        acc=acc.astype(np.float32, copy=False),
        gyro=gyro.astype(np.float32, copy=False),
        timestamps_sec=np.asarray(imu_sequence.timestamps_sec, dtype=np.float32),
        source=f"{imu_sequence.source}_real_calibrated",
    )


def _percentile_map_virtual_to_real(
    *,
    virtual_matrix: np.ndarray,
    real_matrix: np.ndarray,
    virtual_labels: np.ndarray | None,
    real_labels: np.ndarray | None,
    percentile_resolution: int,
    per_class: bool,
    fallback_to_global: bool,
    activity_label: Any,
) -> Dict[str, Any]:
    if virtual_matrix.ndim != 2 or real_matrix.ndim != 2:
        raise ValueError("Percentile calibration expects matrices with shape [N, C].")
    if virtual_matrix.shape[1] != real_matrix.shape[1]:
        raise ValueError("Virtual and real IMU matrices must have the same channel count.")
    if not np.isfinite(virtual_matrix).all():
        raise ValueError("Virtual IMU matrix contains NaN or infinite values before calibration.")
    if not np.isfinite(real_matrix).all():
        raise ValueError("Real IMU matrix contains NaN or infinite values.")
    if int(percentile_resolution) < 2:
        raise ValueError("percentile_resolution must be at least 2.")

    notes: list[str] = []
    pcts = np.linspace(0.0, 100.0, int(percentile_resolution), dtype=np.float64)
    transformed = np.full(virtual_matrix.shape, np.nan, dtype=np.float64)
    per_class_applied = False

    if per_class and activity_label is not None and real_labels is not None:
        real_mask = _build_label_mask(real_labels, activity_label)
        if np.any(real_mask):
            transformed = _map_percentiles(
                source=virtual_matrix,
                target=real_matrix[real_mask],
                pcts=pcts,
            )
            per_class_applied = True
        elif fallback_to_global:
            notes.append(f"calibration_label_missing_in_reference_fallback_global:{activity_label}")
        else:
            raise ValueError(f"Activity label '{activity_label}' not found in the real IMU reference labels.")
    elif per_class and activity_label is None:
        notes.append("calibration_per_class_requested_without_activity_label_fallback_global")
    elif per_class and real_labels is None:
        notes.append("calibration_per_class_requested_without_reference_labels_fallback_global")

    if not np.isfinite(transformed).all():
        transformed = _map_percentiles(
            source=virtual_matrix,
            target=real_matrix,
            pcts=pcts,
        )

    return {
        "status": "warning" if len(notes) > 0 else "ok",
        "transformed_matrix": transformed.astype(np.float32, copy=False),
        "per_class_applied": bool(per_class_applied),
        "notes": list(dict.fromkeys(notes)),
    }


def _build_label_mask(labels: np.ndarray, activity_label: Any) -> np.ndarray:
    array = np.asarray(labels).reshape(-1)
    if array.size == 0:
        return np.zeros((0,), dtype=bool)
    if np.issubdtype(array.dtype, np.number):
        try:
            numeric_label = float(activity_label)
        except (TypeError, ValueError):
            return np.asarray([str(value) == str(activity_label) for value in array], dtype=bool)
        return np.isclose(array.astype(np.float64, copy=False), numeric_label)
    return np.asarray([str(value) == str(activity_label) for value in array], dtype=bool)


def _map_percentiles(*, source: np.ndarray, target: np.ndarray, pcts: np.ndarray) -> np.ndarray:
    source_percentiles = np.percentile(source, pcts, axis=0)
    target_percentiles = np.percentile(target, pcts, axis=0)
    transformed = np.empty_like(source, dtype=np.float64)
    for channel_index in range(int(source.shape[1])):
        bins = np.asarray(source_percentiles[:, channel_index], dtype=np.float64)
        values = np.asarray(target_percentiles[:, channel_index], dtype=np.float64)
        indices = np.searchsorted(bins, source[:, channel_index], side="left")
        indices = np.clip(indices, 0, len(values) - 1)
        transformed[:, channel_index] = values[indices]
    return transformed
