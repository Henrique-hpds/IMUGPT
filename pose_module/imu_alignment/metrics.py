"""Metrics used to compare real and aligned IMU signals."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .interfaces import AlignmentResult

_AXIS_NAMES = ("x", "y", "z")
_EPSILON = 1e-8


def compute_axiswise_rmse(reference_xyz: np.ndarray, estimate_xyz: np.ndarray) -> dict[str, float]:
    """Compute per-axis RMSE for two aligned vector streams."""

    reference, estimate = _validate_pair(reference_xyz, estimate_xyz)
    delta = reference - estimate
    return {
        axis_name: float(np.sqrt(np.mean(np.square(delta[:, axis_index]))))
        for axis_index, axis_name in enumerate(_AXIS_NAMES)
    }


def compute_axiswise_corr(reference_xyz: np.ndarray, estimate_xyz: np.ndarray) -> dict[str, float | None]:
    """Compute per-axis Pearson correlation for two aligned vector streams."""

    reference, estimate = _validate_pair(reference_xyz, estimate_xyz)
    output: dict[str, float | None] = {}
    for axis_index, axis_name in enumerate(_AXIS_NAMES):
        output[axis_name] = _pearson_or_none(reference[:, axis_index], estimate[:, axis_index])
    return output


def compute_vector_angle_error(reference_xyz: np.ndarray, estimate_xyz: np.ndarray) -> dict[str, float | None]:
    """Compute mean and median angular error in degrees."""

    reference, estimate = _validate_pair(reference_xyz, estimate_xyz)
    reference_norm = np.linalg.norm(reference, axis=1)
    estimate_norm = np.linalg.norm(estimate, axis=1)
    valid_mask = (reference_norm > _EPSILON) & (estimate_norm > _EPSILON)
    if int(np.count_nonzero(valid_mask)) == 0:
        return {"mean_deg": None, "median_deg": None}
    cosine = np.sum(reference[valid_mask] * estimate[valid_mask], axis=1) / (
        reference_norm[valid_mask] * estimate_norm[valid_mask]
    )
    cosine = np.clip(cosine, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(cosine))
    return {
        "mean_deg": float(np.mean(angles_deg)),
        "median_deg": float(np.median(angles_deg)),
    }


def compute_norm_error(reference_xyz: np.ndarray, estimate_xyz: np.ndarray) -> dict[str, float]:
    """Compute mean absolute and RMSE norm errors."""

    reference, estimate = _validate_pair(reference_xyz, estimate_xyz)
    reference_norm = np.linalg.norm(reference, axis=1)
    estimate_norm = np.linalg.norm(estimate, axis=1)
    delta = reference_norm - estimate_norm
    return {
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
    }


def summarize_alignment_metrics(
    *,
    real_acc: np.ndarray | None = None,
    estimate_acc: np.ndarray | None = None,
    real_gyro: np.ndarray | None = None,
    estimate_gyro: np.ndarray | None = None
) -> dict[str, Any]:
    """Summarize per-modality alignment metrics in a JSON-friendly structure."""

    summary: dict[str, Any] = {
        "num_samples": 0,
        "modalities": {},
    }
    for modality_name, reference, estimate in (
        ("acc", real_acc, estimate_acc),
        ("gyro", real_gyro, estimate_gyro),
    ):
        if reference is None or estimate is None:
            continue
        reference_array, estimate_array = _validate_pair(reference, estimate)
        summary["num_samples"] = max(int(summary["num_samples"]), int(reference_array.shape[0]))
        summary["modalities"][modality_name] = {
            "rmse_per_axis": compute_axiswise_rmse(reference_array, estimate_array),
            "corr_per_axis": compute_axiswise_corr(reference_array, estimate_array),
            "norm_error": compute_norm_error(reference_array, estimate_array),
            "angle_error_deg": compute_vector_angle_error(reference_array, estimate_array),
        }
    return summary


def aggregate_alignment_results(results: Sequence[AlignmentResult]) -> dict[str, Any]:
    """Aggregate per-sensor results into capture-level summary metrics."""

    per_sensor = []
    before_gyro_corr = []
    after_gyro_corr = []
    before_acc_corr = []
    after_acc_corr = []
    for result in results:
        before_modalities = dict(result.metrics_before.get("modalities", {}))
        after_modalities = dict(result.metrics_after.get("modalities", {}))
        per_sensor.append(
            {
                "subject_id": result.subject_id,
                "capture_id": result.capture_id,
                "sensor_name": result.sensor_name,
                "lag_samples": int(result.lag_samples),
                "metrics_before": dict(result.metrics_before),
                "metrics_after": dict(result.metrics_after),
            }
        )
        before_acc_corr.extend(_finite_metric_values(before_modalities.get("acc", {}), "corr_per_axis"))
        before_gyro_corr.extend(_finite_metric_values(before_modalities.get("gyro", {}), "corr_per_axis"))
        after_acc_corr.extend(_finite_metric_values(after_modalities.get("acc", {}), "corr_per_axis"))
        after_gyro_corr.extend(_finite_metric_values(after_modalities.get("gyro", {}), "corr_per_axis"))

    return {
        "num_results": int(len(results)),
        "per_sensor": per_sensor,
        "mean_acc_corr_before": _mean_or_none(before_acc_corr),
        "mean_acc_corr_after": _mean_or_none(after_acc_corr),
        "mean_gyro_corr_before": _mean_or_none(before_gyro_corr),
        "mean_gyro_corr_after": _mean_or_none(after_gyro_corr),
    }


def _validate_pair(reference_xyz: np.ndarray, estimate_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference_xyz, dtype=np.float64)
    estimate = np.asarray(estimate_xyz, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference_xyz must have shape [T, 3].")
    if estimate.ndim != 2 or estimate.shape[1] != 3:
        raise ValueError("estimate_xyz must have shape [T, 3].")
    if reference.shape != estimate.shape:
        raise ValueError("reference_xyz and estimate_xyz must share the same shape.")
    finite_mask = np.isfinite(reference).all(axis=1) & np.isfinite(estimate).all(axis=1)
    filtered_reference = reference[finite_mask]
    filtered_estimate = estimate[finite_mask]
    if filtered_reference.shape[0] == 0:
        raise ValueError("reference_xyz and estimate_xyz do not share any finite samples.")
    return filtered_reference, filtered_estimate


def _pearson_or_none(x: np.ndarray, y: np.ndarray) -> float | None:
    x_centered = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    y_centered = np.asarray(y, dtype=np.float64) - float(np.mean(y))
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator <= 0.0:
        return None
    return float(np.dot(x_centered, y_centered) / denominator)


def _finite_metric_values(modality_metrics: Mapping[str, Any], key: str) -> list[float]:
    values = modality_metrics.get(key, {})
    if not isinstance(values, Mapping):
        return []
    finite_values = []
    for value in values.values():
        if value is None:
            continue
        value_float = float(value)
        if np.isfinite(value_float):
            finite_values.append(value_float)
    return finite_values


def _mean_or_none(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return None
    return float(np.mean(array))
