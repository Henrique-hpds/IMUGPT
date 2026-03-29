"""Diagnostic estimation of per-clip sensor-frame alignment against real IMU data."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from pose_module.interfaces import VirtualIMUSequence
from pose_module.io.cache import write_json_file
from pose_module.robot_emotions.metadata import resolve_sensor_names

DEFAULT_TARGET_SENSOR_NAMES = ("left_forearm", "right_forearm")
DEFAULT_MAX_LAG_SEC = 2.5
DEFAULT_DYNAMIC_PERCENTILE = 70.0
DEFAULT_STATIC_PERCENTILE = 30.0
DEFAULT_MIN_VALID_FRAMES = 100
DEFAULT_MIN_DYNAMIC_GYRO_NORM = 0.05
DEFAULT_FRAME_ALIGNED_IMU_FILENAME = "virtual_imu_frame_aligned.npz"
DEFAULT_FRAME_ESTIMATION_REPORT_FILENAME = "sensor_frame_estimation_report.json"
DEFAULT_FRAME_ALIGNMENT_QUALITY_REPORT_FILENAME = "frame_alignment_quality_report.json"
_EPSILON = 1e-8


def estimate_sensor_frame_alignment(
    raw_virtual_imu_sequence: VirtualIMUSequence,
    *,
    real_imu_npz_path: str | Path | None,
    target_sensor_names: Sequence[str] = DEFAULT_TARGET_SENSOR_NAMES,
    output_dir: str | Path | None = None,
    aligned_imu_sequence_filename: str = DEFAULT_FRAME_ALIGNED_IMU_FILENAME,
    report_filename: str = DEFAULT_FRAME_ESTIMATION_REPORT_FILENAME,
    quality_report_filename: str = DEFAULT_FRAME_ALIGNMENT_QUALITY_REPORT_FILENAME,
    max_lag_sec: float = DEFAULT_MAX_LAG_SEC,
    dynamic_percentile: float = DEFAULT_DYNAMIC_PERCENTILE,
    static_percentile: float = DEFAULT_STATIC_PERCENTILE,
    min_valid_frames: int = DEFAULT_MIN_VALID_FRAMES,
    min_dynamic_gyro_norm: float = DEFAULT_MIN_DYNAMIC_GYRO_NORM,
) -> Dict[str, Any]:
    """Estimate a fixed sensor frame per target sensor for a single clip."""

    target_sensor_names = tuple(str(name) for name in target_sensor_names)
    raw_acc = np.asarray(raw_virtual_imu_sequence.acc, dtype=np.float32)
    raw_gyro = np.asarray(raw_virtual_imu_sequence.gyro, dtype=np.float32)
    aligned_acc = raw_acc.copy()
    aligned_gyro = raw_gyro.copy()
    notes: list[str] = [
        "frame_alignment_is_diagnostic_only",
        "frame_alignment_uses_raw_virtual_imu_before_percentile_calibration",
        "frame_alignment_assumes_constant_per_sensor_rotation_over_clip",
    ]

    resolved_real_path = None if real_imu_npz_path in (None, "") else Path(real_imu_npz_path)
    real_clip = None
    if resolved_real_path is None:
        notes.append("real_imu_npz_path_missing")
    elif not resolved_real_path.exists():
        notes.append(f"real_imu_npz_path_not_found:{resolved_real_path}")
    else:
        real_clip = _load_real_imu_clip(resolved_real_path)

    sensor_reports: dict[str, Dict[str, Any]] = {}
    lag_sensor_reports: dict[str, Dict[str, Any]] = {}

    for sensor_name in target_sensor_names:
        sensor_report = _build_default_sensor_report(sensor_name)
        if real_clip is None:
            sensor_report["notes"].append("real_imu_clip_unavailable")
            sensor_reports[sensor_name] = sensor_report
            lag_sensor_reports[sensor_name] = _build_lag_sensor_report(sensor_report)
            continue

        virtual_index = _find_sensor_index(raw_virtual_imu_sequence.sensor_names, sensor_name)
        real_index = _find_sensor_index(real_clip["sensor_names"], sensor_name)
        if virtual_index is None:
            sensor_report["notes"].append("sensor_missing_in_virtual_clip")
            sensor_reports[sensor_name] = sensor_report
            lag_sensor_reports[sensor_name] = _build_lag_sensor_report(sensor_report)
            continue
        if real_index is None:
            sensor_report["notes"].append("sensor_missing_in_real_clip")
            sensor_reports[sensor_name] = sensor_report
            lag_sensor_reports[sensor_name] = _build_lag_sensor_report(sensor_report)
            continue

        per_sensor_result = _estimate_single_sensor_frame(
            sensor_name=sensor_name,
            virtual_timestamps=np.asarray(raw_virtual_imu_sequence.timestamps_sec, dtype=np.float32),
            virtual_acc=raw_acc[:, virtual_index, :],
            virtual_gyro=raw_gyro[:, virtual_index, :],
            real_timestamps=np.asarray(real_clip["timestamps_sec"], dtype=np.float32),
            real_acc=np.asarray(real_clip["acc"], dtype=np.float32)[:, real_index, :],
            real_gyro=np.asarray(real_clip["gyro"], dtype=np.float32)[:, real_index, :],
            max_lag_sec=float(max_lag_sec),
            dynamic_percentile=float(dynamic_percentile),
            static_percentile=float(static_percentile),
            min_valid_frames=int(min_valid_frames),
            min_dynamic_gyro_norm=float(min_dynamic_gyro_norm),
        )
        sensor_report.update(per_sensor_result["sensor_report"])
        if sensor_report["rotation_matrix"] is not None:
            rotation_matrix = np.asarray(sensor_report["rotation_matrix"], dtype=np.float32)
            aligned_acc[:, virtual_index, :] = _apply_rotation(raw_acc[:, virtual_index, :], rotation_matrix)
            aligned_gyro[:, virtual_index, :] = _apply_rotation(raw_gyro[:, virtual_index, :], rotation_matrix)
        sensor_reports[sensor_name] = sensor_report
        lag_sensor_reports[sensor_name] = dict(per_sensor_result["lag_sensor_report"])

    overall_status = _merge_sensor_status(sensor_reports.values())
    aligned_virtual_imu_sequence = VirtualIMUSequence(
        clip_id=str(raw_virtual_imu_sequence.clip_id),
        fps=None if raw_virtual_imu_sequence.fps is None else float(raw_virtual_imu_sequence.fps),
        sensor_names=list(raw_virtual_imu_sequence.sensor_names),
        acc=aligned_acc.astype(np.float32, copy=False),
        gyro=aligned_gyro.astype(np.float32, copy=False),
        timestamps_sec=np.asarray(raw_virtual_imu_sequence.timestamps_sec, dtype=np.float32),
        source=f"{raw_virtual_imu_sequence.source}_frame_aligned",
    )
    quality_report = _build_frame_alignment_quality_report(
        clip_id=str(raw_virtual_imu_sequence.clip_id),
        status=str(overall_status),
        target_sensor_names=target_sensor_names,
        sensor_reports=sensor_reports,
        notes=notes,
        real_imu_npz_path=None if resolved_real_path is None else str(resolved_real_path.resolve()),
    )
    frame_estimation_report = {
        "clip_id": str(raw_virtual_imu_sequence.clip_id),
        "status": str(overall_status),
        "real_imu_npz_path": None if resolved_real_path is None else str(resolved_real_path.resolve()),
        "target_sensor_names": list(target_sensor_names),
        "sensor_reports": sensor_reports,
        "notes": list(dict.fromkeys(notes + quality_report.get("notes", []))),
    }
    lag_report = {
        "clip_id": str(raw_virtual_imu_sequence.clip_id),
        "status": str(overall_status),
        "real_imu_npz_path": None if resolved_real_path is None else str(resolved_real_path.resolve()),
        "target_sensor_names": list(target_sensor_names),
        "sensor_lags": lag_sensor_reports,
        "notes": list(dict.fromkeys(notes)),
    }

    artifacts: Dict[str, Any] = {
        "virtual_imu_frame_aligned_npz_path": None,
        "sensor_frame_estimation_report_json_path": None,
        "frame_alignment_quality_report_json_path": None,
    }
    if output_dir is not None:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        aligned_path = output_dir_path / str(aligned_imu_sequence_filename)
        np.savez_compressed(aligned_path, **aligned_virtual_imu_sequence.to_npz_payload())
        artifacts["virtual_imu_frame_aligned_npz_path"] = str(aligned_path.resolve())

        report_path = output_dir_path / str(report_filename)
        write_json_file(frame_estimation_report, report_path)
        artifacts["sensor_frame_estimation_report_json_path"] = str(report_path.resolve())

        quality_path = output_dir_path / str(quality_report_filename)
        write_json_file(quality_report, quality_path)
        artifacts["frame_alignment_quality_report_json_path"] = str(quality_path.resolve())

    return {
        "status": str(overall_status),
        "aligned_virtual_imu_sequence": aligned_virtual_imu_sequence,
        "frame_estimation_report": frame_estimation_report,
        "lag_report": lag_report,
        "quality_report": quality_report,
        "artifacts": artifacts,
    }


def _estimate_single_sensor_frame(
    *,
    sensor_name: str,
    virtual_timestamps: np.ndarray,
    virtual_acc: np.ndarray,
    virtual_gyro: np.ndarray,
    real_timestamps: np.ndarray,
    real_acc: np.ndarray,
    real_gyro: np.ndarray,
    max_lag_sec: float,
    dynamic_percentile: float,
    static_percentile: float,
    min_valid_frames: int,
    min_dynamic_gyro_norm: float,
) -> Dict[str, Any]:
    sensor_report = _build_default_sensor_report(sensor_name)
    lag_sec, lag_corr, lag_overlap = _estimate_lag_from_gyro_norm(
        virtual_timestamps=virtual_timestamps,
        virtual_gyro=virtual_gyro,
        real_timestamps=real_timestamps,
        real_gyro=real_gyro,
        max_lag_sec=max_lag_sec,
    )
    sensor_report["lag_sec"] = _optional_float(lag_sec)
    sensor_report["lag_correlation"] = _optional_float(lag_corr)
    sensor_report["lag_overlap_frames"] = int(lag_overlap)

    aligned_times = virtual_timestamps + np.float32(lag_sec)
    real_gyro_resampled = _interpolate_signal(real_timestamps, real_gyro, aligned_times)
    real_acc_resampled = _interpolate_signal(real_timestamps, real_acc, aligned_times)
    valid_gyro_mask = np.isfinite(virtual_gyro).all(axis=1) & np.isfinite(real_gyro_resampled).all(axis=1)
    valid_acc_mask = np.isfinite(virtual_acc).all(axis=1) & np.isfinite(real_acc_resampled).all(axis=1)
    valid_mask = valid_gyro_mask & valid_acc_mask

    if int(np.count_nonzero(valid_mask)) < int(min_valid_frames):
        sensor_report["notes"].append("insufficient_valid_frames_after_lag_alignment")
        return {
            "sensor_report": sensor_report,
            "lag_sensor_report": _build_lag_sensor_report(sensor_report),
        }

    virtual_gyro_norm = np.linalg.norm(virtual_gyro, axis=1)
    real_gyro_norm = np.linalg.norm(real_gyro_resampled, axis=1)
    dynamic_mask, static_mask = _select_dynamic_and_static_frames(
        valid_mask=valid_mask,
        virtual_gyro_norm=virtual_gyro_norm,
        real_gyro_norm=real_gyro_norm,
        dynamic_percentile=dynamic_percentile,
        static_percentile=static_percentile,
    )
    sensor_report["num_dynamic_frames"] = int(np.count_nonzero(dynamic_mask))
    sensor_report["num_static_frames"] = int(np.count_nonzero(static_mask))

    if sensor_report["num_dynamic_frames"] < int(min_valid_frames):
        sensor_report["notes"].append("insufficient_dynamic_frames")
        return {
            "sensor_report": sensor_report,
            "lag_sensor_report": _build_lag_sensor_report(sensor_report),
        }
    if sensor_report["num_static_frames"] < int(min_valid_frames):
        sensor_report["notes"].append("insufficient_static_frames")
        return {
            "sensor_report": sensor_report,
            "lag_sensor_report": _build_lag_sensor_report(sensor_report),
        }

    dynamic_virtual_norm = float(np.mean(virtual_gyro_norm[dynamic_mask]))
    dynamic_real_norm = float(np.mean(real_gyro_norm[dynamic_mask]))
    sensor_report["dynamic_virtual_gyro_norm_mean"] = dynamic_virtual_norm
    sensor_report["dynamic_real_gyro_norm_mean"] = dynamic_real_norm
    if dynamic_virtual_norm < float(min_dynamic_gyro_norm) or dynamic_real_norm < float(min_dynamic_gyro_norm):
        sensor_report["notes"].append("dynamic_gyro_energy_too_low")
        return {
            "sensor_report": sensor_report,
            "lag_sensor_report": _build_lag_sensor_report(sensor_report),
        }

    signed_permutation_matrix, signed_permutation_payload = _best_signed_permutation(
        virtual_vectors=virtual_gyro[dynamic_mask],
        real_vectors=real_gyro_resampled[dynamic_mask],
    )
    rotated_virtual_gyro_init = _apply_rotation(virtual_gyro, signed_permutation_matrix)
    delta_rotation = _solve_orthogonal_procrustes(
        rotated_virtual_gyro_init[dynamic_mask],
        real_gyro_resampled[dynamic_mask],
    )
    projected_rotation = _project_to_rotation(delta_rotation @ signed_permutation_matrix)
    rotated_virtual_gyro = _apply_rotation(virtual_gyro, projected_rotation)
    rotated_virtual_acc = _apply_rotation(virtual_acc, projected_rotation)

    gyro_corr_before = _mean_component_correlation(
        virtual_gyro[valid_mask],
        real_gyro_resampled[valid_mask],
    )
    gyro_corr_after = _mean_component_correlation(
        rotated_virtual_gyro[valid_mask],
        real_gyro_resampled[valid_mask],
    )
    acc_corr_before = _mean_component_correlation(
        virtual_acc[static_mask],
        real_acc_resampled[static_mask],
    )
    acc_corr_after = _mean_component_correlation(
        rotated_virtual_acc[static_mask],
        real_acc_resampled[static_mask],
    )
    gravity_angle_error_deg = _gravity_angle_error_deg(
        rotated_virtual_acc[static_mask],
        real_acc_resampled[static_mask],
    )
    confidence_score = _compute_confidence_score(
        gyro_corr_before=gyro_corr_before,
        gyro_corr_after=gyro_corr_after,
        acc_corr_before=acc_corr_before,
        acc_corr_after=acc_corr_after,
        gravity_angle_error_deg=gravity_angle_error_deg,
    )

    sensor_report.update(
        {
            "status": "ok",
            "signed_permutation_init": signed_permutation_payload,
            "rotation_matrix": projected_rotation.astype(np.float32).tolist(),
            "rotation_quat_xyzw": Rotation.from_matrix(projected_rotation).as_quat().astype(np.float32).tolist(),
            "gyro_corr_before": _optional_float(gyro_corr_before),
            "gyro_corr_after": _optional_float(gyro_corr_after),
            "acc_corr_before": _optional_float(acc_corr_before),
            "acc_corr_after": _optional_float(acc_corr_after),
            "gravity_angle_error_deg": _optional_float(gravity_angle_error_deg),
            "confidence_score": float(confidence_score),
        }
    )
    if gyro_corr_after <= gyro_corr_before:
        sensor_report["notes"].append("gyro_correlation_not_improved")
        sensor_report["status"] = "warning"
    if acc_corr_after <= acc_corr_before:
        sensor_report["notes"].append("static_acc_correlation_not_improved")
        sensor_report["status"] = "warning"
    if gravity_angle_error_deg is not None and gravity_angle_error_deg > 30.0:
        sensor_report["notes"].append("gravity_alignment_error_above_30_deg")
        sensor_report["status"] = "warning"
    if confidence_score < 0.35:
        sensor_report["notes"].append("low_frame_alignment_confidence")
        sensor_report["status"] = "warning"

    return {
        "sensor_report": sensor_report,
        "lag_sensor_report": _build_lag_sensor_report(sensor_report),
    }


def _load_real_imu_clip(real_imu_npz_path: Path) -> Dict[str, Any]:
    with np.load(real_imu_npz_path, allow_pickle=True) as payload:
        payload_dict = {key: payload[key] for key in payload.files}
    if "imu" in payload_dict:
        imu = np.asarray(payload_dict["imu"], dtype=np.float32)
        if imu.ndim != 3 or imu.shape[-1] < 6:
            raise ValueError("Expected real IMU payload['imu'] with shape [T, S, 6].")
        acc = np.asarray(imu[..., :3], dtype=np.float32)
        gyro = np.asarray(imu[..., 3:6], dtype=np.float32)
    elif "acc" in payload_dict and "gyro" in payload_dict:
        acc = np.asarray(payload_dict["acc"], dtype=np.float32)
        gyro = np.asarray(payload_dict["gyro"], dtype=np.float32)
    else:
        raise ValueError("Unsupported real IMU clip payload. Expected 'imu' or both 'acc' and 'gyro'.")

    timestamps_sec = np.asarray(payload_dict.get("timestamps_sec"), dtype=np.float32)
    if timestamps_sec.ndim != 1 or timestamps_sec.shape[0] != acc.shape[0]:
        raise ValueError("Real IMU clip must expose timestamps_sec aligned with the time dimension.")

    sensor_names = _resolve_real_sensor_names(payload_dict, real_imu_npz_path, sensor_count=int(acc.shape[1]))
    return {
        "timestamps_sec": timestamps_sec,
        "sensor_names": sensor_names,
        "acc": acc,
        "gyro": gyro,
    }


def _resolve_real_sensor_names(
    payload: Mapping[str, Any],
    real_imu_npz_path: Path,
    *,
    sensor_count: int,
) -> list[str]:
    metadata_path = real_imu_npz_path.with_name("metadata.json")
    metadata = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("dataset") == "RobotEmotions":
            sensor_ids = _resolve_real_sensor_ids(payload, metadata, sensor_count=sensor_count)
            if sensor_ids is not None:
                return resolve_sensor_names(sensor_ids)

    if "sensor_names" in payload:
        sensor_names = [str(value) for value in np.asarray(payload["sensor_names"]).tolist()]
        if len(sensor_names) == sensor_count:
            return sensor_names

    if metadata is not None:
        metadata_sensor_names = metadata.get("imu", {}).get("sensor_names", [])
        if isinstance(metadata_sensor_names, list) and len(metadata_sensor_names) == sensor_count:
            return [str(value) for value in metadata_sensor_names]
        sensor_name_by_id = {
            str(key): str(value)
            for key, value in dict(metadata.get("imu", {}).get("sensor_id_to_name", {})).items()
        }
        if "sensor_ids" in payload:
            sensor_ids = [str(int(value)) for value in np.asarray(payload["sensor_ids"]).tolist()]
            if len(sensor_ids) == sensor_count and all(sensor_id in sensor_name_by_id for sensor_id in sensor_ids):
                return [sensor_name_by_id[sensor_id] for sensor_id in sensor_ids]

    if "sensor_ids" in payload:
        sensor_ids = [int(value) for value in np.asarray(payload["sensor_ids"]).tolist()]
        if len(sensor_ids) == sensor_count:
            return [f"sensor_{sensor_id}" for sensor_id in sensor_ids]
    return [f"sensor_{index}" for index in range(sensor_count)]


def _resolve_real_sensor_ids(
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    sensor_count: int,
) -> list[int] | None:
    if "sensor_ids" in payload:
        sensor_ids = [int(value) for value in np.asarray(payload["sensor_ids"]).tolist()]
        if len(sensor_ids) == sensor_count:
            return sensor_ids

    metadata_sensor_ids = metadata.get("imu", {}).get("sensor_ids", [])
    if isinstance(metadata_sensor_ids, list) and len(metadata_sensor_ids) == sensor_count:
        return [int(value) for value in metadata_sensor_ids]
    return None


def _estimate_lag_from_gyro_norm(
    *,
    virtual_timestamps: np.ndarray,
    virtual_gyro: np.ndarray,
    real_timestamps: np.ndarray,
    real_gyro: np.ndarray,
    max_lag_sec: float,
) -> tuple[float, float, int]:
    virtual_norm = np.linalg.norm(np.asarray(virtual_gyro, dtype=np.float32), axis=1)
    real_norm = np.linalg.norm(np.asarray(real_gyro, dtype=np.float32), axis=1)
    lag_step_sec = _lag_step_sec(virtual_timestamps, real_timestamps)
    candidate_lags = np.arange(
        -float(max_lag_sec),
        float(max_lag_sec) + lag_step_sec * 0.5,
        lag_step_sec,
        dtype=np.float32,
    )
    best_lag = np.float32(0.0)
    best_corr = -np.inf
    best_overlap = 0
    for lag_sec in candidate_lags:
        sampled_real_norm = _interpolate_scalar(real_timestamps, real_norm, virtual_timestamps + lag_sec)
        valid_mask = np.isfinite(virtual_norm) & np.isfinite(sampled_real_norm)
        overlap = int(np.count_nonzero(valid_mask))
        if overlap < 10:
            continue
        corr = _safe_correlation(virtual_norm[valid_mask], sampled_real_norm[valid_mask])
        if corr > best_corr:
            best_corr = corr
            best_lag = np.float32(lag_sec)
            best_overlap = overlap
    if not np.isfinite(best_corr):
        return 0.0, 0.0, 0
    return float(best_lag), float(best_corr), int(best_overlap)


def _lag_step_sec(virtual_timestamps: np.ndarray, real_timestamps: np.ndarray) -> float:
    candidates = []
    for timestamps in (virtual_timestamps, real_timestamps):
        diffs = np.diff(np.asarray(timestamps, dtype=np.float32))
        positive_diffs = diffs[diffs > 0.0]
        if positive_diffs.size > 0:
            candidates.append(float(np.median(positive_diffs)))
    if len(candidates) == 0:
        return 0.01
    return max(min(candidates), 0.005)


def _select_dynamic_and_static_frames(
    *,
    valid_mask: np.ndarray,
    virtual_gyro_norm: np.ndarray,
    real_gyro_norm: np.ndarray,
    dynamic_percentile: float,
    static_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid_virtual = np.asarray(virtual_gyro_norm[valid_mask], dtype=np.float32)
    valid_real = np.asarray(real_gyro_norm[valid_mask], dtype=np.float32)
    virtual_dynamic_threshold = np.percentile(valid_virtual, dynamic_percentile)
    real_dynamic_threshold = np.percentile(valid_real, dynamic_percentile)
    virtual_static_threshold = np.percentile(valid_virtual, static_percentile)
    real_static_threshold = np.percentile(valid_real, static_percentile)
    dynamic_mask = (
        valid_mask
        & (virtual_gyro_norm >= np.float32(virtual_dynamic_threshold))
        & (real_gyro_norm >= np.float32(real_dynamic_threshold))
    )
    static_mask = (
        valid_mask
        & (virtual_gyro_norm <= np.float32(virtual_static_threshold))
        & (real_gyro_norm <= np.float32(real_static_threshold))
    )
    return dynamic_mask, static_mask


def _best_signed_permutation(
    *,
    virtual_vectors: np.ndarray,
    real_vectors: np.ndarray,
) -> tuple[np.ndarray, Dict[str, Any]]:
    best_matrix = np.eye(3, dtype=np.float32)
    best_payload = {"indices": [0, 1, 2], "signs": [1, 1, 1]}
    best_corr = -np.inf
    for matrix, payload in _iter_signed_permutation_matrices():
        rotated = _apply_rotation(virtual_vectors, matrix)
        corr = _mean_component_correlation(rotated, real_vectors)
        if corr > best_corr:
            best_corr = corr
            best_matrix = matrix.astype(np.float32, copy=False)
            best_payload = payload
    return best_matrix, best_payload


def _iter_signed_permutation_matrices() -> Iterable[tuple[np.ndarray, Dict[str, Any]]]:
    for indices in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float32)
            for row_index, (column_index, sign_value) in enumerate(zip(indices, signs)):
                matrix[row_index, column_index] = np.float32(sign_value)
            yield matrix, {
                "indices": [int(value) for value in indices],
                "signs": [int(value) for value in signs],
            }


def _solve_orthogonal_procrustes(source_vectors: np.ndarray, target_vectors: np.ndarray) -> np.ndarray:
    source = np.asarray(source_vectors, dtype=np.float64)
    target = np.asarray(target_vectors, dtype=np.float64)
    covariance = source.T @ target
    u, _, vt = np.linalg.svd(covariance, full_matrices=False)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    return rotation.astype(np.float32, copy=False)


def _project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64), full_matrices=False)
    correction = np.eye(3, dtype=np.float64)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    projected = u @ correction @ vt
    return projected.astype(np.float32, copy=False)


def _apply_rotation(vectors: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    return np.asarray(vectors, dtype=np.float32) @ np.asarray(rotation_matrix, dtype=np.float32).T


def _mean_component_correlation(first: np.ndarray, second: np.ndarray) -> float:
    correlations = []
    first_values = np.asarray(first, dtype=np.float32)
    second_values = np.asarray(second, dtype=np.float32)
    for axis_index in range(first_values.shape[1]):
        corr = _safe_correlation(first_values[:, axis_index], second_values[:, axis_index])
        correlations.append(corr)
    return float(np.mean(correlations)) if len(correlations) > 0 else 0.0


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    valid_mask = np.isfinite(first_values) & np.isfinite(second_values)
    if int(np.count_nonzero(valid_mask)) < 2:
        return 0.0
    first_valid = first_values[valid_mask]
    second_valid = second_values[valid_mask]
    first_std = float(np.std(first_valid))
    second_std = float(np.std(second_valid))
    if first_std <= _EPSILON or second_std <= _EPSILON:
        return 0.0
    return float(np.corrcoef(first_valid, second_valid)[0, 1])


def _gravity_angle_error_deg(aligned_virtual_acc: np.ndarray, real_acc: np.ndarray) -> float | None:
    if aligned_virtual_acc.shape[0] == 0 or real_acc.shape[0] == 0:
        return None
    virtual_mean = np.mean(np.asarray(aligned_virtual_acc, dtype=np.float64), axis=0)
    real_mean = np.mean(np.asarray(real_acc, dtype=np.float64), axis=0)
    virtual_norm = float(np.linalg.norm(virtual_mean))
    real_norm = float(np.linalg.norm(real_mean))
    if virtual_norm <= _EPSILON or real_norm <= _EPSILON:
        return None
    cosine = float(np.dot(virtual_mean, real_mean) / (virtual_norm * real_norm))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _compute_confidence_score(
    *,
    gyro_corr_before: float,
    gyro_corr_after: float,
    acc_corr_before: float,
    acc_corr_after: float,
    gravity_angle_error_deg: float | None,
) -> float:
    gyro_after_clamped = max(float(gyro_corr_after), 0.0)
    acc_after_clamped = max(float(acc_corr_after), 0.0)
    gyro_gain = max(float(gyro_corr_after - gyro_corr_before), 0.0)
    acc_gain = max(float(acc_corr_after - acc_corr_before), 0.0)
    gravity_score = 0.5
    if gravity_angle_error_deg is not None:
        gravity_score = max(0.0, 1.0 - float(gravity_angle_error_deg) / 45.0)
    confidence = (
        0.45 * gyro_after_clamped
        + 0.20 * gyro_gain
        + 0.15 * acc_after_clamped
        + 0.10 * acc_gain
        + 0.10 * gravity_score
    )
    return float(np.clip(confidence, 0.0, 1.0))


def _interpolate_signal(timestamps: np.ndarray, values: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    target_timestamps = np.asarray(target_timestamps, dtype=np.float32)
    interpolated = np.empty((target_timestamps.shape[0], values.shape[1]), dtype=np.float32)
    for axis_index in range(values.shape[1]):
        interpolated[:, axis_index] = _interpolate_scalar(
            timestamps,
            values[:, axis_index],
            target_timestamps,
        )
    return interpolated


def _interpolate_scalar(timestamps: np.ndarray, values: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    target_timestamps = np.asarray(target_timestamps, dtype=np.float32)
    valid_mask = np.isfinite(timestamps) & np.isfinite(values)
    if int(np.count_nonzero(valid_mask)) < 2:
        return np.full(target_timestamps.shape, np.nan, dtype=np.float32)
    valid_timestamps = timestamps[valid_mask]
    valid_values = values[valid_mask]
    return np.interp(
        target_timestamps,
        valid_timestamps,
        valid_values,
        left=np.nan,
        right=np.nan,
    ).astype(np.float32, copy=False)


def _find_sensor_index(sensor_names: Sequence[str], sensor_name: str) -> int | None:
    for index, candidate in enumerate(sensor_names):
        if str(candidate) == str(sensor_name):
            return int(index)
    return None


def _merge_sensor_status(sensor_reports: Iterable[Mapping[str, Any]]) -> str:
    statuses = [str(report.get("status", "warning")) for report in sensor_reports]
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status == "warning" for status in statuses):
        return "warning"
    return "ok"


def _build_frame_alignment_quality_report(
    *,
    clip_id: str,
    status: str,
    target_sensor_names: Sequence[str],
    sensor_reports: Mapping[str, Mapping[str, Any]],
    notes: Sequence[str],
    real_imu_npz_path: str | None,
) -> Dict[str, Any]:
    gyro_before = _mean_report_value(sensor_reports.values(), "gyro_corr_before")
    gyro_after = _mean_report_value(sensor_reports.values(), "gyro_corr_after")
    acc_before = _mean_report_value(sensor_reports.values(), "acc_corr_before")
    acc_after = _mean_report_value(sensor_reports.values(), "acc_corr_after")
    quality_notes = list(notes)
    for sensor_name, sensor_report in sensor_reports.items():
        quality_notes.extend([f"{sensor_name}:{value}" for value in sensor_report.get("notes", [])])
    return {
        "clip_id": str(clip_id),
        "enabled": True,
        "status": str(status),
        "real_imu_npz_path": real_imu_npz_path,
        "target_sensor_names": [str(name) for name in target_sensor_names],
        "estimated_sensor_names": [
            str(sensor_name)
            for sensor_name, sensor_report in sensor_reports.items()
            if sensor_report.get("rotation_matrix") is not None
        ],
        "num_target_sensors": int(len(target_sensor_names)),
        "num_estimated_sensors": int(
            sum(1 for sensor_report in sensor_reports.values() if sensor_report.get("rotation_matrix") is not None)
        ),
        "mean_gyro_corr_before": _optional_float(gyro_before),
        "mean_gyro_corr_after": _optional_float(gyro_after),
        "mean_acc_corr_before": _optional_float(acc_before),
        "mean_acc_corr_after": _optional_float(acc_after),
        "sensor_status_by_name": {
            str(sensor_name): str(sensor_report.get("status", "warning"))
            for sensor_name, sensor_report in sensor_reports.items()
        },
        "notes": list(dict.fromkeys(quality_notes)),
    }


def _mean_report_value(sensor_reports: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values = []
    for sensor_report in sensor_reports:
        value = sensor_report.get(key)
        if value is None:
            continue
        values.append(float(value))
    if len(values) == 0:
        return None
    return float(np.mean(values))


def _build_default_sensor_report(sensor_name: str) -> Dict[str, Any]:
    return {
        "status": "warning",
        "lag_sec": None,
        "lag_correlation": None,
        "lag_overlap_frames": 0,
        "rotation_matrix": None,
        "rotation_quat_xyzw": None,
        "signed_permutation_init": None,
        "gyro_corr_before": None,
        "gyro_corr_after": None,
        "acc_corr_before": None,
        "acc_corr_after": None,
        "gravity_angle_error_deg": None,
        "num_dynamic_frames": 0,
        "num_static_frames": 0,
        "confidence_score": 0.0,
        "notes": [],
    }


def _build_lag_sensor_report(sensor_report: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "status": str(sensor_report.get("status", "warning")),
        "lag_sec": sensor_report.get("lag_sec"),
        "lag_correlation": sensor_report.get("lag_correlation"),
        "lag_overlap_frames": sensor_report.get("lag_overlap_frames"),
        "notes": list(sensor_report.get("notes", [])),
    }


def _optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)
