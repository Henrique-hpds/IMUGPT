"""Pipeline adapter that applies geometric IMU alignment inside pose_module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from pose_module.interfaces import VirtualIMUSequence
from pose_module.io.cache import write_json_file

from .apply import apply_sensor_subject_transform, apply_transforms_to_imu_sequence
from .fit import fit_sensor_subject_transforms
from .interfaces import build_identity_transform
from .io_utils import (
    build_alignment_config,
    load_alignment_runtime_settings,
    load_real_imu_sequence,
    load_transforms_json,
    save_transforms_json,
    sequence_from_virtual_imu,
    virtual_from_imu_sequence,
)
from .metrics import aggregate_alignment_results

DEFAULT_GEOMETRIC_ALIGNED_IMU_FILENAME = "virtual_imu_geometric_aligned.npz"
DEFAULT_GEOMETRIC_ALIGNMENT_TRANSFORMS_FILENAME = "imu_alignment_transforms.json"
DEFAULT_GEOMETRIC_ALIGNMENT_METRICS_FILENAME = "imu_alignment_metrics.json"
DEFAULT_GEOMETRIC_ALIGNMENT_REPORT_FILENAME = "imu_alignment_quality_report.json"


def run_geometric_alignment(
    virtual_imu_sequence: VirtualIMUSequence,
    output_dir: str | Path,
    real_imu_npz_path: str | Path | None = None,
    config_path: str | Path | None = None,
    *,
    subject_id: str | None = None,
    capture_id: str | None = None,
    transforms: Mapping[Tuple[str, str], Any] | None = None
) -> Dict[str, Any]:
    """Apply optional geometric alignment between virtual and real IMU."""

    settings = load_alignment_runtime_settings(config_path)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    disabled_result = _build_disabled_result(virtual_imu_sequence=virtual_imu_sequence, settings=settings)
    if not bool(settings.get("enable", False)):
        return disabled_result

    resolved_real_path = None if real_imu_npz_path in (None, "") else Path(real_imu_npz_path)
    resolved_subject_id = (
        _infer_subject_id(resolved_real_path, clip_id=str(virtual_imu_sequence.clip_id))
        if subject_id in (None, "")
        else str(subject_id)
    )
    resolved_capture_id = str(virtual_imu_sequence.clip_id) if capture_id in (None, "") else str(capture_id)
    alignment_config = build_alignment_config(settings)
    virtual_sequence = sequence_from_virtual_imu(
        virtual_imu_sequence,
        subject_id=resolved_subject_id,
        capture_id=resolved_capture_id,
    )

    active_transforms: Dict[Tuple[str, str], Any] = {}
    transform_source = "identity"
    transform_path = None if transforms is not None else _resolve_transform_path(settings, output_dir_path)
    if transforms is not None:
        active_transforms = {key: value for key, value in transforms.items()}
        transform_source = "provided"
    elif transform_path is not None and transform_path.exists():
        active_transforms = load_transforms_json(transform_path)
        transform_source = "loaded"
    elif resolved_real_path is not None and bool(settings.get("fit_from_current_pair", True)):
        real_sequence = load_real_imu_sequence(
            resolved_real_path,
            subject_id=resolved_subject_id,
            capture_id=resolved_capture_id,
        )
        if bool(settings.get("enable_rotation_alignment", True)):
            active_transforms = fit_sensor_subject_transforms([real_sequence], [virtual_sequence], alignment_config)
            transform_source = "fitted_on_current_pair"
        else:
            active_transforms = {
                (resolved_subject_id, sensor_name): build_identity_transform(
                    subject_id=resolved_subject_id,
                    sensor_name=sensor_name,
                    fitted_capture_ids=[resolved_capture_id],
                    diagnostics={"notes": ["rotation_alignment_disabled_identity_transform"]},
                )
                for sensor_name in virtual_sequence.sensor_names
            }
            transform_source = "identity_rotation_disabled"
        if transform_path is not None and bool(settings.get("save_transforms", True)):
            save_transforms_json(active_transforms, transform_path)

    if len(active_transforms) == 0:
        skip_reason = "no_real_reference_or_persisted_transforms_available"
        if resolved_real_path is not None and not bool(settings.get("fit_from_current_pair", True)):
            skip_reason = "fit_from_current_pair_disabled_and_no_persisted_transforms_available"
        skipped_result = _build_skipped_result(
            virtual_imu_sequence=virtual_imu_sequence,
            settings=settings,
            subject_id=resolved_subject_id,
            capture_id=resolved_capture_id,
            real_imu_npz_path=resolved_real_path,
            skip_reason=skip_reason,
        )
        return skipped_result

    aligned_sequence = apply_transforms_to_imu_sequence(virtual_sequence, active_transforms)
    aligned_virtual_sequence = virtual_from_imu_sequence(
        aligned_sequence,
        clip_id=str(virtual_imu_sequence.clip_id),
        source=f"{virtual_imu_sequence.source}_geometric_aligned",
        fps=None if virtual_imu_sequence.fps is None else float(virtual_imu_sequence.fps),
    )

    metrics_before: dict[str, Any] = {}
    metrics_after: dict[str, Any] = {}
    if resolved_real_path is not None:
        real_sequence = load_real_imu_sequence(
            resolved_real_path,
            subject_id=resolved_subject_id,
            capture_id=resolved_capture_id,
        )
        sensor_results = apply_sensor_subject_transform(
            real_sequence,
            virtual_sequence,
            active_transforms,
            alignment_config,
        )
        aggregated = aggregate_alignment_results(sensor_results)
        metrics_before = {
            "num_results": int(aggregated["num_results"]),
            "per_sensor": [
                {
                    "subject_id": item["subject_id"],
                    "capture_id": item["capture_id"],
                    "sensor_name": item["sensor_name"],
                    "lag_samples": item["lag_samples"],
                    "metrics": item["metrics_before"],
                }
                for item in aggregated["per_sensor"]
            ],
            "mean_acc_corr": aggregated["mean_acc_corr_before"],
            "mean_gyro_corr": aggregated["mean_gyro_corr_before"],
        }
        metrics_after = {
            "num_results": int(aggregated["num_results"]),
            "per_sensor": [
                {
                    "subject_id": item["subject_id"],
                    "capture_id": item["capture_id"],
                    "sensor_name": item["sensor_name"],
                    "lag_samples": item["lag_samples"],
                    "metrics": item["metrics_after"],
                }
                for item in aggregated["per_sensor"]
            ],
            "mean_acc_corr": aggregated["mean_acc_corr_after"],
            "mean_gyro_corr": aggregated["mean_gyro_corr_after"],
        }
    quality_report = {
        "enabled": True,
        "status": "ok",
        "subject_id": resolved_subject_id,
        "capture_id": resolved_capture_id,
        "transform_source": transform_source,
        "real_imu_npz_path": None if resolved_real_path is None else str(resolved_real_path.resolve()),
        "estimated_sensor_names": sorted({key[1] for key in active_transforms.keys()}),
        "mean_acc_corr_before": metrics_before.get("mean_acc_corr"),
        "mean_acc_corr_after": metrics_after.get("mean_acc_corr"),
        "mean_gyro_corr_before": metrics_before.get("mean_gyro_corr"),
        "mean_gyro_corr_after": metrics_after.get("mean_gyro_corr"),
        "notes": [],
    }
    result = {
        "status": quality_report["status"],
        "enabled": True,
        "aligned_virtual_imu_sequence": aligned_virtual_sequence,
        "transforms": active_transforms,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "quality_report": quality_report,
        "artifacts": {
            "virtual_imu_geometric_aligned_npz_path": str(
                (output_dir_path / DEFAULT_GEOMETRIC_ALIGNED_IMU_FILENAME).resolve()
            ),
            "imu_alignment_transforms_json_path": None if transform_path is None else str(transform_path.resolve()),
            "imu_alignment_metrics_json_path": str(
                (output_dir_path / DEFAULT_GEOMETRIC_ALIGNMENT_METRICS_FILENAME).resolve()
            ),
            "imu_alignment_quality_report_json_path": str(
                (output_dir_path / DEFAULT_GEOMETRIC_ALIGNMENT_REPORT_FILENAME).resolve()
            ),
            "imu_alignment_config_path": settings.get("config_path"),
        },
    }
    _write_result_artifacts(result, output_dir_path, save_metrics=bool(settings.get("save_metrics", True)))
    return result


def _build_disabled_result(*, virtual_imu_sequence: VirtualIMUSequence, settings: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "not_enabled",
        "enabled": False,
        "aligned_virtual_imu_sequence": VirtualIMUSequence(
            clip_id=str(virtual_imu_sequence.clip_id),
            fps=virtual_imu_sequence.fps,
            sensor_names=list(virtual_imu_sequence.sensor_names),
            acc=np.asarray(virtual_imu_sequence.acc, dtype=np.float32),
            gyro=np.asarray(virtual_imu_sequence.gyro, dtype=np.float32),
            timestamps_sec=np.asarray(virtual_imu_sequence.timestamps_sec, dtype=np.float32),
            source=str(virtual_imu_sequence.source),
        ),
        "transforms": {},
        "metrics_before": {},
        "metrics_after": {},
        "quality_report": {
            "enabled": False,
            "status": None,
            "subject_id": None,
            "capture_id": str(virtual_imu_sequence.clip_id),
            "transform_source": None,
            "real_imu_npz_path": None,
            "estimated_sensor_names": [],
            "mean_acc_corr_before": None,
            "mean_acc_corr_after": None,
            "mean_gyro_corr_before": None,
            "mean_gyro_corr_after": None,
            "notes": [],
        },
        "artifacts": {
            "virtual_imu_geometric_aligned_npz_path": None,
            "imu_alignment_transforms_json_path": None,
            "imu_alignment_metrics_json_path": None,
            "imu_alignment_quality_report_json_path": None,
            "imu_alignment_config_path": settings.get("config_path"),
        },
    }


def _build_skipped_result(
    *,
    virtual_imu_sequence: VirtualIMUSequence,
    settings: Dict[str, Any],
    subject_id: str,
    capture_id: str,
    real_imu_npz_path: Path | None,
    skip_reason: str
) -> Dict[str, Any]:
    aligned_virtual_sequence = VirtualIMUSequence(
        clip_id=str(virtual_imu_sequence.clip_id),
        fps=virtual_imu_sequence.fps,
        sensor_names=list(virtual_imu_sequence.sensor_names),
        acc=np.asarray(virtual_imu_sequence.acc, dtype=np.float32),
        gyro=np.asarray(virtual_imu_sequence.gyro, dtype=np.float32),
        timestamps_sec=np.asarray(virtual_imu_sequence.timestamps_sec, dtype=np.float32),
        source=str(virtual_imu_sequence.source),
    )
    return {
        "status": "skipped",
        "enabled": False,
        "aligned_virtual_imu_sequence": aligned_virtual_sequence,
        "transforms": {},
        "metrics_before": {},
        "metrics_after": {},
        "quality_report": {
            "enabled": False,
            "status": None,
            "subject_id": str(subject_id),
            "capture_id": str(capture_id),
            "transform_source": None,
            "real_imu_npz_path": None if real_imu_npz_path is None else str(real_imu_npz_path.resolve()),
            "estimated_sensor_names": [],
            "mean_acc_corr_before": None,
            "mean_acc_corr_after": None,
            "mean_gyro_corr_before": None,
            "mean_gyro_corr_after": None,
            "skip_reason": str(skip_reason),
            "notes": [],
        },
        "artifacts": {
            "virtual_imu_geometric_aligned_npz_path": None,
            "imu_alignment_transforms_json_path": None,
            "imu_alignment_metrics_json_path": None,
            "imu_alignment_quality_report_json_path": None,
            "imu_alignment_config_path": settings.get("config_path"),
        },
    }


def _resolve_transform_path(settings: Dict[str, Any], output_dir: Path) -> Path | None:
    configured_path = settings.get("transforms_path")
    if configured_path not in (None, ""):
        return Path(configured_path)
    if bool(settings.get("save_transforms", True)):
        return output_dir / DEFAULT_GEOMETRIC_ALIGNMENT_TRANSFORMS_FILENAME
    return None


def _infer_subject_id(real_imu_npz_path: Path | None, *, clip_id: str) -> str:
    if real_imu_npz_path is None:
        return str(clip_id)
    metadata_path = real_imu_npz_path.with_name("metadata.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(metadata.get("domain"), str) and isinstance(metadata.get("user_id"), int):
            return f"{str(metadata['domain'])}_user_{int(metadata['user_id']):02d}"
        if isinstance(metadata.get("user_id"), int):
            return f"user_{int(metadata['user_id']):02d}"
    return str(clip_id)


def _write_result_artifacts(result: Dict[str, Any], output_dir: Path, *, save_metrics: bool) -> None:
    aligned_path = output_dir / DEFAULT_GEOMETRIC_ALIGNED_IMU_FILENAME
    np.savez_compressed(aligned_path, **result["aligned_virtual_imu_sequence"].to_npz_payload())
    result["artifacts"]["virtual_imu_geometric_aligned_npz_path"] = str(aligned_path.resolve())

    quality_path = output_dir / DEFAULT_GEOMETRIC_ALIGNMENT_REPORT_FILENAME
    write_json_file(result["quality_report"], quality_path)
    result["artifacts"]["imu_alignment_quality_report_json_path"] = str(quality_path.resolve())

    metrics_path = output_dir / DEFAULT_GEOMETRIC_ALIGNMENT_METRICS_FILENAME
    metrics_payload = {
        "status": result["status"],
        "metrics_before": result.get("metrics_before", {}),
        "metrics_after": result.get("metrics_after", {}),
    }
    if save_metrics:
        write_json_file(metrics_payload, metrics_path)
        result["artifacts"]["imu_alignment_metrics_json_path"] = str(metrics_path.resolve())
    else:
        result["artifacts"]["imu_alignment_metrics_json_path"] = None
