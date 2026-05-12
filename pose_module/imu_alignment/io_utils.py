"""I/O, config, and serialization helpers for the IMU alignment package."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import yaml

from pose_module.interfaces import VirtualIMUSequence
from pose_module.robot_emotions.metadata import resolve_sensor_names

from .interfaces import AlignmentConfig, IMUSequence, SensorSubjectTransform

DEFAULT_ALIGNMENT_CONFIG_PATH = (Path(__file__).resolve().parent.parent / "configs" / "imu_alignment_config.yaml")


def load_alignment_runtime_settings(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML settings used by :func:`run_geometric_alignment`."""

    resolved_path = _resolve_config_path(config_path)
    if resolved_path is None or not resolved_path.exists():
        return _default_runtime_settings(None)
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("imu_alignment_config.yaml must contain a mapping at the top level.")
    settings = _default_runtime_settings(resolved_path)
    settings.update(dict(payload))
    settings["config_path"] = str(resolved_path.resolve())
    return settings


def build_alignment_config(settings: Mapping[str, Any]) -> AlignmentConfig:
    """Instantiate :class:`AlignmentConfig` from raw YAML settings."""

    config = AlignmentConfig(
        estimate_time_lag=bool(
            settings.get("estimate_time_lag", True) and settings.get("enable_time_lag_estimation", True)
        ),
        max_lag_samples=int(settings.get("max_lag_samples", 20)),
        lag_signal=str(settings.get("lag_signal", "gyro_norm")),
        use_acc=bool(settings.get("use_acc", True)),
        use_gyro=bool(settings.get("use_gyro", True)),
        use_bias=bool(settings.get("use_bias", False) and settings.get("enable_bias_scale_refinement", False)),
        use_scale=bool(settings.get("use_scale", False) and settings.get("enable_bias_scale_refinement", False)),
        gravity_window_only=bool(settings.get("gravity_window_only", False)),
        min_motion_norm_gyro=float(settings.get("min_motion_norm_gyro", 0.1)),
        min_motion_norm_acc=float(settings.get("min_motion_norm_acc", 0.2)),
        ridge_lambda=float(settings.get("ridge_lambda", 1e-6)),
    )
    config.validate()
    return config


def sequence_from_virtual_imu(
    virtual_sequence: VirtualIMUSequence,
    *,
    subject_id: str,
    capture_id: str
) -> IMUSequence:
    """Adapt a pipeline ``VirtualIMUSequence`` into the alignment package contract."""

    return IMUSequence(
        subject_id=str(subject_id),
        capture_id=str(capture_id),
        sensor_names=[str(name) for name in virtual_sequence.sensor_names],
        fps=0 if virtual_sequence.fps is None else float(virtual_sequence.fps),
        timestamps=np.asarray(virtual_sequence.timestamps_sec, dtype=np.float32),
        acc=np.asarray(virtual_sequence.acc, dtype=np.float32),
        gyro=np.asarray(virtual_sequence.gyro, dtype=np.float32),
    )


def virtual_from_imu_sequence(
    imu_sequence: IMUSequence,
    *,
    clip_id: str,
    source: str,
    fps: float | None = None
) -> VirtualIMUSequence:
    """Convert an :class:`IMUSequence` back into the pipeline-facing contract."""

    return VirtualIMUSequence(
        clip_id=str(clip_id),
        fps=None if fps is None else float(fps),
        sensor_names=list(imu_sequence.sensor_names),
        acc=np.asarray(imu_sequence.acc, dtype=np.float32),
        gyro=np.asarray(imu_sequence.gyro, dtype=np.float32),
        timestamps_sec=np.asarray(imu_sequence.timestamps, dtype=np.float32),
        source=str(source),
    )


def load_real_imu_sequence(
    real_imu_npz_path: str | Path,
    *,
    subject_id: str | None = None,
    capture_id: str | None = None
) -> IMUSequence:
    """Load a real IMU clip into :class:`IMUSequence`."""

    path = Path(real_imu_npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Real IMU NPZ not found: {path}")

    with np.load(path, allow_pickle=True) as payload:
        payload_dict = {key: payload[key] for key in payload.files}

    acc, gyro = _extract_acc_gyro(payload_dict)
    timestamps = _extract_timestamps(payload_dict, num_frames=int(acc.shape[0]))
    resolved_sensor_names = _resolve_sensor_names(payload_dict, path, sensor_count=int(acc.shape[1]))
    metadata = _load_neighbor_metadata(path)

    resolved_subject_id = (
        None
        if subject_id in (None, "")
        else str(subject_id)
    )
    if resolved_subject_id is None:
        if "subject_id" in payload_dict:
            resolved_subject_id = str(np.asarray(payload_dict["subject_id"]).item())
        elif isinstance(metadata.get("user_id"), int):
            resolved_subject_id = f"user_{int(metadata['user_id']):02d}"
        else:
            resolved_subject_id = str(metadata.get("clip_id", path.stem))

    resolved_capture_id = None if capture_id in (None, "") else str(capture_id)
    if resolved_capture_id is None:
        if "capture_id" in payload_dict:
            resolved_capture_id = str(np.asarray(payload_dict["capture_id"]).item())
        elif "clip_id" in payload_dict:
            resolved_capture_id = str(np.asarray(payload_dict["clip_id"]).item())
        else:
            resolved_capture_id = str(metadata.get("clip_id", path.stem))

    fps = _extract_fps(payload_dict, timestamps)
    return IMUSequence(
        subject_id=resolved_subject_id,
        capture_id=resolved_capture_id,
        sensor_names=resolved_sensor_names,
        fps=fps,
        timestamps=timestamps,
        acc=acc,
        gyro=gyro,
    )


def save_transforms_json(transforms: Mapping[Tuple[str, str], SensorSubjectTransform], path: str | Path) -> None:
    """Persist learned transforms in a human-readable JSON payload."""

    serialized = {
        "transforms": [
            {
                "subject_id": transform.subject_id,
                "sensor_name": transform.sensor_name,
                "rotation": np.asarray(transform.rotation, dtype=np.float32).tolist(),
                "acc_bias": np.asarray(transform.acc_bias, dtype=np.float32).tolist(),
                "gyro_bias": np.asarray(transform.gyro_bias, dtype=np.float32).tolist(),
                "acc_scale": np.asarray(transform.acc_scale, dtype=np.float32).tolist(),
                "gyro_scale": np.asarray(transform.gyro_scale, dtype=np.float32).tolist(),
                "fitted_capture_ids": list(transform.fitted_capture_ids),
                "diagnostics": _to_jsonable(transform.diagnostics),
            }
            for _, transform in sorted(transforms.items())
        ]
    }
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(serialized, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def load_transforms_json(path: str | Path) -> Dict[Tuple[str, str], SensorSubjectTransform]:
    """Load persisted transforms from disk."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    transforms = {}
    for item in payload.get("transforms", []):
        transform = SensorSubjectTransform(
            subject_id=str(item["subject_id"]),
            sensor_name=str(item["sensor_name"]),
            rotation=np.asarray(item["rotation"], dtype=np.float32),
            acc_bias=np.asarray(item.get("acc_bias", [0.0, 0.0, 0.0]), dtype=np.float32),
            gyro_bias=np.asarray(item.get("gyro_bias", [0.0, 0.0, 0.0]), dtype=np.float32),
            acc_scale=np.asarray(item.get("acc_scale", [1.0, 1.0, 1.0]), dtype=np.float32),
            gyro_scale=np.asarray(item.get("gyro_scale", [1.0, 1.0, 1.0]), dtype=np.float32),
            fitted_capture_ids=[str(value) for value in item.get("fitted_capture_ids", [])],
            diagnostics=dict(item.get("diagnostics", {})),
        )
        transforms[(transform.subject_id, transform.sensor_name)] = transform
    return transforms


def split_sequences_by_capture_ids(sequences: Sequence[IMUSequence],*,calibration_capture_ids: Sequence[str]) -> tuple[list[IMUSequence], list[IMUSequence]]:
    """Split sequences into calibration and test partitions by ``capture_id``."""

    calibration_ids = {str(value) for value in calibration_capture_ids}
    calibration_sequences = [sequence for sequence in sequences if str(sequence.capture_id) in calibration_ids]
    test_sequences = [sequence for sequence in sequences if str(sequence.capture_id) not in calibration_ids]
    return calibration_sequences, test_sequences


def _resolve_config_path(config_path: str | Path | None) -> Path | None:
    if config_path not in (None, ""):
        return Path(config_path)
    env_path = os.environ.get("IMU_ALIGNMENT_CONFIG_PATH")
    if env_path not in (None, ""):
        return Path(env_path)
    return DEFAULT_ALIGNMENT_CONFIG_PATH


def _default_runtime_settings(config_path: Path | None) -> dict[str, Any]:
    return {
        "enable": False,
        "enable_time_lag_estimation": True,
        "enable_rotation_alignment": True,
        "enable_bias_scale_refinement": False,
        "estimate_time_lag": True,
        "max_lag_samples": 20,
        "lag_signal": "gyro_norm",
        "use_acc": True,
        "use_gyro": True,
        "use_bias": False,
        "use_scale": False,
        "gravity_window_only": False,
        "min_motion_norm_gyro": 0.1,
        "min_motion_norm_acc": 0.2,
        "ridge_lambda": 1e-6,
        "output_dir": "./output/imu_alignment",
        "save_transforms": True,
        "save_metrics": True,
        "fit_from_current_pair": False,
        "transforms_path": None,
        "config_path": None if config_path is None else str(config_path.resolve()),
    }


def _extract_acc_gyro(payload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "acc" in payload and "gyro" in payload:
        acc = np.asarray(payload["acc"], dtype=np.float32)
        gyro = np.asarray(payload["gyro"], dtype=np.float32)
    elif "imu" in payload:
        imu = np.asarray(payload["imu"], dtype=np.float32)
        if imu.ndim != 3 or imu.shape[-1] < 6:
            raise ValueError("Expected payload['imu'] with shape [T, S, 6] or larger.")
        acc = imu[:, :, :3]
        gyro = imu[:, :, 3:6]
    else:
        raise ValueError("Real IMU NPZ must provide either acc/gyro arrays or an imu array.")
    if acc.ndim != 3 or acc.shape[-1] != 3:
        raise ValueError("acc must have shape [T, S, 3].")
    if gyro.ndim != 3 or gyro.shape[-1] != 3:
        raise ValueError("gyro must have shape [T, S, 3].")
    if acc.shape != gyro.shape:
        raise ValueError("acc and gyro must share the same shape.")
    return acc.astype(np.float32, copy=False), gyro.astype(np.float32, copy=False)


def _extract_timestamps(payload: Mapping[str, Any], *, num_frames: int) -> np.ndarray:
    if "timestamps_sec" in payload:
        timestamps = np.asarray(payload["timestamps_sec"], dtype=np.float32).reshape(-1)
        if timestamps.shape[0] == num_frames:
            return timestamps
    if "fps" in payload:
        fps = float(np.asarray(payload["fps"]).item())
        if fps > 0.0:
            return (np.arange(num_frames, dtype=np.float32) / np.float32(fps)).astype(np.float32, copy=False)
    return np.arange(num_frames, dtype=np.float32)


def _extract_fps(payload: Mapping[str, Any], timestamps: np.ndarray) -> float:
    if "fps" in payload:
        fps = float(np.asarray(payload["fps"]).item())
        if fps > 0.0:
            return float(fps)
    if timestamps.shape[0] >= 2:
        delta = float(np.median(np.diff(timestamps)))
        if delta > 0.0:
            return float(1.0 / delta)
    return 0.0


def _resolve_sensor_names(payload: Mapping[str, Any], path: Path, *, sensor_count: int) -> list[str]:
    if "sensor_ids" in payload:
        sensor_ids = [int(value) for value in np.asarray(payload["sensor_ids"]).reshape(-1).tolist()]
        if len(sensor_ids) == sensor_count:
            return resolve_sensor_names(sensor_ids)
    if "sensor_names" in payload:
        sensor_names = [str(value) for value in np.asarray(payload["sensor_names"]).reshape(-1).tolist()]
        if len(sensor_names) == sensor_count:
            return sensor_names
    metadata = _load_neighbor_metadata(path)
    metadata_sensor_names = metadata.get("imu", {}).get("sensor_names", [])
    if isinstance(metadata_sensor_names, list) and len(metadata_sensor_names) == sensor_count:
        return [str(value) for value in metadata_sensor_names]
    metadata_sensor_ids = metadata.get("imu", {}).get("sensor_ids", [])
    if isinstance(metadata_sensor_ids, list) and len(metadata_sensor_ids) == sensor_count:
        return resolve_sensor_names([int(value) for value in metadata_sensor_ids])
    return [f"sensor_{index}" for index in range(sensor_count)]


def _load_neighbor_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path.with_name("metadata.json")
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value
