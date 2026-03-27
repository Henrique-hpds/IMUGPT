"""Stage 5.10: synthesize virtual IMU streams from IK outputs and sensor layout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from pose_module.export.ik_adapter import forward_kinematics_from_ik_sequence
from pose_module.interfaces import IKSequence, VirtualIMUSequence
from pose_module.io.cache import write_json_file
from pose_module.processing.imu_calibration import (
    DEFAULT_CALIBRATION_PERCENTILE_RESOLUTION,
    DEFAULT_CALIBRATION_SIGNAL_MODE,
    calibrate_virtual_imu_sequence,
)

DEFAULT_SENSOR_LAYOUT_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "sensor_layout.yaml"
)
DEFAULT_GRAVITY_M_S2 = (0.0, -9.81, 0.0)
DEFAULT_IMU_SEQUENCE_FILENAME = "virtual_imu.npz"
DEFAULT_RAW_IMU_SEQUENCE_FILENAME = "virtual_imu_raw.npz"
DEFAULT_IMU_REPORT_FILENAME = "virtual_imu_report.json"
DEFAULT_IMU_CALIBRATION_REPORT_FILENAME = "virtual_imu_calibration_report.json"
DEFAULT_SENSOR_LAYOUT_REPORT_FILENAME = "sensor_layout_resolved.json"
DEFAULT_ACC_NOISE_STD_M_S2 = 0.0
DEFAULT_GYRO_NOISE_STD_RAD_S = 0.0
DEFAULT_RANDOM_SEED = 0
DEFAULT_MAX_ACCELERATION_WARNING_M_S2 = 100.0
DEFAULT_MAX_GYRO_WARNING_RAD_S = 25.0
_EPSILON = 1e-6


def run_imusim(
    local_joint_rotations: np.ndarray | IKSequence,
    root_translation_m: np.ndarray | None = None,
    sensor_layout_path: str | Path | None = None,
    *,
    joint_names: Sequence[str] | None = None,
    parents: Sequence[int] | None = None,
    joint_offsets_m: np.ndarray | None = None,
    fps: float | None = None,
    clip_id: str | None = None,
    timestamps_sec: np.ndarray | None = None,
    output_dir: str | Path | None = None,
    gravity_m_s2: Sequence[float] = DEFAULT_GRAVITY_M_S2,
    acc_noise_std_m_s2: float | None = None,
    gyro_noise_std_rad_s: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    imu_sequence_filename: str = DEFAULT_IMU_SEQUENCE_FILENAME,
    raw_imu_sequence_filename: str = DEFAULT_RAW_IMU_SEQUENCE_FILENAME,
    report_filename: str = DEFAULT_IMU_REPORT_FILENAME,
    calibration_report_filename: str = DEFAULT_IMU_CALIBRATION_REPORT_FILENAME,
    layout_report_filename: str = DEFAULT_SENSOR_LAYOUT_REPORT_FILENAME,
    max_acceleration_warning_m_s2: float = DEFAULT_MAX_ACCELERATION_WARNING_M_S2,
    max_gyro_warning_rad_s: float = DEFAULT_MAX_GYRO_WARNING_RAD_S,
    real_imu_reference_path: str | Path | None = None,
    real_imu_activity_label: Any = None,
    real_imu_signal_mode: str = DEFAULT_CALIBRATION_SIGNAL_MODE,
    real_imu_percentile_resolution: int = DEFAULT_CALIBRATION_PERCENTILE_RESOLUTION,
    real_imu_per_class_calibration: bool = True,
) -> Dict[str, Any]:
    """Generate virtual accelerometer and gyroscope sequences from IK outputs."""

    ik_sequence = _normalize_imusim_inputs(
        local_joint_rotations=local_joint_rotations,
        root_translation_m=root_translation_m,
        joint_names=joint_names,
        parents=parents,
        joint_offsets_m=joint_offsets_m,
        fps=fps,
        clip_id=clip_id,
        timestamps_sec=timestamps_sec,
    )
    sensor_layout = load_sensor_layout(
        DEFAULT_SENSOR_LAYOUT_PATH if sensor_layout_path is None else sensor_layout_path
    )
    if acc_noise_std_m_s2 is None:
        acc_noise_std_m_s2 = float(sensor_layout.get("default_acc_noise_std_m_s2", DEFAULT_ACC_NOISE_STD_M_S2))
    if gyro_noise_std_rad_s is None:
        gyro_noise_std_rad_s = float(
            sensor_layout.get("default_gyro_noise_std_rad_s", DEFAULT_GYRO_NOISE_STD_RAD_S)
        )
    gravity_vector = np.asarray(
        sensor_layout.get("gravity_m_s2", list(gravity_m_s2)),
        dtype=np.float32,
    )
    if gravity_vector.shape != (3,):
        raise ValueError("gravity_m_s2 must be a 3-vector.")

    kinematics = forward_kinematics_from_ik_sequence(ik_sequence)
    joint_positions_global_m = np.asarray(kinematics["joint_positions_global_m"], dtype=np.float32)
    joint_rotation_global_matrices = np.asarray(
        kinematics["joint_rotation_global_matrices"],
        dtype=np.float32,
    )
    joint_name_to_index = {
        str(name): index for index, name in enumerate(ik_sequence.joint_names_3d)
    }
    resolved_sensors = _resolve_sensor_specs(
        sensor_layout=sensor_layout,
        joint_name_to_index=joint_name_to_index,
        joint_offsets_m=np.asarray(ik_sequence.joint_offsets_m, dtype=np.float32),
    )

    sensor_positions = np.zeros((ik_sequence.num_frames, len(resolved_sensors), 3), dtype=np.float32)
    sensor_rotations = np.zeros((ik_sequence.num_frames, len(resolved_sensors), 3, 3), dtype=np.float32)
    for sensor_index, sensor_spec in enumerate(resolved_sensors):
        anchor_joint_index = int(sensor_spec["anchor_joint_index"])
        orientation_joint_index = int(sensor_spec["orientation_joint_index"])
        local_offset_m = np.asarray(sensor_spec["local_offset_m"], dtype=np.float32)
        sensor_rotations[:, sensor_index] = joint_rotation_global_matrices[:, orientation_joint_index]
        sensor_positions[:, sensor_index] = (
            joint_positions_global_m[:, anchor_joint_index]
            + np.einsum(
                "tij,j->ti",
                joint_rotation_global_matrices[:, orientation_joint_index],
                local_offset_m,
                dtype=np.float32,
            )
        ).astype(np.float32, copy=False)

    timestamps = np.asarray(ik_sequence.timestamps_sec, dtype=np.float32)
    world_acceleration = _second_derivative(sensor_positions, timestamps)
    specific_force_world = world_acceleration - gravity_vector[None, None, :]
    acc = np.einsum(
        "tsji,tsj->tsi",
        sensor_rotations,
        specific_force_world,
        dtype=np.float32,
    ).astype(np.float32, copy=False)
    gyro = _estimate_local_angular_velocity(sensor_rotations, timestamps)

    rng = np.random.default_rng(int(random_seed))
    if float(acc_noise_std_m_s2) > 0.0:
        acc = acc + rng.normal(
            0.0,
            float(acc_noise_std_m_s2),
            size=acc.shape,
        ).astype(np.float32, copy=False)
    if float(gyro_noise_std_rad_s) > 0.0:
        gyro = gyro + rng.normal(
            0.0,
            float(gyro_noise_std_rad_s),
            size=gyro.shape,
        ).astype(np.float32, copy=False)

    raw_imu_sequence = VirtualIMUSequence(
        clip_id=str(ik_sequence.clip_id),
        fps=None if ik_sequence.fps is None else float(ik_sequence.fps),
        sensor_names=[str(sensor_spec["name"]) for sensor_spec in resolved_sensors],
        acc=acc.astype(np.float32, copy=False),
        gyro=gyro.astype(np.float32, copy=False),
        timestamps_sec=timestamps.astype(np.float32, copy=False),
        source=f"{ik_sequence.source}_virtual_imu",
    )
    calibration_report = None
    imu_sequence = raw_imu_sequence
    if real_imu_reference_path not in (None, ""):
        calibration_result = calibrate_virtual_imu_sequence(
            raw_imu_sequence,
            real_imu_reference_path=str(real_imu_reference_path),
            activity_label=real_imu_activity_label,
            signal_mode=str(real_imu_signal_mode),
            percentile_resolution=int(real_imu_percentile_resolution),
            per_class=bool(real_imu_per_class_calibration),
        )
        imu_sequence = calibration_result["virtual_imu_sequence"]
        calibration_report = dict(calibration_result["calibration_report"])

    quality_report = _build_virtual_imu_quality_report(
        imu_sequence=imu_sequence,
        acc_noise_std_m_s2=float(acc_noise_std_m_s2),
        gyro_noise_std_rad_s=float(gyro_noise_std_rad_s),
        max_acceleration_warning_m_s2=float(max_acceleration_warning_m_s2),
        max_gyro_warning_rad_s=float(max_gyro_warning_rad_s),
        calibration_report=calibration_report,
    )

    output_dir_path = None if output_dir is None else Path(output_dir)
    artifacts: Dict[str, Any] = {
        "virtual_imu_npz_path": None,
        "virtual_imu_raw_npz_path": None,
        "virtual_imu_report_json_path": None,
        "virtual_imu_calibration_report_json_path": None,
        "sensor_layout_resolved_json_path": None,
    }
    if output_dir_path is not None:
        output_dir_path.mkdir(parents=True, exist_ok=True)
        imu_sequence_path = output_dir_path / str(imu_sequence_filename)
        np.savez_compressed(imu_sequence_path, **imu_sequence.to_npz_payload())
        artifacts["virtual_imu_npz_path"] = str(imu_sequence_path.resolve())
        if calibration_report is not None:
            raw_imu_sequence_path = output_dir_path / str(raw_imu_sequence_filename)
            np.savez_compressed(raw_imu_sequence_path, **raw_imu_sequence.to_npz_payload())
            artifacts["virtual_imu_raw_npz_path"] = str(raw_imu_sequence_path.resolve())

        report_path = output_dir_path / str(report_filename)
        write_json_file(quality_report, report_path)
        artifacts["virtual_imu_report_json_path"] = str(report_path.resolve())
        if calibration_report is not None:
            calibration_report_path = output_dir_path / str(calibration_report_filename)
            write_json_file(calibration_report, calibration_report_path)
            artifacts["virtual_imu_calibration_report_json_path"] = str(calibration_report_path.resolve())

        layout_report_path = output_dir_path / str(layout_report_filename)
        write_json_file(
            {
                "sensor_names": [str(sensor_spec["name"]) for sensor_spec in resolved_sensors],
                "resolved_sensors": [
                    {
                        "name": str(sensor_spec["name"]),
                        "anchor_joint": str(sensor_spec["anchor_joint"]),
                        "orientation_joint": str(sensor_spec["orientation_joint"]),
                        "local_offset_m": [float(x) for x in sensor_spec["local_offset_m"]],
                    }
                    for sensor_spec in resolved_sensors
                ],
            },
            layout_report_path,
        )
        artifacts["sensor_layout_resolved_json_path"] = str(layout_report_path.resolve())

    return {
        "virtual_imu_sequence": imu_sequence,
        "raw_virtual_imu_sequence": raw_imu_sequence,
        "quality_report": quality_report,
        "calibration_report": calibration_report,
        "artifacts": artifacts,
        "sensor_positions_global_m": sensor_positions.astype(np.float32, copy=False),
        "sensor_rotation_global_matrices": sensor_rotations.astype(np.float32, copy=False),
    }


def load_sensor_layout(sensor_layout_path: str | Path) -> Mapping[str, Any]:
    path = Path(sensor_layout_path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
    except ModuleNotFoundError:
        payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise ValueError("sensor_layout must define a mapping at the top level.")
    if "sensors" not in payload:
        raise ValueError("sensor_layout must define a top-level 'sensors' list.")
    if not isinstance(payload["sensors"], list) or len(payload["sensors"]) == 0:
        raise ValueError("sensor_layout 'sensors' must be a non-empty list.")
    return payload


def _normalize_imusim_inputs(
    *,
    local_joint_rotations: np.ndarray | IKSequence,
    root_translation_m: np.ndarray | None,
    joint_names: Sequence[str] | None,
    parents: Sequence[int] | None,
    joint_offsets_m: np.ndarray | None,
    fps: float | None,
    clip_id: str | None,
    timestamps_sec: np.ndarray | None,
) -> IKSequence:
    if isinstance(local_joint_rotations, IKSequence):
        ik_sequence = local_joint_rotations
        if root_translation_m is not None or joint_names is not None or parents is not None or joint_offsets_m is not None:
            raise ValueError("run_imusim should receive either an IKSequence or the raw IK tensors, not both.")
        return ik_sequence

    rotations = np.asarray(local_joint_rotations, dtype=np.float32)
    if rotations.ndim != 3 or rotations.shape[-1] != 4:
        raise ValueError("run_imusim expects local_joint_rotations with shape [T, J, 4].")
    if root_translation_m is None or joint_names is None or parents is None or joint_offsets_m is None:
        raise ValueError(
            "run_imusim requires root_translation_m, joint_names, parents, and joint_offsets_m with raw IK tensors."
        )
    root_translation = np.asarray(root_translation_m, dtype=np.float32)
    offsets = np.asarray(joint_offsets_m, dtype=np.float32)
    if root_translation.shape != (rotations.shape[0], 3):
        raise ValueError("run_imusim expects root_translation_m with shape [T, 3].")
    if offsets.shape != (rotations.shape[1], 3):
        raise ValueError("run_imusim expects joint_offsets_m with shape [J, 3].")
    if timestamps_sec is None:
        if fps is None or float(fps) <= 0.0:
            timestamps = np.arange(rotations.shape[0], dtype=np.float32)
        else:
            timestamps = np.arange(rotations.shape[0], dtype=np.float32) / np.float32(float(fps))
    else:
        timestamps = np.asarray(timestamps_sec, dtype=np.float32)
    return IKSequence(
        clip_id=str("clip_virtual_imu" if clip_id is None else clip_id),
        fps=None if fps is None else float(fps),
        fps_original=None,
        joint_names_3d=[str(name) for name in joint_names],
        local_joint_rotations=rotations.astype(np.float32, copy=False),
        root_translation_m=root_translation.astype(np.float32, copy=False),
        joint_offsets_m=offsets.astype(np.float32, copy=False),
        skeleton_parents=[int(parent) for parent in parents],
        frame_indices=np.arange(rotations.shape[0], dtype=np.int32),
        timestamps_sec=timestamps.astype(np.float32, copy=False),
        source="ik_sequence",
        rotation_representation="quaternion_wxyz",
    )


def _resolve_sensor_specs(
    *,
    sensor_layout: Mapping[str, Any],
    joint_name_to_index: Mapping[str, int],
    joint_offsets_m: np.ndarray,
) -> list[Dict[str, Any]]:
    resolved_sensors: list[Dict[str, Any]] = []
    for raw_sensor in sensor_layout.get("sensors", []):
        if not isinstance(raw_sensor, Mapping):
            raise ValueError("Each sensor definition must be a mapping.")
        name = str(raw_sensor["name"])
        orientation_joint = raw_sensor.get("orientation_joint")
        local_offset_m = raw_sensor.get("offset_m")

        if "attach_joint" in raw_sensor:
            anchor_joint = str(raw_sensor["attach_joint"])
            if anchor_joint not in joint_name_to_index:
                raise ValueError(f"Unknown attach_joint '{anchor_joint}' in sensor layout.")
            if local_offset_m is None:
                local_offset = np.zeros((3,), dtype=np.float32)
            else:
                local_offset = np.asarray(local_offset_m, dtype=np.float32)
            orientation_joint = anchor_joint if orientation_joint is None else str(orientation_joint)
        else:
            start_joint = str(raw_sensor["segment_start_joint"])
            end_joint = str(raw_sensor["segment_end_joint"])
            if start_joint not in joint_name_to_index or end_joint not in joint_name_to_index:
                raise ValueError(
                    f"Unknown segment joints '{start_joint}' -> '{end_joint}' in sensor layout."
                )
            fraction = float(raw_sensor.get("segment_fraction", 0.5))
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("segment_fraction must be within [0.0, 1.0].")
            anchor_joint = start_joint
            local_offset = (
                np.asarray(joint_offsets_m[joint_name_to_index[end_joint]], dtype=np.float32) * np.float32(fraction)
            ).astype(np.float32, copy=False)
            orientation_joint = start_joint if orientation_joint is None else str(orientation_joint)

        if orientation_joint not in joint_name_to_index:
            raise ValueError(f"Unknown orientation_joint '{orientation_joint}' in sensor layout.")
        if local_offset.shape != (3,):
            raise ValueError(f"Sensor '{name}' offset_m/local segment offset must be a 3-vector.")

        resolved_sensors.append(
            {
                "name": str(name),
                "anchor_joint": str(anchor_joint),
                "anchor_joint_index": int(joint_name_to_index[anchor_joint]),
                "orientation_joint": str(orientation_joint),
                "orientation_joint_index": int(joint_name_to_index[orientation_joint]),
                "local_offset_m": local_offset.astype(np.float32, copy=False),
            }
        )
    return resolved_sensors


def _second_derivative(values: np.ndarray, timestamps_sec: np.ndarray) -> np.ndarray:
    values_float = np.asarray(values, dtype=np.float64)
    timestamps = np.asarray(timestamps_sec, dtype=np.float64)
    if values_float.shape[0] <= 1:
        return np.zeros_like(values_float, dtype=np.float32)
    if timestamps.ndim != 1 or timestamps.shape[0] != values_float.shape[0]:
        raise ValueError("timestamps_sec must match the frame dimension.")
    velocities = np.gradient(values_float, timestamps, axis=0, edge_order=1)
    accelerations = np.gradient(velocities, timestamps, axis=0, edge_order=1)
    return accelerations.astype(np.float32, copy=False)


def _estimate_local_angular_velocity(
    rotation_matrices: np.ndarray,
    timestamps_sec: np.ndarray,
) -> np.ndarray:
    rotations = np.asarray(rotation_matrices, dtype=np.float64)
    timestamps = np.asarray(timestamps_sec, dtype=np.float64)
    num_frames, num_sensors = rotations.shape[:2]
    if num_frames <= 1:
        return np.zeros((num_frames, num_sensors, 3), dtype=np.float32)

    interval_gyro = np.zeros((num_frames - 1, num_sensors, 3), dtype=np.float64)
    for frame_index in range(num_frames - 1):
        dt = float(max(timestamps[frame_index + 1] - timestamps[frame_index], _EPSILON))
        delta_rotation = np.einsum(
            "sji,sjk->sik",
            rotations[frame_index],
            rotations[frame_index + 1],
        )
        # Equivalent to R_t^T @ R_t+1 for each sensor.
        interval_gyro[frame_index] = Rotation.from_matrix(delta_rotation).as_rotvec() / dt

    gyro = np.zeros((num_frames, num_sensors, 3), dtype=np.float64)
    gyro[0] = interval_gyro[0]
    gyro[-1] = interval_gyro[-1]
    if num_frames > 2:
        gyro[1:-1] = 0.5 * (interval_gyro[:-1] + interval_gyro[1:])
    return gyro.astype(np.float32, copy=False)


def _build_virtual_imu_quality_report(
    *,
    imu_sequence: VirtualIMUSequence,
    acc_noise_std_m_s2: float,
    gyro_noise_std_rad_s: float,
    max_acceleration_warning_m_s2: float,
    max_gyro_warning_rad_s: float,
    calibration_report: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    acc = np.asarray(imu_sequence.acc, dtype=np.float32)
    gyro = np.asarray(imu_sequence.gyro, dtype=np.float32)
    notes = []
    finite_acc = np.isfinite(acc).all()
    finite_gyro = np.isfinite(gyro).all()
    virtual_imu_ok = bool(finite_acc and finite_gyro)
    if not finite_acc:
        notes.append("virtual_acc_contains_nan")
    if not finite_gyro:
        notes.append("virtual_gyro_contains_nan")

    max_acc_norm = float(np.max(np.linalg.norm(acc, axis=2))) if acc.size > 0 else 0.0
    max_gyro_norm = float(np.max(np.linalg.norm(gyro, axis=2))) if gyro.size > 0 else 0.0
    if max_acc_norm > float(max_acceleration_warning_m_s2):
        notes.append(f"virtual_acc_norm_above_threshold:{max_acc_norm:.4f}")
    if max_gyro_norm > float(max_gyro_warning_rad_s):
        notes.append(f"virtual_gyro_norm_above_threshold:{max_gyro_norm:.4f}")
    if float(acc_noise_std_m_s2) > 0.0:
        notes.append("accelerometer_noise_applied")
    if float(gyro_noise_std_rad_s) > 0.0:
        notes.append("gyroscope_noise_applied")
    if calibration_report is not None:
        notes.extend([str(value) for value in calibration_report.get("notes", [])])

    status = "ok"
    if not virtual_imu_ok:
        status = "fail"
    elif max_acc_norm > float(max_acceleration_warning_m_s2) or max_gyro_norm > float(max_gyro_warning_rad_s):
        status = "warning"
    if calibration_report is not None and calibration_report.get("status") == "fail":
        status = "fail"
    elif calibration_report is not None and calibration_report.get("status") == "warning" and status != "fail":
        status = "warning"

    return {
        "clip_id": str(imu_sequence.clip_id),
        "status": str(status),
        "fps": None if imu_sequence.fps is None else float(imu_sequence.fps),
        "num_frames": int(imu_sequence.num_frames),
        "num_sensors": int(imu_sequence.num_sensors),
        "sensor_names": list(imu_sequence.sensor_names),
        "virtual_imu_ok": bool(virtual_imu_ok),
        "acc_noise_std_m_s2": float(acc_noise_std_m_s2),
        "gyro_noise_std_rad_s": float(gyro_noise_std_rad_s),
        "real_imu_calibration_applied": bool(calibration_report is not None),
        "real_imu_calibration_signal_mode": None if calibration_report is None else calibration_report.get("signal_mode"),
        "real_imu_calibration_per_class_applied": (
            None if calibration_report is None else calibration_report.get("per_class_applied")
        ),
        "real_imu_calibration_reference_path": (
            None if calibration_report is None else calibration_report.get("reference_path")
        ),
        "real_imu_calibration_matched_sensor_names": (
            [] if calibration_report is None else list(calibration_report.get("matched_sensor_names", []))
        ),
        "real_imu_calibration_mean_abs_delta": (
            None if calibration_report is None else calibration_report.get("mean_abs_delta")
        ),
        "real_imu_calibration_max_abs_delta": (
            None if calibration_report is None else calibration_report.get("max_abs_delta")
        ),
        "max_acceleration_norm_m_s2": float(max_acc_norm),
        "max_gyro_norm_rad_s": float(max_gyro_norm),
        "assumptions": [
            "sensor_layout_matches_target_dataset",
            "imu_specific_force_uses_world_gravity_subtraction",
            "sensor_orientation_follows_attached_segment_joint",
            "real_imu_calibration_is_optional_and_reference_driven",
        ],
        "limitations": [
            "kinematic_adapter_not_full_imusim_dynamics",
            "no_contact_or_soft_tissue_model",
            "noise_model_is_simple_gaussian",
        ],
        "notes": list(dict.fromkeys(notes)),
    }
