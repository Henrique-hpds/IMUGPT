from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pose_module.interfaces import PoseSequence3D

_EPSILON = 1e-8
_VALID_IMU_FEATURE_MODES = {"acc_gyro", "acc_euler"}

SENSOR_TO_SEGMENT_JOINTS: dict[str, tuple[str, str]] = {
    "waist": ("Pelvis", "Spine3"),
    "head": ("Head", "Neck"),
    "right_forearm": ("Right_wrist", "Right_elbow"),
    "left_forearm": ("Left_wrist", "Left_elbow"),
}

def load_pose_sequence3d(pose3d_npz_path: str) -> PoseSequence3D:
    with np.load(pose3d_npz_path, allow_pickle=False) as payload:
        return PoseSequence3D.from_npz_payload({key: payload[key] for key in payload.files})

def temporal_derivative(values: np.ndarray, timestamps_sec: np.ndarray) -> np.ndarray:
    block = np.asarray(values, dtype=np.float64)
    timestamps = np.asarray(timestamps_sec, dtype=np.float64)
    if block.shape[0] != timestamps.shape[0]:
        raise ValueError("values and timestamps_sec must share the same temporal dimension.")
    derivative = np.gradient(block, timestamps, axis=0, edge_order=1)
    return np.asarray(derivative, dtype=np.float32)

def compute_bone_vectors(joint_positions_xyz: np.ndarray, skeleton_parents: Sequence[int]) -> np.ndarray:
    positions = np.asarray(joint_positions_xyz, dtype=np.float32)
    parents = [int(parent) for parent in skeleton_parents]
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("joint_positions_xyz must have shape [T, J, 3].")
    if positions.shape[1] != len(parents):
        raise ValueError("skeleton_parents must match the joint dimension in joint_positions_xyz.")

    bone_vectors = np.zeros_like(positions, dtype=np.float32)
    for joint_index, parent_index in enumerate(parents):
        if parent_index < 0:
            continue
        bone_vectors[:, joint_index, :] = positions[:, joint_index, :] - positions[:, parent_index, :]
    return bone_vectors

def compute_joint_angle_feature(joint_positions_xyz: np.ndarray, skeleton_parents: Sequence[int]) -> np.ndarray:
    positions = np.asarray(joint_positions_xyz, dtype=np.float32)
    parents = [int(parent) for parent in skeleton_parents]
    children_by_parent: dict[int, list[int]] = {}
    for joint_index, parent_index in enumerate(parents):
        if parent_index >= 0:
            children_by_parent.setdefault(parent_index, []).append(joint_index)

    angle_feature = np.zeros((positions.shape[0], positions.shape[1], 1), dtype=np.float32)
    for joint_index, parent_index in enumerate(parents):
        child_indices = children_by_parent.get(joint_index, [])
        if parent_index < 0 or len(child_indices) == 0:
            continue

        parent_vector = positions[:, parent_index, :] - positions[:, joint_index, :]
        child_vector = np.mean(
            np.stack([positions[:, child_index, :] - positions[:, joint_index, :] for child_index in child_indices], axis=0),
            axis=0
        )
        
        parent_norm = np.linalg.norm(parent_vector, axis=1)
        child_norm = np.linalg.norm(child_vector, axis=1)
        valid_mask = (parent_norm > _EPSILON) & (child_norm > _EPSILON)
        if not np.any(valid_mask):
            continue
        cosine = np.sum(parent_vector[valid_mask] * child_vector[valid_mask], axis=1) / (parent_norm[valid_mask] * child_norm[valid_mask])
        cosine = np.clip(cosine, -1.0, 1.0)
        angle_feature[valid_mask, joint_index, 0] = np.arccos(cosine).astype(np.float32, copy=False)
    return angle_feature

def build_pose_feature_tensor(sequence: PoseSequence3D) -> dict[str, Any]:
    positions = np.asarray(sequence.joint_positions_xyz, dtype=np.float32)
    timestamps_sec = np.asarray(sequence.timestamps_sec, dtype=np.float32)
    velocity = temporal_derivative(positions, timestamps_sec)
    acceleration = temporal_derivative(velocity, timestamps_sec)
    bone_vectors = compute_bone_vectors(positions, sequence.skeleton_parents)
    joint_angles = compute_joint_angle_feature(positions, sequence.skeleton_parents)
    confidence = np.asarray(sequence.joint_confidence, dtype=np.float32)[:, :, None]
    observed_mask = sequence.resolved_observed_mask().astype(np.float32)[:, :, None]
    imputed_mask = sequence.resolved_imputed_mask().astype(np.float32)[:, :, None]

    feature_tensor = np.concatenate(
        [
            positions,
            velocity,
            acceleration,
            confidence,
            bone_vectors,
            joint_angles,
            observed_mask,
            imputed_mask,
        ], axis=2).astype(np.float32, copy=False)
    channel_names = [
        "pos_x",
        "pos_y",
        "pos_z",
        "vel_x",
        "vel_y",
        "vel_z",
        "acc_x",
        "acc_y",
        "acc_z",
        "confidence",
        "bone_x",
        "bone_y",
        "bone_z",
        "joint_angle",
        "observed_mask",
        "imputed_mask"
    ]
    
    return {
        "values": feature_tensor,
        "channel_names": channel_names,
        "timestamps_sec": timestamps_sec,
        "joint_names": [str(value) for value in sequence.joint_names_3d],
        "skeleton_parents": [int(value) for value in sequence.skeleton_parents]
    }

def _segment_angular_velocity(anchor_positions_xyz: np.ndarray, secondary_positions_xyz: np.ndarray, timestamps_sec: np.ndarray) -> np.ndarray:
    anchor_positions = np.asarray(anchor_positions_xyz, dtype=np.float32)
    secondary_positions = np.asarray(secondary_positions_xyz, dtype=np.float32)
    timestamps = np.asarray(timestamps_sec, dtype=np.float32)
    segment = anchor_positions - secondary_positions
    segment_norm = np.linalg.norm(segment, axis=1, keepdims=True)
    unit_segment = segment / np.maximum(segment_norm, _EPSILON)

    angular_velocity = np.zeros_like(unit_segment, dtype=np.float32)
    for frame_index in range(1, unit_segment.shape[0]):
        previous = unit_segment[frame_index - 1]
        current = unit_segment[frame_index]
        cross = np.cross(previous, current)
        cross_norm = float(np.linalg.norm(cross))
        if cross_norm <= _EPSILON:
            continue
        axis = cross / cross_norm
        dot = float(np.clip(np.dot(previous, current), -1.0, 1.0))
        delta_angle = float(np.arctan2(cross_norm, dot))
        delta_time_sec = float(max(timestamps[frame_index] - timestamps[frame_index - 1], _EPSILON))
        angular_velocity[frame_index] = axis * (delta_angle / delta_time_sec)
    return angular_velocity.astype(np.float32, copy=False)

def build_pose_sensor_proxy(sequence: PoseSequence3D, *, selected_sensors: Sequence[str] | None = None) -> dict[str, Any]:
    joint_names = [str(value) for value in sequence.joint_names_3d]
    joint_index_by_name = {joint_name: joint_index for joint_index, joint_name in enumerate(joint_names)}
    sensor_names = (
        [str(sensor_name) for sensor_name in selected_sensors] if selected_sensors is not None
        else list(SENSOR_TO_SEGMENT_JOINTS.keys())
    )
    
    missing_sensors = [sensor_name for sensor_name in sensor_names if sensor_name not in SENSOR_TO_SEGMENT_JOINTS]
    if len(missing_sensors) > 0:
        raise ValueError(f"Unsupported sensor names for pose proxy extraction: {missing_sensors}")

    positions = np.asarray(sequence.joint_positions_xyz, dtype=np.float32)
    timestamps_sec = np.asarray(sequence.timestamps_sec, dtype=np.float32)
    velocity = temporal_derivative(positions, timestamps_sec)
    acceleration = temporal_derivative(velocity, timestamps_sec)

    acc_proxy = np.zeros((positions.shape[0], len(sensor_names), 3), dtype=np.float32)
    gyro_proxy = np.zeros_like(acc_proxy)
    
    for sensor_index, sensor_name in enumerate(sensor_names):
        anchor_name, secondary_name = SENSOR_TO_SEGMENT_JOINTS[sensor_name]
        
        if anchor_name not in joint_index_by_name or secondary_name not in joint_index_by_name:
            raise ValueError(
                f"Pose sequence does not expose the joints required for sensor '{sensor_name}': "
                f"{anchor_name}, {secondary_name}"
            )
            
        anchor_index = joint_index_by_name[anchor_name]
        secondary_index = joint_index_by_name[secondary_name]
        acc_proxy[:, sensor_index, :] = acceleration[:, anchor_index, :]
        gyro_proxy[:, sensor_index, :] = _segment_angular_velocity(
            positions[:, anchor_index, :],
            positions[:, secondary_index, :],
            timestamps_sec
        )

    return {
        "timestamps_sec": timestamps_sec,
        "sensor_names": sensor_names,
        "acc": acc_proxy,
        "gyro": gyro_proxy
    }

def build_motion_summary_signal(acc_xyz: np.ndarray, gyro_xyz: np.ndarray) -> np.ndarray:
    acc = np.asarray(acc_xyz, dtype=np.float32)
    gyro = np.asarray(gyro_xyz, dtype=np.float32)
    
    if acc.shape != gyro.shape:
        raise ValueError("acc_xyz and gyro_xyz must share the same shape.")
    if acc.ndim != 3 or acc.shape[2] != 3:
        raise ValueError("acc_xyz and gyro_xyz must have shape [T, S, 3].")

    acc_norm = np.linalg.norm(acc, axis=2)
    gyro_norm = np.linalg.norm(gyro, axis=2)
    summary = np.concatenate([acc_norm, gyro_norm], axis=1)
    centered = summary - np.mean(summary, axis=0, keepdims=True)
    std = np.std(centered, axis=0, keepdims=True)
    standardized = centered / np.maximum(std, _EPSILON)
    return np.mean(standardized, axis=1).astype(np.float32, copy=False)

def compute_euler_angles_from_acceleration(acc_xyz: np.ndarray) -> np.ndarray:
    acc = np.asarray(acc_xyz, dtype=np.float32)
    if acc.ndim != 3 or acc.shape[2] != 3:
        raise ValueError("acc_xyz must have shape [T, S, 3].")

    acc_x = acc[:, :, 0]
    acc_y = acc[:, :, 1]
    acc_z = acc[:, :, 2]

    theta = np.degrees(np.arctan(acc_x / np.maximum(np.sqrt((acc_y ** 2) + (acc_z ** 2)), _EPSILON)))
    psi = np.degrees(np.arctan(acc_y / np.maximum(np.sqrt((acc_x ** 2) + (acc_z ** 2)), _EPSILON)))
    phi = np.degrees(np.arctan(acc_z / np.maximum(np.sqrt((acc_x ** 2) + (acc_y ** 2)), _EPSILON)))
    return np.stack([theta, psi, phi], axis=2).astype(np.float32, copy=False)

def resolve_imu_orientation_features(acc_xyz: np.ndarray, gyro_xyz: np.ndarray, *, feature_mode: str = "acc_gyro") -> dict[str, Any]:
    acc = np.asarray(acc_xyz, dtype=np.float32)
    gyro = np.asarray(gyro_xyz, dtype=np.float32)
    normalized_feature_mode = str(feature_mode).strip().lower()
    
    if normalized_feature_mode not in _VALID_IMU_FEATURE_MODES:
        raise ValueError(
            "feature_mode must be one of "
            f"{sorted(_VALID_IMU_FEATURE_MODES)}."
        )
    
    if acc.shape != gyro.shape:
        raise ValueError("acc_xyz and gyro_xyz must have the same shape.")
    if acc.ndim != 3 or acc.shape[2] != 3:
        raise ValueError("acc_xyz and gyro_xyz must have shape [T, S, 3].")

    if normalized_feature_mode == "acc_gyro":
        orientation_block = gyro
        orientation_channel_names = [
            "gyro_x",
            "gyro_y",
            "gyro_z"
        ]
        
        orientation_norm_name = "gyro_norm"
    else:
        orientation_block = compute_euler_angles_from_acceleration(acc)
        orientation_channel_names = [
            "euler_theta_deg",
            "euler_psi_deg",
            "euler_phi_deg"
        ]
        orientation_norm_name = "euler_norm_deg"

    return {
        "values": np.asarray(orientation_block, dtype=np.float32),
        "norm": np.linalg.norm(orientation_block, axis=2, keepdims=True).astype(np.float32, copy=False),
        "channel_names": orientation_channel_names,
        "norm_name": orientation_norm_name,
        "feature_mode": normalized_feature_mode
    }

def build_imu_feature_tensor(acc_xyz: np.ndarray, gyro_xyz: np.ndarray, timestamps_sec: np.ndarray, *, feature_mode: str = "acc_gyro") -> dict[str, Any]:
    acc = np.asarray(acc_xyz, dtype=np.float32)
    timestamps = np.asarray(timestamps_sec, dtype=np.float32)
    orientation = resolve_imu_orientation_features(acc_xyz, gyro_xyz, feature_mode=feature_mode)

    if acc.shape[0] != timestamps.shape[0]:
        raise ValueError("timestamps_sec must match the temporal dimension of acc_xyz and gyro_xyz.")

    jerk = temporal_derivative(acc, timestamps)
    acc_norm = np.linalg.norm(acc, axis=2, keepdims=True)
    jerk_norm = np.linalg.norm(jerk, axis=2, keepdims=True)

    feature_tensor = np.concatenate(
        [acc, orientation["values"], jerk, acc_norm, orientation["norm"], jerk_norm],
        axis=2
    ).astype(np.float32, copy=False)
    
    channel_names = [
        "acc_x",
        "acc_y",
        "acc_z",
        *orientation["channel_names"],
        "jerk_x",
        "jerk_y",
        "jerk_z",
        "acc_norm",
        orientation["norm_name"],
        "jerk_norm"
    ]
    
    return {
        "values": feature_tensor,
        "channel_names": channel_names,
        "timestamps_sec": timestamps,
        "feature_mode": str(orientation["feature_mode"])
    }

def extract_quality_vector(quality_report: dict[str, Any] | None, *, pose_imu_alignment: dict[str, Any] | None = None, imu_feature_mode: str = "acc_gyro") -> dict[str, Any]:
    report = {} if quality_report is None else dict(quality_report)
    alignment = {} if pose_imu_alignment is None else dict(pose_imu_alignment)
    normalized_feature_mode = str(imu_feature_mode).strip().lower()
    
    if normalized_feature_mode not in _VALID_IMU_FEATURE_MODES:
        raise ValueError(
            "imu_feature_mode must be one of "
            f"{sorted(_VALID_IMU_FEATURE_MODES)}."
        )

    feature_items: list[tuple[str, float]] = [
        ("visible_joint_ratio", float(report.get("visible_joint_ratio", 0.0) or 0.0)),
        ("mean_confidence", float(report.get("mean_confidence", 0.0) or 0.0)),
        ("temporal_jitter_score", float(report.get("temporal_jitter_score", 0.0) or 0.0)),
        ("root_drift_score", float(report.get("root_drift_score", 0.0) or 0.0)),
        ("geometric_alignment_mean_acc_corr_after", float(report.get("geometric_alignment_mean_acc_corr_after", 0.0) or 0.0)),
        ("pose_imu_correlation_after_dtw", float(alignment.get("correlation_after_dtw", 0.0) or 0.0)),
        ("pose_imu_dtw_normalized_distance", float(alignment.get("dtw_normalized_distance", 0.0) or 0.0))
    ]
    
    if normalized_feature_mode == "acc_gyro":
        feature_items.insert(5, ("geometric_alignment_mean_gyro_corr_after", float(report.get("geometric_alignment_mean_gyro_corr_after", 0.0) or 0.0)))

    values = np.asarray([value for _, value in feature_items], dtype=np.float32)
    return {
        "values": values,
        "feature_names": [name for name, _ in feature_items],
    }