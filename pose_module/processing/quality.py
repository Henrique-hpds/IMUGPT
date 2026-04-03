"""Quality-report helpers for the pose pipeline."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def merge_pose2d_quality_reports(
    *,
    clip_id: str,
    backend_quality: Mapping[str, Any],
    track_report: Mapping[str, Any],
    pose_quality: Mapping[str, Any],
    cleaner_quality: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    cleaner_quality = {} if cleaner_quality is None else dict(cleaner_quality)
    notes = []
    notes.extend([str(value) for value in backend_quality.get("warnings", [])])
    notes.extend([str(value) for value in track_report.get("warnings", [])])
    notes.extend([str(value) for value in pose_quality.get("notes", [])])
    notes.extend([str(value) for value in cleaner_quality.get("notes", [])])

    status = "ok"
    if (
        track_report.get("status") == "fail"
        or pose_quality.get("status") == "fail"
        or cleaner_quality.get("status") == "fail"
    ):
        status = "fail"
    elif (
        track_report.get("status") == "warning"
        or pose_quality.get("status") == "warning"
        or cleaner_quality.get("status") == "warning"
        or len(notes) > 0
    ):
        status = "warning"

    return {
        "clip_id": str(clip_id),
        "status": str(status),
        "fps_original": backend_quality.get("fps_original", pose_quality.get("fps_original")),
        "fps": backend_quality.get(
            "effective_fps",
            cleaner_quality.get("fps", pose_quality.get("fps")),
        ),
        "frames_total": backend_quality.get("frames_total"),
        "frames_selected": backend_quality.get("frames_selected", pose_quality.get("num_selected_frames")),
        "frames_with_detections": track_report.get("frames_with_detections"),
        "frames_with_selected_track": pose_quality.get("frames_with_selected_track"),
        "selected_track_id": track_report.get("selected_track_id"),
        "selected_track_stability": track_report.get("selected_track_stability"),
        "visible_joint_ratio": cleaner_quality.get("visible_joint_ratio", pose_quality.get("visible_joint_ratio")),
        "mean_confidence": cleaner_quality.get("mean_confidence", pose_quality.get("mean_confidence")),
        "temporal_jitter_score": cleaner_quality.get("temporal_jitter_score"),
        "outlier_ratio": cleaner_quality.get("outlier_ratio"),
        "frames_interpolated": cleaner_quality.get("frames_interpolated"),
        "interpolated_joint_ratio": cleaner_quality.get("interpolated_joint_ratio"),
        "frames_over_missing_joint_threshold": cleaner_quality.get("frames_over_missing_joint_threshold"),
        "normalization_mode": cleaner_quality.get("normalization_mode"),
        "notes": list(dict.fromkeys(notes)),
    }


def merge_motionbert_quality_reports(
    *,
    pose2d_quality: Mapping[str, Any],
    lifter_quality: Mapping[str, Any],
) -> Dict[str, Any]:
    pose2d_quality = dict(pose2d_quality)
    lifter_quality = dict(lifter_quality)

    notes = []
    notes.extend([str(value) for value in pose2d_quality.get("notes", [])])
    notes.extend([str(value) for value in lifter_quality.get("notes", [])])

    status = "ok"
    if pose2d_quality.get("status") == "fail" or lifter_quality.get("status") == "fail":
        status = "fail"
    elif (
        pose2d_quality.get("status") == "warning"
        or lifter_quality.get("status") == "warning"
        or len(notes) > 0
    ):
        status = "warning"

    merged = dict(pose2d_quality)
    merged.update(
        {
            "status": str(status),
            "pose3d_num_frames": lifter_quality.get("num_frames"),
            "pose3d_num_joints": lifter_quality.get("num_joints"),
            "pose3d_coordinate_space": lifter_quality.get("coordinate_space"),
            "pose3d_joint_format": lifter_quality.get("output_joint_format"),
            "motionbert_backend_name": lifter_quality.get("backend_name"),
            "motionbert_window_size": lifter_quality.get("window_size"),
            "motionbert_window_overlap": lifter_quality.get("window_overlap"),
            "motionbert_num_windows": lifter_quality.get("num_windows"),
            "motionbert_input_channels": lifter_quality.get("input_channels"),
            "depth_variation": lifter_quality.get("depth_variation"),
            "window_coverage_ratio": lifter_quality.get("window_coverage_ratio"),
            "notes": list(dict.fromkeys(notes)),
        }
    )
    return merged


def merge_pose3d_mapping_quality_reports(
    *,
    pose2d_quality: Mapping[str, Any],
    lifter_quality: Mapping[str, Any],
    mapper_quality: Mapping[str, Any],
) -> Dict[str, Any]:
    pose2d_quality = dict(pose2d_quality)
    lifter_quality = dict(lifter_quality)
    mapper_quality = dict(mapper_quality)

    notes = []
    notes.extend([str(value) for value in pose2d_quality.get("notes", [])])
    notes.extend([str(value) for value in lifter_quality.get("notes", [])])
    notes.extend([str(value) for value in mapper_quality.get("notes", [])])

    status = "ok"
    if (
        pose2d_quality.get("status") == "fail"
        or lifter_quality.get("status") == "fail"
        or mapper_quality.get("status") == "fail"
    ):
        status = "fail"
    elif (
        pose2d_quality.get("status") == "warning"
        or lifter_quality.get("status") == "warning"
        or mapper_quality.get("status") == "warning"
        or len(notes) > 0
    ):
        status = "warning"

    merged = dict(pose2d_quality)
    merged.update(
        {
            "status": str(status),
            "pose3d_num_frames": mapper_quality.get("num_frames", lifter_quality.get("num_frames")),
            "pose3d_num_joints": mapper_quality.get("num_joints", lifter_quality.get("num_joints")),
            "pose3d_coordinate_space": mapper_quality.get(
                "coordinate_space",
                lifter_quality.get("coordinate_space"),
            ),
            "pose3d_joint_format": mapper_quality.get(
                "output_joint_format",
                lifter_quality.get("output_joint_format"),
            ),
            "motionbert_output_joint_format": lifter_quality.get("output_joint_format"),
            "motionbert_backend_name": lifter_quality.get("backend_name"),
            "motionbert_window_size": lifter_quality.get("window_size"),
            "motionbert_window_overlap": lifter_quality.get("window_overlap"),
            "motionbert_num_windows": lifter_quality.get("num_windows"),
            "motionbert_input_channels": lifter_quality.get("input_channels"),
            "depth_variation": lifter_quality.get("depth_variation"),
            "window_coverage_ratio": lifter_quality.get("window_coverage_ratio"),
            "skeleton_mapping_ok": mapper_quality.get("skeleton_mapping_ok"),
            "skeleton_mapper_input_joint_format": mapper_quality.get("input_joint_format"),
            "skeleton_mapper_output_joint_format": mapper_quality.get("output_joint_format"),
            "handedness_swapped_frames": mapper_quality.get("handedness_swapped_frames"),
            "forward_vector_fallback_frames": mapper_quality.get("forward_vector_fallback_frames"),
            "notes": list(dict.fromkeys(notes)),
        }
    )
    return merged


def merge_metric_pose_quality_reports(
    *,
    pose2d_quality: Mapping[str, Any],
    lifter_quality: Mapping[str, Any],
    lower_limb_quality: Mapping[str, Any] | None = None,
    mapper_quality: Mapping[str, Any],
    normalizer_quality: Mapping[str, Any],
) -> Dict[str, Any]:
    pose2d_quality = dict(pose2d_quality)
    lifter_quality = dict(lifter_quality)
    lower_limb_quality = {} if lower_limb_quality is None else dict(lower_limb_quality)
    mapper_quality = dict(mapper_quality)
    normalizer_quality = dict(normalizer_quality)

    notes = []
    notes.extend([str(value) for value in pose2d_quality.get("notes", [])])
    notes.extend([str(value) for value in lifter_quality.get("notes", [])])
    notes.extend([str(value) for value in lower_limb_quality.get("notes", [])])
    notes.extend([str(value) for value in mapper_quality.get("notes", [])])
    notes.extend([str(value) for value in normalizer_quality.get("notes", [])])

    status = "ok"
    if (
        pose2d_quality.get("status") == "fail"
        or lifter_quality.get("status") == "fail"
        or lower_limb_quality.get("status") == "fail"
        or mapper_quality.get("status") == "fail"
        or normalizer_quality.get("status") == "fail"
    ):
        status = "fail"
    elif (
        pose2d_quality.get("status") == "warning"
        or lifter_quality.get("status") == "warning"
        or lower_limb_quality.get("status") == "warning"
        or mapper_quality.get("status") == "warning"
        or normalizer_quality.get("status") == "warning"
        or len(notes) > 0
    ):
        status = "warning"

    merged = dict(pose2d_quality)
    merged.update(
        {
            "status": str(status),
            "pose3d_num_frames": normalizer_quality.get("num_frames", mapper_quality.get("num_frames")),
            "pose3d_num_joints": normalizer_quality.get("num_joints", mapper_quality.get("num_joints")),
            "pose3d_coordinate_space": normalizer_quality.get(
                "coordinate_space",
                mapper_quality.get("coordinate_space", lifter_quality.get("coordinate_space")),
            ),
            "pose3d_joint_format": normalizer_quality.get(
                "output_joint_format",
                mapper_quality.get("output_joint_format", lifter_quality.get("output_joint_format")),
            ),
            "motionbert_output_joint_format": lifter_quality.get("output_joint_format"),
            "motionbert_backend_name": lifter_quality.get("backend_name"),
            "motionbert_window_size": lifter_quality.get("window_size"),
            "motionbert_window_overlap": lifter_quality.get("window_overlap"),
            "motionbert_num_windows": lifter_quality.get("num_windows"),
            "motionbert_input_channels": lifter_quality.get("input_channels"),
            "depth_variation": lifter_quality.get("depth_variation"),
            "window_coverage_ratio": lifter_quality.get("window_coverage_ratio"),
            "lower_limb_left_correction_frames": lower_limb_quality.get("left_leg_correction_frames"),
            "lower_limb_right_correction_frames": lower_limb_quality.get("right_leg_correction_frames"),
            "lower_limb_left_uncertain_frames": lower_limb_quality.get("left_leg_uncertain_frames"),
            "lower_limb_right_uncertain_frames": lower_limb_quality.get("right_leg_uncertain_frames"),
            "skeleton_mapping_ok": mapper_quality.get("skeleton_mapping_ok"),
            "skeleton_mapper_input_joint_format": mapper_quality.get("input_joint_format"),
            "skeleton_mapper_output_joint_format": mapper_quality.get("output_joint_format"),
            "handedness_swapped_frames": mapper_quality.get("handedness_swapped_frames"),
            "forward_vector_fallback_frames": mapper_quality.get("forward_vector_fallback_frames"),
            "metric_pose_ok": normalizer_quality.get("metric_pose_ok"),
            "metric_normalizer_coordinate_space": normalizer_quality.get("coordinate_space"),
            "metric_normalizer_scale_factor": normalizer_quality.get("scale_factor"),
            "metric_normalizer_target_femur_length_m": normalizer_quality.get("target_femur_length_m"),
            "metric_normalizer_target_tibia_length_m": normalizer_quality.get("target_tibia_length_m"),
            "metric_normalizer_observed_femur_length_model_units": normalizer_quality.get(
                "observed_femur_length_model_units"
            ),
            "metric_normalizer_body_frame_fallback_frames": normalizer_quality.get(
                "body_frame_fallback_frames"
            ),
            "metric_normalizer_tibia_prior_applied_frames": normalizer_quality.get(
                "tibia_prior_applied_frames"
            ),
            "metric_normalizer_smoothing_window_length": normalizer_quality.get(
                "smoothing_window_length"
            ),
            "metric_normalizer_corrected_smoothing_window_length": normalizer_quality.get(
                "corrected_smoothing_window_length"
            ),
            "metric_normalizer_smoothing_polyorder": normalizer_quality.get("smoothing_polyorder"),
            "notes": list(dict.fromkeys(notes)),
        }
    )
    return merged


def merge_pose3d_quality_reports(
    *,
    pose2d_quality: Mapping[str, Any],
    lifter_quality: Mapping[str, Any],
    lower_limb_quality: Mapping[str, Any] | None = None,
    mapper_quality: Mapping[str, Any],
    normalizer_quality: Mapping[str, Any],
    root_quality: Mapping[str, Any],
) -> Dict[str, Any]:
    root_quality = dict(root_quality)
    merged = merge_metric_pose_quality_reports(
        pose2d_quality=pose2d_quality,
        lifter_quality=lifter_quality,
        lower_limb_quality=lower_limb_quality,
        mapper_quality=mapper_quality,
        normalizer_quality=normalizer_quality,
    )

    notes = []
    notes.extend([str(value) for value in merged.get("notes", [])])
    notes.extend([str(value) for value in root_quality.get("notes", [])])

    status = str(merged.get("status", "ok"))
    if root_quality.get("status") == "fail":
        status = "fail"
    elif root_quality.get("status") == "warning" and status != "fail":
        status = "warning"

    merged.update(
        {
            "status": str(status),
            "pose3d_num_frames": root_quality.get("num_frames", merged.get("pose3d_num_frames")),
            "pose3d_num_joints": root_quality.get("num_joints", merged.get("pose3d_num_joints")),
            "pose3d_coordinate_space": root_quality.get(
                "coordinate_space",
                merged.get("pose3d_coordinate_space"),
            ),
            "pose3d_joint_format": root_quality.get(
                "output_joint_format",
                merged.get("pose3d_joint_format"),
            ),
            "root_translation_ok": root_quality.get("root_translation_ok"),
            "root_estimator_coordinate_space": root_quality.get("coordinate_space"),
            "root_estimator_planarize_vertical": root_quality.get("planarize_vertical"),
            "root_estimator_scale_factor": root_quality.get("scale_factor"),
            "root_drift_score": root_quality.get("root_drift_score"),
            "root_path_length_m": root_quality.get("root_path_length_m"),
            "root_horizontal_path_length_m": root_quality.get("root_horizontal_path_length_m"),
            "root_vertical_span_m": root_quality.get("root_vertical_span_m"),
            "root_max_step_m": root_quality.get("root_max_step_m"),
            "root_mean_step_m": root_quality.get("root_mean_step_m"),
            "root_smoothing_window_length": root_quality.get("smoothing_window_length"),
            "root_smoothing_polyorder": root_quality.get("smoothing_polyorder"),
            "notes": list(dict.fromkeys(notes)),
        }
    )
    return merged


def merge_virtual_imu_quality_reports(
    *,
    pose3d_quality: Mapping[str, Any],
    ik_quality: Mapping[str, Any],
    virtual_imu_quality: Mapping[str, Any],
    geometric_alignment_quality: Mapping[str, Any] | None = None,
    frame_alignment_quality: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    pose3d_quality = dict(pose3d_quality)
    ik_quality = dict(ik_quality)
    virtual_imu_quality = dict(virtual_imu_quality)
    geometric_alignment_quality = (
        {} if geometric_alignment_quality is None else dict(geometric_alignment_quality)
    )
    frame_alignment_quality = {} if frame_alignment_quality is None else dict(frame_alignment_quality)

    notes = []
    notes.extend([str(value) for value in pose3d_quality.get("notes", [])])
    notes.extend([str(value) for value in ik_quality.get("notes", [])])
    notes.extend([str(value) for value in virtual_imu_quality.get("notes", [])])
    notes.extend([str(value) for value in geometric_alignment_quality.get("notes", [])])
    notes.extend([str(value) for value in frame_alignment_quality.get("notes", [])])

    status = "ok"
    if (
        pose3d_quality.get("status") == "fail"
        or ik_quality.get("status") == "fail"
        or virtual_imu_quality.get("status") == "fail"
        or geometric_alignment_quality.get("status") == "fail"
        or frame_alignment_quality.get("status") == "fail"
    ):
        status = "fail"
    elif (
        pose3d_quality.get("status") == "warning"
        or ik_quality.get("status") == "warning"
        or virtual_imu_quality.get("status") == "warning"
        or geometric_alignment_quality.get("status") == "warning"
        or frame_alignment_quality.get("status") == "warning"
        or len(notes) > 0
    ):
        status = "warning"

    merged = dict(pose3d_quality)
    merged.update(
        {
            "status": str(status),
            "ik_ok": ik_quality.get("ik_ok"),
            "ik_rotation_representation": ik_quality.get("rotation_representation"),
            "ik_reconstruction_error_mean_m": ik_quality.get("reconstruction_error_mean_m"),
            "ik_reconstruction_error_max_m": ik_quality.get("reconstruction_error_max_m"),
            "virtual_imu_ok": virtual_imu_quality.get("virtual_imu_ok"),
            "virtual_imu_num_sensors": virtual_imu_quality.get("num_sensors"),
            "virtual_imu_sensor_names": virtual_imu_quality.get("sensor_names"),
            "virtual_imu_acc_noise_std_m_s2": virtual_imu_quality.get("acc_noise_std_m_s2"),
            "virtual_imu_gyro_noise_std_rad_s": virtual_imu_quality.get("gyro_noise_std_rad_s"),
            "virtual_imu_real_calibration_applied": virtual_imu_quality.get("real_imu_calibration_applied"),
            "virtual_imu_real_calibration_signal_mode": virtual_imu_quality.get(
                "real_imu_calibration_signal_mode"
            ),
            "virtual_imu_real_calibration_per_class_applied": virtual_imu_quality.get(
                "real_imu_calibration_per_class_applied"
            ),
            "virtual_imu_real_calibration_reference_path": virtual_imu_quality.get(
                "real_imu_calibration_reference_path"
            ),
            "virtual_imu_real_calibration_matched_sensor_names": virtual_imu_quality.get(
                "real_imu_calibration_matched_sensor_names"
            ),
            "virtual_imu_real_calibration_mean_abs_delta": virtual_imu_quality.get(
                "real_imu_calibration_mean_abs_delta"
            ),
            "virtual_imu_real_calibration_max_abs_delta": virtual_imu_quality.get(
                "real_imu_calibration_max_abs_delta"
            ),
            "geometric_alignment_enabled": geometric_alignment_quality.get("enabled"),
            "geometric_alignment_status": geometric_alignment_quality.get("status"),
            "geometric_alignment_subject_id": geometric_alignment_quality.get("subject_id"),
            "geometric_alignment_capture_id": geometric_alignment_quality.get("capture_id"),
            "geometric_alignment_real_imu_npz_path": geometric_alignment_quality.get("real_imu_npz_path"),
            "geometric_alignment_transform_source": geometric_alignment_quality.get("transform_source"),
            "geometric_alignment_estimated_sensors": geometric_alignment_quality.get("estimated_sensor_names"),
            "geometric_alignment_mean_gyro_corr_before": geometric_alignment_quality.get(
                "mean_gyro_corr_before"
            ),
            "geometric_alignment_mean_gyro_corr_after": geometric_alignment_quality.get(
                "mean_gyro_corr_after"
            ),
            "geometric_alignment_mean_acc_corr_before": geometric_alignment_quality.get(
                "mean_acc_corr_before"
            ),
            "geometric_alignment_mean_acc_corr_after": geometric_alignment_quality.get(
                "mean_acc_corr_after"
            ),
            "sensor_frame_estimation_enabled": frame_alignment_quality.get("enabled"),
            "sensor_frame_estimation_status": frame_alignment_quality.get("status"),
            "sensor_frame_estimation_target_sensors": frame_alignment_quality.get("target_sensor_names"),
            "sensor_frame_estimation_estimated_sensors": frame_alignment_quality.get("estimated_sensor_names"),
            "sensor_frame_estimation_real_imu_npz_path": frame_alignment_quality.get("real_imu_npz_path"),
            "sensor_frame_estimation_mean_gyro_corr_before": frame_alignment_quality.get(
                "mean_gyro_corr_before"
            ),
            "sensor_frame_estimation_mean_gyro_corr_after": frame_alignment_quality.get(
                "mean_gyro_corr_after"
            ),
            "sensor_frame_estimation_mean_acc_corr_before": frame_alignment_quality.get(
                "mean_acc_corr_before"
            ),
            "sensor_frame_estimation_mean_acc_corr_after": frame_alignment_quality.get(
                "mean_acc_corr_after"
            ),
            "virtual_imu_max_acceleration_norm_m_s2": virtual_imu_quality.get("max_acceleration_norm_m_s2"),
            "virtual_imu_max_gyro_norm_rad_s": virtual_imu_quality.get("max_gyro_norm_rad_s"),
            "notes": list(dict.fromkeys(notes)),
        }
    )
    return merged
