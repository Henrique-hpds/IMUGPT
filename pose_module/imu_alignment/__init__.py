"""Public API for the geometric IMU alignment module."""

from __future__ import annotations

from .apply import apply_sensor_subject_transform, apply_transforms_to_imu_sequence
from .fit import AlignmentFittingError, fit_sensor_subject_transforms
from .interfaces import (
    AlignmentConfig,
    AlignmentResult,
    IMUSequence,
    SensorSubjectTransform,
    build_identity_transform,
)
from .io_utils import (
    DEFAULT_ALIGNMENT_CONFIG_PATH,
    build_alignment_config,
    load_alignment_runtime_settings,
    load_real_imu_sequence,
    load_transforms_json,
    save_transforms_json,
    sequence_from_virtual_imu,
    split_sequences_by_capture_ids,
    virtual_from_imu_sequence,
)
from .metrics import (
    aggregate_alignment_results,
    compute_axiswise_corr,
    compute_axiswise_rmse,
    compute_norm_error,
    compute_vector_angle_error,
    summarize_alignment_metrics,
)
from .pipeline_adapter import run_geometric_alignment
from .rotation import apply_rotation, estimate_rotation_procrustes
from .temporal import align_streams_with_lag, estimate_time_lag

__all__ = [
    "AlignmentConfig",
    "AlignmentFittingError",
    "AlignmentResult",
    "DEFAULT_ALIGNMENT_CONFIG_PATH",
    "IMUSequence",
    "SensorSubjectTransform",
    "aggregate_alignment_results",
    "align_streams_with_lag",
    "apply_rotation",
    "apply_sensor_subject_transform",
    "apply_transforms_to_imu_sequence",
    "build_alignment_config",
    "build_identity_transform",
    "compute_axiswise_corr",
    "compute_axiswise_rmse",
    "compute_norm_error",
    "compute_vector_angle_error",
    "estimate_rotation_procrustes",
    "estimate_time_lag",
    "fit_sensor_subject_transforms",
    "load_alignment_runtime_settings",
    "load_real_imu_sequence",
    "load_transforms_json",
    "run_geometric_alignment",
    "save_transforms_json",
    "sequence_from_virtual_imu",
    "split_sequences_by_capture_ids",
    "summarize_alignment_metrics",
    "virtual_from_imu_sequence"
]
