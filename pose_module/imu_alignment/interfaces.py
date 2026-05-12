"""Core contracts for geometric alignment between virtual and real IMU streams."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

_ROTATION_DET_TOL = 1e-3


@dataclass
class IMUSequence:
    """Canonical IMU container used by the alignment package."""

    subject_id: str
    capture_id: str
    sensor_names: List[str]
    fps: int | float
    timestamps: np.ndarray
    acc: np.ndarray
    gyro: np.ndarray

    def __post_init__(self) -> None:
        self.subject_id = str(self.subject_id)
        self.capture_id = str(self.capture_id)
        self.sensor_names = [str(name) for name in self.sensor_names]
        self.timestamps = np.asarray(self.timestamps, dtype=np.float32)
        self.acc = np.asarray(self.acc, dtype=np.float32)
        self.gyro = np.asarray(self.gyro, dtype=np.float32)
        self.validate()

    @property
    def num_frames(self) -> int:
        return int(self.acc.shape[0])

    @property
    def num_sensors(self) -> int:
        return int(self.acc.shape[1])

    def validate(self) -> None:
        if self.timestamps.ndim != 1:
            raise ValueError("IMUSequence.timestamps must have shape [T].")
        if self.acc.ndim != 3 or self.acc.shape[-1] != 3:
            raise ValueError("IMUSequence.acc must have shape [T, S, 3].")
        if self.gyro.ndim != 3 or self.gyro.shape[-1] != 3:
            raise ValueError("IMUSequence.gyro must have shape [T, S, 3].")
        if self.acc.shape != self.gyro.shape:
            raise ValueError("IMUSequence.acc and IMUSequence.gyro must share the same shape.")
        if self.acc.shape[0] != self.timestamps.shape[0]:
            raise ValueError("IMUSequence timestamps and signals must share the same frame count.")
        if self.acc.shape[1] != len(self.sensor_names):
            raise ValueError("IMUSequence.sensor_names length must match the signal sensor dimension.")
        if self.num_frames <= 0:
            raise ValueError("IMUSequence must contain at least one frame.")
        if self.num_sensors <= 0:
            raise ValueError("IMUSequence must contain at least one sensor.")


@dataclass
class AlignmentConfig:
    """Configurable fitting and application options for geometric alignment."""

    estimate_time_lag: bool = True
    max_lag_samples: int = 20
    lag_signal: str = "gyro_norm"
    use_acc: bool = True
    use_gyro: bool = True
    use_bias: bool = False
    use_scale: bool = False
    gravity_window_only: bool = False
    min_motion_norm_gyro: float = 0.1
    min_motion_norm_acc: float = 0.2
    ridge_lambda: float = 1e-6

    def validate(self) -> None:
        if self.lag_signal not in {"gyro_norm", "acc_norm"}:
            raise ValueError("AlignmentConfig.lag_signal must be 'gyro_norm' or 'acc_norm'.")
        if int(self.max_lag_samples) < 0:
            raise ValueError("AlignmentConfig.max_lag_samples must be non-negative.")
        if not bool(self.use_acc) and not bool(self.use_gyro):
            raise ValueError("AlignmentConfig must enable at least one of use_acc or use_gyro.")
        if float(self.min_motion_norm_gyro) < 0.0:
            raise ValueError("AlignmentConfig.min_motion_norm_gyro must be non-negative.")
        if float(self.min_motion_norm_acc) < 0.0:
            raise ValueError("AlignmentConfig.min_motion_norm_acc must be non-negative.")
        if float(self.ridge_lambda) < 0.0:
            raise ValueError("AlignmentConfig.ridge_lambda must be non-negative.")


@dataclass
class SensorSubjectTransform:
    """Learned per-subject, per-sensor transformation."""

    subject_id: str
    sensor_name: str
    rotation: np.ndarray
    acc_bias: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    acc_scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))
    gyro_scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))
    fitted_capture_ids: List[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.subject_id = str(self.subject_id)
        self.sensor_name = str(self.sensor_name)
        self.rotation = np.asarray(self.rotation, dtype=np.float32)
        self.acc_bias = np.asarray(self.acc_bias, dtype=np.float32)
        self.gyro_bias = np.asarray(self.gyro_bias, dtype=np.float32)
        self.acc_scale = np.asarray(self.acc_scale, dtype=np.float32)
        self.gyro_scale = np.asarray(self.gyro_scale, dtype=np.float32)
        self.fitted_capture_ids = [str(value) for value in self.fitted_capture_ids]
        self.diagnostics = dict(self.diagnostics)
        self.validate()

    def validate(self) -> None:
        if self.rotation.shape != (3, 3):
            raise ValueError("SensorSubjectTransform.rotation must have shape [3, 3].")
        if not np.isfinite(self.rotation).all():
            raise ValueError("SensorSubjectTransform.rotation contains NaN or infinite values.")
        det_rotation = float(np.linalg.det(self.rotation))
        if det_rotation < 1.0 - _ROTATION_DET_TOL:
            raise ValueError(
                "SensorSubjectTransform.rotation must be a proper rotation with det(R) close to +1."
            )
        for name, value in (
            ("acc_bias", self.acc_bias),
            ("gyro_bias", self.gyro_bias),
            ("acc_scale", self.acc_scale),
            ("gyro_scale", self.gyro_scale),
        ):
            if value.shape != (3,):
                raise ValueError(f"SensorSubjectTransform.{name} must have shape [3].")
            if not np.isfinite(value).all():
                raise ValueError(f"SensorSubjectTransform.{name} contains NaN or infinite values.")


@dataclass
class AlignmentResult:
    """Per-capture, per-sensor application result used for evaluation."""

    subject_id: str
    capture_id: str
    sensor_name: str
    lag_samples: int
    acc_aligned: np.ndarray
    gyro_aligned: np.ndarray
    metrics_before: Dict[str, Any]
    metrics_after: Dict[str, Any]

    def __post_init__(self) -> None:
        self.subject_id = str(self.subject_id)
        self.capture_id = str(self.capture_id)
        self.sensor_name = str(self.sensor_name)
        self.lag_samples = int(self.lag_samples)
        self.acc_aligned = np.asarray(self.acc_aligned, dtype=np.float32)
        self.gyro_aligned = np.asarray(self.gyro_aligned, dtype=np.float32)
        self.metrics_before = dict(self.metrics_before)
        self.metrics_after = dict(self.metrics_after)
        self.validate()

    def validate(self) -> None:
        if self.acc_aligned.ndim != 2 or self.acc_aligned.shape[-1] != 3:
            raise ValueError("AlignmentResult.acc_aligned must have shape [T, 3].")
        if self.gyro_aligned.ndim != 2 or self.gyro_aligned.shape[-1] != 3:
            raise ValueError("AlignmentResult.gyro_aligned must have shape [T, 3].")
        if self.acc_aligned.shape[0] != self.gyro_aligned.shape[0]:
            raise ValueError("AlignmentResult.acc_aligned and gyro_aligned must share the same frame count.")


def build_identity_transform(
    *,
    subject_id: str,
    sensor_name: str,
    fitted_capture_ids: List[str] | None = None,
    diagnostics: Dict[str, Any] | None = None
) -> SensorSubjectTransform:
    """Build a neutral transform for a subject/sensor pair."""

    return SensorSubjectTransform(
        subject_id=str(subject_id),
        sensor_name=str(sensor_name),
        rotation=np.eye(3, dtype=np.float32),
        acc_bias=np.zeros(3, dtype=np.float32),
        gyro_bias=np.zeros(3, dtype=np.float32),
        acc_scale=np.ones(3, dtype=np.float32),
        gyro_scale=np.ones(3, dtype=np.float32),
        fitted_capture_ids=[] if fitted_capture_ids is None else list(fitted_capture_ids),
        diagnostics={} if diagnostics is None else dict(diagnostics),
    )
