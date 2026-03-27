import tempfile
import unittest
from pathlib import Path

import numpy as np

from pose_module.export.ik_adapter import run_ik
from pose_module.export.imusim_adapter import run_imusim
from pose_module.processing.metric_normalizer import run_metric_normalizer
from pose_module.processing.root_estimator import run_root_trajectory_estimator
from tests.test_metric_normalizer import _make_imugpt22_sequence


def _make_ik_sequence():
    mapped_sequence = _make_imugpt22_sequence(num_frames=12, yaw_rad=0.0)
    metric_result = run_metric_normalizer(
        mapped_sequence,
        target_femur_length_m=0.45,
        smoothing_window_length=5,
        smoothing_polyorder=2,
    )
    root_result = run_root_trajectory_estimator(
        metric_result["pose_sequence"],
        normalization_result=metric_result["normalization_result"],
        smoothing_window_length=5,
        smoothing_polyorder=2,
    )
    return run_ik(root_result["pose_sequence"])["ik_sequence"]


class IMUSimAdapterTests(unittest.TestCase):
    def test_run_imusim_exports_virtual_imu_artifacts(self) -> None:
        ik_sequence = _make_ik_sequence()

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_imusim(ik_sequence, output_dir=tmp_dir)

            sequence = result["virtual_imu_sequence"]
            self.assertEqual(sequence.sensor_names, ["waist", "head", "right_forearm", "left_forearm"])
            self.assertEqual(sequence.acc.shape, (12, 4, 3))
            self.assertEqual(sequence.gyro.shape, (12, 4, 3))
            self.assertTrue(Path(result["artifacts"]["virtual_imu_npz_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["virtual_imu_report_json_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["sensor_layout_resolved_json_path"]).exists())
            self.assertEqual(result["quality_report"]["status"], "ok")
            self.assertTrue(result["quality_report"]["virtual_imu_ok"])
            self.assertTrue(np.isfinite(sequence.acc).all())
            self.assertTrue(np.isfinite(sequence.gyro).all())
            self.assertGreater(float(np.median(np.linalg.norm(sequence.acc[:, 0], axis=1))), 5.0)

    def test_run_imusim_can_calibrate_with_real_imu_reference(self) -> None:
        ik_sequence = _make_ik_sequence()

        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as calibrated_dir:
            raw_result = run_imusim(ik_sequence, output_dir=raw_dir)
            raw_sequence = raw_result["virtual_imu_sequence"]
            reference_path = Path(raw_dir) / "real_imu_reference.npz"
            np.savez_compressed(
                reference_path,
                acc=(np.asarray(raw_sequence.acc, dtype=np.float32) * np.float32(0.35) + np.float32(1.5)),
                sensor_names=np.asarray(raw_sequence.sensor_names),
                y=np.asarray(["walking"] * raw_sequence.num_frames),
            )

            calibrated_result = run_imusim(
                ik_sequence,
                output_dir=calibrated_dir,
                real_imu_reference_path=reference_path,
                real_imu_activity_label="walking",
            )

            calibrated_sequence = calibrated_result["virtual_imu_sequence"]
            calibrated_raw_sequence = calibrated_result["raw_virtual_imu_sequence"]
            self.assertIsNotNone(calibrated_result["calibration_report"])
            self.assertTrue(calibrated_result["quality_report"]["real_imu_calibration_applied"])
            self.assertTrue(
                Path(calibrated_result["artifacts"]["virtual_imu_calibration_report_json_path"]).exists()
            )
            self.assertTrue(Path(calibrated_result["artifacts"]["virtual_imu_raw_npz_path"]).exists())
            self.assertFalse(np.allclose(calibrated_sequence.acc, calibrated_raw_sequence.acc))
            np.testing.assert_allclose(
                calibrated_sequence.gyro,
                calibrated_raw_sequence.gyro,
                atol=1e-6,
            )
