import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pose_module.export.ik_adapter import run_ik
from pose_module.export.imusim_adapter import run_imusim
from pose_module.interfaces import VirtualIMUSequence
from pose_module.processing.metric_normalizer import run_metric_normalizer
from pose_module.processing.imu_calibration import calibrate_virtual_imu_sequence
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

    def test_calibrate_virtual_imu_sequence_uses_robot_emotions_sensor_ids_for_legacy_reference_names(self) -> None:
        timestamps_sec = np.arange(240, dtype=np.float32) / np.float32(20.0)
        sensor_names = ["waist", "head", "right_forearm", "left_forearm"]
        acc = np.zeros((timestamps_sec.shape[0], len(sensor_names), 3), dtype=np.float32)
        gyro = np.zeros_like(acc)

        for sensor_index, sensor_phase in enumerate((0.0, 0.3, 0.7, 1.1)):
            acc[:, sensor_index, 0] = np.sin(0.8 * timestamps_sec + sensor_phase)
            acc[:, sensor_index, 1] = 9.81 + 0.3 * np.cos(0.5 * timestamps_sec + sensor_phase)
            acc[:, sensor_index, 2] = 0.2 * np.sin(1.1 * timestamps_sec + 0.5 * sensor_phase)

        virtual_sequence = VirtualIMUSequence(
            clip_id="synthetic_legacy_robot_emotions_reference",
            fps=20.0,
            sensor_names=sensor_names,
            acc=acc,
            gyro=gyro,
            timestamps_sec=timestamps_sec,
            source="synthetic_virtual",
        )

        # Real RobotEmotions clips are stored in sensor-id order [1, 2, 3, 4] which is
        # [waist, head, left_forearm, right_forearm] after the corrected mapping.
        real_order_acc = acc[:, [0, 1, 3, 2], :]
        reference_scale = np.asarray([0.4, 0.6, 1.1, 1.8], dtype=np.float32)
        reference_bias = np.asarray([0.5, 1.5, 3.0, 7.0], dtype=np.float32)
        reference_acc = (
            real_order_acc * reference_scale[None, :, None] + reference_bias[None, :, None]
        ).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            reference_path = Path(tmp_dir) / "real_imu_reference.npz"
            np.savez_compressed(
                reference_path,
                acc=reference_acc,
                sensor_ids=np.asarray([1, 2, 3, 4], dtype=np.int32),
                sensor_names=np.asarray(sensor_names),
                y=np.asarray(["walking"] * virtual_sequence.num_frames),
            )
            reference_path.with_name("metadata.json").write_text(
                json.dumps(
                    {
                        "dataset": "RobotEmotions",
                        "imu": {
                            "sensor_ids": [1, 2, 3, 4],
                            "sensor_names": sensor_names,
                        },
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )

            calibration_result = calibrate_virtual_imu_sequence(
                virtual_sequence,
                real_imu_reference_path=reference_path,
                activity_label="walking",
                percentile_resolution=32,
                per_class=False,
            )

            calibrated_acc = np.asarray(calibration_result["virtual_imu_sequence"].acc, dtype=np.float32)
            expected_same = np.asarray(acc, dtype=np.float32).copy()
            expected_same[:, 0, :] = acc[:, 0, :] * np.float32(0.4) + np.float32(0.5)
            expected_same[:, 1, :] = acc[:, 1, :] * np.float32(0.6) + np.float32(1.5)
            expected_same[:, 2, :] = acc[:, 2, :] * np.float32(1.8) + np.float32(7.0)
            expected_same[:, 3, :] = acc[:, 3, :] * np.float32(1.1) + np.float32(3.0)
            expected_cross = np.asarray(expected_same, dtype=np.float32).copy()
            expected_cross[:, 2, :] = acc[:, 2, :] * np.float32(1.1) + np.float32(3.0)
            expected_cross[:, 3, :] = acc[:, 3, :] * np.float32(1.8) + np.float32(7.0)

            error_same = float(np.mean(np.abs(calibrated_acc - expected_same)))
            error_cross = float(np.mean(np.abs(calibrated_acc - expected_cross)))
            self.assertLess(error_same, error_cross * 0.5)
