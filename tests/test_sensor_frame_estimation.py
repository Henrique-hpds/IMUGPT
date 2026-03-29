import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pose_module.interfaces import VirtualIMUSequence
from pose_module.processing.sensor_frame_estimation import estimate_sensor_frame_alignment


def _gyro_signal_left(timestamps_sec: np.ndarray) -> np.ndarray:
    t = np.asarray(timestamps_sec, dtype=np.float32)
    return np.stack(
        [
            0.7 * np.sin(2.1 * t) + 0.2 * np.cos(0.7 * t),
            0.5 * np.cos(1.3 * t + 0.2),
            0.4 * np.sin(1.7 * t + 0.5),
        ],
        axis=1,
    ).astype(np.float32)


def _gyro_signal_right(timestamps_sec: np.ndarray) -> np.ndarray:
    t = np.asarray(timestamps_sec, dtype=np.float32)
    return np.stack(
        [
            0.65 * np.sin(1.8 * t + 0.1),
            0.45 * np.cos(1.5 * t + 0.35) + 0.1 * np.sin(0.4 * t),
            0.55 * np.sin(1.1 * t + 0.9),
        ],
        axis=1,
    ).astype(np.float32)


def _acc_signal_left(timestamps_sec: np.ndarray) -> np.ndarray:
    t = np.asarray(timestamps_sec, dtype=np.float32)
    return np.stack(
        [
            0.35 * np.sin(0.8 * t),
            9.81 + 0.18 * np.cos(0.6 * t + 0.1),
            0.22 * np.sin(0.5 * t + 0.3),
        ],
        axis=1,
    ).astype(np.float32)


def _acc_signal_right(timestamps_sec: np.ndarray) -> np.ndarray:
    t = np.asarray(timestamps_sec, dtype=np.float32)
    return np.stack(
        [
            0.25 * np.cos(0.7 * t),
            9.81 + 0.16 * np.sin(0.55 * t + 0.2),
            0.27 * np.cos(0.45 * t + 0.6),
        ],
        axis=1,
    ).astype(np.float32)


def _apply_rotation(values: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32) @ np.asarray(rotation_matrix, dtype=np.float32).T


class SensorFrameEstimationTests(unittest.TestCase):
    def test_estimate_sensor_frame_alignment_recovers_known_forearm_rotations_and_lags(self) -> None:
        virtual_timestamps_sec = np.arange(360, dtype=np.float32) / np.float32(20.0)
        real_timestamps_sec = np.arange(1800, dtype=np.float32) / np.float32(100.0)
        sensor_names = ["waist", "head", "right_forearm", "left_forearm"]

        virtual_acc = np.zeros((virtual_timestamps_sec.shape[0], len(sensor_names), 3), dtype=np.float32)
        virtual_gyro = np.zeros_like(virtual_acc)
        virtual_acc[:, 2, :] = _acc_signal_right(virtual_timestamps_sec)
        virtual_acc[:, 3, :] = _acc_signal_left(virtual_timestamps_sec)
        virtual_gyro[:, 2, :] = _gyro_signal_right(virtual_timestamps_sec)
        virtual_gyro[:, 3, :] = _gyro_signal_left(virtual_timestamps_sec)

        virtual_sequence = VirtualIMUSequence(
            clip_id="synthetic_frame_alignment",
            fps=20.0,
            sensor_names=sensor_names,
            acc=virtual_acc,
            gyro=virtual_gyro,
            timestamps_sec=virtual_timestamps_sec,
            source="synthetic_virtual",
        )

        right_rotation = Rotation.from_euler("xyz", [18.0, -12.0, 35.0], degrees=True).as_matrix().astype(np.float32)
        left_rotation = Rotation.from_euler("xyz", [-22.0, 14.0, 28.0], degrees=True).as_matrix().astype(np.float32)
        right_lag_sec = -0.18
        left_lag_sec = 0.31

        real_acc = np.zeros((real_timestamps_sec.shape[0], len(sensor_names), 3), dtype=np.float32)
        real_gyro = np.zeros_like(real_acc)
        real_acc[:, 2, :] = _apply_rotation(_acc_signal_right(real_timestamps_sec - np.float32(right_lag_sec)), right_rotation)
        real_acc[:, 3, :] = _apply_rotation(_acc_signal_left(real_timestamps_sec - np.float32(left_lag_sec)), left_rotation)
        real_gyro[:, 2, :] = _apply_rotation(
            _gyro_signal_right(real_timestamps_sec - np.float32(right_lag_sec)),
            right_rotation,
        )
        real_gyro[:, 3, :] = _apply_rotation(
            _gyro_signal_left(real_timestamps_sec - np.float32(left_lag_sec)),
            left_rotation,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            real_path = Path(tmp_dir) / "imu.npz"
            np.savez_compressed(
                real_path,
                timestamps_sec=real_timestamps_sec,
                acc=real_acc,
                gyro=real_gyro,
                sensor_names=np.asarray(sensor_names),
            )

            result = estimate_sensor_frame_alignment(
                virtual_sequence,
                real_imu_npz_path=real_path,
                output_dir=tmp_dir,
            )

            self.assertEqual(result["status"], "ok")
            self.assertTrue(Path(result["artifacts"]["virtual_imu_frame_aligned_npz_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["sensor_frame_estimation_report_json_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["frame_alignment_quality_report_json_path"]).exists())
            self.assertTrue(result["quality_report"]["enabled"])
            self.assertGreater(result["quality_report"]["mean_gyro_corr_after"], 0.95)
            self.assertGreater(result["quality_report"]["mean_acc_corr_after"], 0.95)

            right_report = result["frame_estimation_report"]["sensor_reports"]["right_forearm"]
            left_report = result["frame_estimation_report"]["sensor_reports"]["left_forearm"]

            self.assertAlmostEqual(float(right_report["lag_sec"]), right_lag_sec, delta=0.05)
            self.assertAlmostEqual(float(left_report["lag_sec"]), left_lag_sec, delta=0.05)
            self.assertGreater(float(right_report["gyro_corr_after"]), float(right_report["gyro_corr_before"]))
            self.assertGreater(float(left_report["gyro_corr_after"]), float(left_report["gyro_corr_before"]))

            estimated_right = np.asarray(right_report["rotation_matrix"], dtype=np.float32)
            estimated_left = np.asarray(left_report["rotation_matrix"], dtype=np.float32)
            right_delta = Rotation.from_matrix(estimated_right) * Rotation.from_matrix(right_rotation).inv()
            left_delta = Rotation.from_matrix(estimated_left) * Rotation.from_matrix(left_rotation).inv()
            self.assertLess(np.degrees(right_delta.magnitude()), 8.0)
            self.assertLess(np.degrees(left_delta.magnitude()), 8.0)

            aligned_sequence = result["aligned_virtual_imu_sequence"]
            np.testing.assert_allclose(aligned_sequence.acc[:, 0, :], virtual_sequence.acc[:, 0, :], atol=1e-6)
            np.testing.assert_allclose(aligned_sequence.gyro[:, 1, :], virtual_sequence.gyro[:, 1, :], atol=1e-6)

    def test_estimate_sensor_frame_alignment_warns_when_clip_is_nearly_static(self) -> None:
        timestamps_sec = np.arange(360, dtype=np.float32) / np.float32(20.0)
        sensor_names = ["waist", "head", "right_forearm", "left_forearm"]
        virtual_acc = np.zeros((timestamps_sec.shape[0], len(sensor_names), 3), dtype=np.float32)
        virtual_acc[..., 1] = np.float32(9.81)
        virtual_gyro = np.zeros_like(virtual_acc)
        virtual_sequence = VirtualIMUSequence(
            clip_id="static_frame_alignment",
            fps=20.0,
            sensor_names=sensor_names,
            acc=virtual_acc,
            gyro=virtual_gyro,
            timestamps_sec=timestamps_sec,
            source="synthetic_virtual_static",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            real_path = Path(tmp_dir) / "imu.npz"
            np.savez_compressed(
                real_path,
                timestamps_sec=timestamps_sec,
                acc=virtual_acc,
                gyro=virtual_gyro,
                sensor_names=np.asarray(sensor_names),
            )

            result = estimate_sensor_frame_alignment(
                virtual_sequence,
                real_imu_npz_path=real_path,
                output_dir=tmp_dir,
            )

            self.assertEqual(result["status"], "warning")
            self.assertEqual(result["quality_report"]["status"], "warning")
            left_report = result["frame_estimation_report"]["sensor_reports"]["left_forearm"]
            self.assertIsNone(left_report["rotation_matrix"])
            self.assertEqual(left_report["confidence_score"], 0.0)
            self.assertIn("dynamic_gyro_energy_too_low", left_report["notes"])

    def test_estimate_sensor_frame_alignment_uses_robot_emotions_sensor_ids_for_legacy_real_labels(self) -> None:
        virtual_timestamps_sec = np.arange(360, dtype=np.float32) / np.float32(20.0)
        real_timestamps_sec = np.arange(1800, dtype=np.float32) / np.float32(100.0)
        virtual_sensor_names = ["waist", "head", "right_forearm", "left_forearm"]
        legacy_sensor_names = ["waist", "head", "right_forearm", "left_forearm"]

        virtual_acc = np.zeros((virtual_timestamps_sec.shape[0], len(virtual_sensor_names), 3), dtype=np.float32)
        virtual_gyro = np.zeros_like(virtual_acc)
        virtual_acc[:, 2, :] = _acc_signal_right(virtual_timestamps_sec)
        virtual_acc[:, 3, :] = _acc_signal_left(virtual_timestamps_sec)
        virtual_gyro[:, 2, :] = _gyro_signal_right(virtual_timestamps_sec)
        virtual_gyro[:, 3, :] = _gyro_signal_left(virtual_timestamps_sec)

        virtual_sequence = VirtualIMUSequence(
            clip_id="synthetic_robot_emotions_legacy_labels",
            fps=20.0,
            sensor_names=virtual_sensor_names,
            acc=virtual_acc,
            gyro=virtual_gyro,
            timestamps_sec=virtual_timestamps_sec,
            source="synthetic_virtual",
        )

        right_rotation = Rotation.from_euler("xyz", [12.0, -7.0, 26.0], degrees=True).as_matrix().astype(np.float32)
        left_rotation = Rotation.from_euler("xyz", [-16.0, 11.0, 21.0], degrees=True).as_matrix().astype(np.float32)
        right_lag_sec = -0.24
        left_lag_sec = 0.37

        real_acc = np.zeros((real_timestamps_sec.shape[0], 4, 3), dtype=np.float32)
        real_gyro = np.zeros_like(real_acc)
        real_acc[:, 2, :] = _apply_rotation(
            _acc_signal_left(real_timestamps_sec - np.float32(left_lag_sec)),
            left_rotation,
        )
        real_acc[:, 3, :] = _apply_rotation(
            _acc_signal_right(real_timestamps_sec - np.float32(right_lag_sec)),
            right_rotation,
        )
        real_gyro[:, 2, :] = _apply_rotation(
            _gyro_signal_left(real_timestamps_sec - np.float32(left_lag_sec)),
            left_rotation,
        )
        real_gyro[:, 3, :] = _apply_rotation(
            _gyro_signal_right(real_timestamps_sec - np.float32(right_lag_sec)),
            right_rotation,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            real_path = Path(tmp_dir) / "imu.npz"
            np.savez_compressed(
                real_path,
                timestamps_sec=real_timestamps_sec,
                acc=real_acc,
                gyro=real_gyro,
                sensor_ids=np.asarray([1, 2, 3, 4], dtype=np.int32),
                sensor_names=np.asarray(legacy_sensor_names),
            )
            metadata_path = Path(tmp_dir) / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "dataset": "RobotEmotions",
                        "imu": {
                            "sensor_ids": [1, 2, 3, 4],
                            "sensor_names": legacy_sensor_names,
                        },
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )

            result = estimate_sensor_frame_alignment(
                virtual_sequence,
                real_imu_npz_path=real_path,
                output_dir=tmp_dir,
            )

            self.assertEqual(result["status"], "ok")
            right_report = result["frame_estimation_report"]["sensor_reports"]["right_forearm"]
            left_report = result["frame_estimation_report"]["sensor_reports"]["left_forearm"]
            self.assertAlmostEqual(float(right_report["lag_sec"]), right_lag_sec, delta=0.05)
            self.assertAlmostEqual(float(left_report["lag_sec"]), left_lag_sec, delta=0.05)
