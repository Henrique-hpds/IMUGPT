import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pose_module.interfaces import (
    IMUGPT_22_JOINT_NAMES,
    IMUGPT_22_PARENT_INDICES,
    PoseSequence3D,
    VirtualIMUSequence,
)
from pose_module.pipeline import run_virtual_imu_pipeline


def _make_pose3d_sequence() -> PoseSequence3D:
    num_frames = 8
    num_joints = len(IMUGPT_22_JOINT_NAMES)
    joint_positions_xyz = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    root_translation_m = np.zeros((num_frames, 3), dtype=np.float32)
    base_offsets = np.linspace(-0.3, 0.3, num_joints, dtype=np.float32)
    for frame_index in range(num_frames):
        root = np.asarray([0.03 * frame_index, 0.0, 0.0], dtype=np.float32)
        root_translation_m[frame_index] = root
        joint_positions_xyz[frame_index, :, 0] = root[0] + base_offsets
        joint_positions_xyz[frame_index, :, 1] = np.linspace(0.8, -0.9, num_joints, dtype=np.float32)
        joint_positions_xyz[frame_index, :, 2] = np.linspace(0.0, 0.5, num_joints, dtype=np.float32)
        joint_positions_xyz[frame_index, 0] = root
    return PoseSequence3D(
        clip_id="clip_virtual_pipeline",
        fps=20.0,
        fps_original=30.0,
        joint_names_3d=list(IMUGPT_22_JOINT_NAMES),
        joint_positions_xyz=joint_positions_xyz,
        joint_confidence=np.full((num_frames, num_joints), 0.95, dtype=np.float32),
        skeleton_parents=list(IMUGPT_22_PARENT_INDICES),
        frame_indices=np.arange(num_frames, dtype=np.int32),
        timestamps_sec=np.arange(num_frames, dtype=np.float32) / np.float32(20.0),
        source="unit_test_pose3d_root",
        coordinate_space="pseudo_global_metric",
        root_translation_m=root_translation_m,
    )


class VirtualIMUPipelineTests(unittest.TestCase):
    def test_run_virtual_imu_pipeline_adds_ik_and_virtual_imu_artifacts(self) -> None:
        pose_sequence = _make_pose3d_sequence()
        pose3d_result = {
            "clip_id": "clip_virtual_pipeline",
            "pose_sequence": pose_sequence,
            "quality_report": {"clip_id": "clip_virtual_pipeline", "status": "ok", "notes": []},
            "artifacts": {
                "pose3d_npz_path": "/tmp/fake_pose3d.npz",
                "quality_report_json_path": "/tmp/fake_pose3d_quality.json",
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("pose_module.pipeline.run_pose3d_pipeline", return_value=pose3d_result):
                result = run_virtual_imu_pipeline(
                    clip_id="clip_virtual_pipeline",
                    video_path=str(Path(tmp_dir) / "video.mp4"),
                    output_dir=tmp_dir,
                    save_debug=False,
                )

            self.assertEqual(result["quality_report"]["status"], "ok")
            self.assertTrue(result["quality_report"]["ik_ok"])
            self.assertTrue(result["quality_report"]["virtual_imu_ok"])
            self.assertTrue(Path(result["artifacts"]["ik_sequence_npz_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["virtual_imu_npz_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["quality_report_json_path"]).exists())
            self.assertEqual(result["virtual_imu_sequence"].acc.shape[:2], (8, 4))

    def test_run_virtual_imu_pipeline_can_apply_real_imu_calibration(self) -> None:
        pose_sequence = _make_pose3d_sequence()
        pose3d_result = {
            "clip_id": "clip_virtual_pipeline",
            "pose_sequence": pose_sequence,
            "quality_report": {"clip_id": "clip_virtual_pipeline", "status": "ok", "notes": []},
            "artifacts": {
                "pose3d_npz_path": "/tmp/fake_pose3d.npz",
                "quality_report_json_path": "/tmp/fake_pose3d_quality.json",
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_output_dir = Path(tmp_dir) / "raw"
            calibrated_output_dir = Path(tmp_dir) / "calibrated"
            with patch("pose_module.pipeline.run_pose3d_pipeline", return_value=pose3d_result):
                raw_result = run_virtual_imu_pipeline(
                    clip_id="clip_virtual_pipeline",
                    video_path=str(Path(tmp_dir) / "video.mp4"),
                    output_dir=raw_output_dir,
                    save_debug=False,
                )

                reference_path = Path(tmp_dir) / "real_imu_reference.npz"
                np.savez_compressed(
                    reference_path,
                    acc=(
                        np.asarray(raw_result["virtual_imu_sequence"].acc, dtype=np.float32) * np.float32(0.5)
                        + np.float32(2.0)
                    ),
                    sensor_names=np.asarray(raw_result["virtual_imu_sequence"].sensor_names),
                    y=np.asarray(["clip_virtual_pipeline"] * raw_result["virtual_imu_sequence"].num_frames),
                )

                calibrated_result = run_virtual_imu_pipeline(
                    clip_id="clip_virtual_pipeline",
                    video_path=str(Path(tmp_dir) / "video.mp4"),
                    output_dir=calibrated_output_dir,
                    activity_label="clip_virtual_pipeline",
                    save_debug=False,
                    real_imu_reference_path=reference_path,
                )

            self.assertIsNotNone(calibrated_result["virtual_imu_calibration_report"])
            self.assertTrue(calibrated_result["quality_report"]["virtual_imu_real_calibration_applied"])
            self.assertTrue(Path(calibrated_result["artifacts"]["virtual_imu_npz_path"]).exists())
            self.assertTrue(Path(calibrated_result["artifacts"]["virtual_imu_raw_npz_path"]).exists())
            self.assertFalse(
                np.allclose(
                    calibrated_result["virtual_imu_sequence"].acc,
                    calibrated_result["imusim_result"]["raw_virtual_imu_sequence"].acc,
                )
            )

    def test_run_virtual_imu_pipeline_calibrates_after_geometric_alignment(self) -> None:
        pose_sequence = _make_pose3d_sequence()
        pose3d_result = {
            "clip_id": "clip_virtual_pipeline",
            "pose_sequence": pose_sequence,
            "quality_report": {"clip_id": "clip_virtual_pipeline", "status": "ok", "notes": []},
            "artifacts": {
                "pose3d_npz_path": "/tmp/fake_pose3d.npz",
                "quality_report_json_path": "/tmp/fake_pose3d_quality.json",
            },
        }
        timestamps = np.arange(8, dtype=np.float32) / np.float32(20.0)
        raw_sequence = VirtualIMUSequence(
            clip_id="clip_virtual_pipeline",
            fps=20.0,
            sensor_names=["waist", "head", "right_forearm", "left_forearm"],
            acc=np.zeros((8, 4, 3), dtype=np.float32),
            gyro=np.zeros((8, 4, 3), dtype=np.float32),
            timestamps_sec=timestamps,
            source="unit_test_virtual_imu_raw",
        )
        aligned_sequence = VirtualIMUSequence(
            clip_id="clip_virtual_pipeline",
            fps=20.0,
            sensor_names=list(raw_sequence.sensor_names),
            acc=np.ones((8, 4, 3), dtype=np.float32),
            gyro=np.ones((8, 4, 3), dtype=np.float32),
            timestamps_sec=timestamps,
            source="unit_test_virtual_imu_aligned",
        )
        calibrated_sequence = VirtualIMUSequence(
            clip_id="clip_virtual_pipeline",
            fps=20.0,
            sensor_names=list(raw_sequence.sensor_names),
            acc=np.full((8, 4, 3), 2.0, dtype=np.float32),
            gyro=np.full((8, 4, 3), 2.0, dtype=np.float32),
            timestamps_sec=timestamps,
            source="unit_test_virtual_imu_aligned_calibrated",
        )
        imusim_result = {
            "virtual_imu_sequence": raw_sequence,
            "raw_virtual_imu_sequence": raw_sequence,
            "quality_report": {
                "clip_id": "clip_virtual_pipeline",
                "status": "ok",
                "virtual_imu_ok": True,
                "acc_noise_std_m_s2": 0.0,
                "gyro_noise_std_rad_s": 0.0,
                "notes": [],
            },
            "calibration_report": None,
            "artifacts": {
                "virtual_imu_npz_path": None,
                "virtual_imu_raw_npz_path": None,
                "virtual_imu_report_json_path": None,
                "virtual_imu_calibration_report_json_path": None,
            },
        }
        geometric_alignment_result = {
            "status": "ok",
            "enabled": True,
            "aligned_virtual_imu_sequence": aligned_sequence,
            "quality_report": {"enabled": True, "status": "ok", "notes": []},
            "artifacts": {},
        }
        calibration_report = {
            "status": "ok",
            "signal_mode": "acc",
            "per_class_applied": True,
            "reference_path": "/tmp/reference.npz",
            "matched_sensor_names": ["waist", "head", "right_forearm", "left_forearm"],
            "mean_abs_delta": 0.1,
            "max_abs_delta": 0.2,
            "notes": [],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("pose_module.pipeline.run_pose3d_pipeline", return_value=pose3d_result):
                with patch("pose_module.pipeline.run_imusim", return_value=imusim_result) as mocked_imusim:
                    with patch(
                        "pose_module.pipeline.load_alignment_runtime_settings",
                        return_value={"enable": True},
                    ):
                        with patch(
                            "pose_module.pipeline._run_optional_geometric_alignment",
                            return_value=geometric_alignment_result,
                        ):
                            with patch(
                                "pose_module.pipeline.calibrate_virtual_imu_sequence",
                                return_value={
                                    "virtual_imu_sequence": calibrated_sequence,
                                    "calibration_report": calibration_report,
                                },
                            ) as mocked_calibration:
                                result = run_virtual_imu_pipeline(
                                    clip_id="clip_virtual_pipeline",
                                    video_path=str(Path(tmp_dir) / "video.mp4"),
                                    output_dir=tmp_dir,
                                    save_debug=False,
                                    real_imu_reference_path=Path(tmp_dir) / "reference.npz",
                                )

            self.assertIsNone(mocked_imusim.call_args.kwargs["real_imu_reference_path"])
            self.assertIs(mocked_calibration.call_args.args[0], aligned_sequence)
            np.testing.assert_allclose(result["virtual_imu_sequence"].acc, calibrated_sequence.acc)
            self.assertTrue(result["virtual_imu_quality_report"]["real_imu_calibration_applied"])
            self.assertTrue(Path(result["artifacts"]["virtual_imu_raw_npz_path"]).exists())

    def test_run_virtual_imu_pipeline_merges_sensor_frame_estimation_outputs(self) -> None:
        pose_sequence = _make_pose3d_sequence()
        pose3d_result = {
            "clip_id": "clip_virtual_pipeline",
            "pose_sequence": pose_sequence,
            "quality_report": {"clip_id": "clip_virtual_pipeline", "status": "ok", "notes": []},
            "artifacts": {
                "pose3d_npz_path": "/tmp/fake_pose3d.npz",
                "quality_report_json_path": "/tmp/fake_pose3d_quality.json",
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "clip" / "pose"
            output_dir.mkdir(parents=True, exist_ok=True)
            aligned_path = output_dir / "virtual_imu_frame_aligned.npz"
            report_path = output_dir / "sensor_frame_estimation_report.json"
            quality_path = output_dir / "frame_alignment_quality_report.json"

            aligned_sequence = VirtualIMUSequence(
                clip_id="clip_virtual_pipeline",
                fps=20.0,
                sensor_names=["waist", "head", "right_forearm", "left_forearm"],
                acc=np.zeros((8, 4, 3), dtype=np.float32),
                gyro=np.zeros((8, 4, 3), dtype=np.float32),
                timestamps_sec=np.arange(8, dtype=np.float32) / np.float32(20.0),
                source="unit_test_virtual_imu_frame_aligned",
            )
            np.savez_compressed(aligned_path, **aligned_sequence.to_npz_payload())
            report_path.write_text(
                json.dumps({"clip_id": "clip_virtual_pipeline", "status": "ok"}, ensure_ascii=True),
                encoding="utf-8",
            )
            quality_path.write_text(
                json.dumps({"clip_id": "clip_virtual_pipeline", "status": "ok"}, ensure_ascii=True),
                encoding="utf-8",
            )
            fake_frame_alignment_result = {
                "status": "ok",
                "aligned_virtual_imu_sequence": aligned_sequence,
                "frame_estimation_report": {"clip_id": "clip_virtual_pipeline", "status": "ok"},
                "lag_report": {"clip_id": "clip_virtual_pipeline", "status": "ok"},
                "quality_report": {
                    "enabled": True,
                    "clip_id": "clip_virtual_pipeline",
                    "status": "ok",
                    "target_sensor_names": ["left_forearm", "right_forearm"],
                    "estimated_sensor_names": ["left_forearm", "right_forearm"],
                    "real_imu_npz_path": str((output_dir.parent / "imu.npz").resolve()),
                    "mean_gyro_corr_before": 0.12,
                    "mean_gyro_corr_after": 0.88,
                    "mean_acc_corr_before": 0.25,
                    "mean_acc_corr_after": 0.79,
                    "notes": [],
                },
                "artifacts": {
                    "virtual_imu_frame_aligned_npz_path": str(aligned_path.resolve()),
                    "sensor_frame_estimation_report_json_path": str(report_path.resolve()),
                    "frame_alignment_quality_report_json_path": str(quality_path.resolve()),
                },
            }

            with patch("pose_module.pipeline.run_pose3d_pipeline", return_value=pose3d_result):
                with patch(
                    "pose_module.pipeline._run_optional_sensor_frame_estimation",
                    return_value=fake_frame_alignment_result,
                ) as mocked_alignment:
                    result = run_virtual_imu_pipeline(
                        clip_id="clip_virtual_pipeline",
                        video_path=str(Path(tmp_dir) / "video.mp4"),
                        output_dir=output_dir,
                        save_debug=False,
                        estimate_sensor_frame=True,
                        estimate_sensor_names=["left_forearm", "right_forearm"],
                    )

            mocked_alignment.assert_called_once()
            self.assertTrue(result["quality_report"]["sensor_frame_estimation_enabled"])
            self.assertEqual(result["quality_report"]["sensor_frame_estimation_status"], "ok")
            self.assertEqual(
                result["quality_report"]["sensor_frame_estimation_target_sensors"],
                ["left_forearm", "right_forearm"],
            )
            self.assertAlmostEqual(result["quality_report"]["sensor_frame_estimation_mean_gyro_corr_after"], 0.88)
            self.assertAlmostEqual(result["quality_report"]["sensor_frame_estimation_mean_acc_corr_after"], 0.79)
            self.assertEqual(result["frame_alignment_quality_report"]["status"], "ok")
            self.assertIsNotNone(result["aligned_virtual_imu_sequence"])
            self.assertEqual(
                result["artifacts"]["virtual_imu_frame_aligned_npz_path"],
                str(aligned_path.resolve()),
            )

    def test_run_virtual_imu_pipeline_survives_geometric_alignment_failure(self) -> None:
        pose_sequence = _make_pose3d_sequence()
        pose3d_result = {
            "clip_id": "clip_virtual_pipeline",
            "pose_sequence": pose_sequence,
            "quality_report": {"clip_id": "clip_virtual_pipeline", "status": "ok", "notes": []},
            "artifacts": {
                "pose3d_npz_path": "/tmp/fake_pose3d.npz",
                "quality_report_json_path": "/tmp/fake_pose3d_quality.json",
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("pose_module.pipeline.run_pose3d_pipeline", return_value=pose3d_result):
                with patch(
                    "pose_module.pipeline.load_alignment_runtime_settings",
                    return_value={"enable": True},
                ):
                    with patch(
                        "pose_module.pipeline.run_geometric_alignment",
                        side_effect=RuntimeError("synthetic_alignment_failure"),
                    ):
                        result = run_virtual_imu_pipeline(
                            clip_id="clip_virtual_pipeline",
                            video_path=str(Path(tmp_dir) / "video.mp4"),
                            output_dir=tmp_dir,
                            save_debug=False,
                        )

            self.assertEqual(result["quality_report"]["status"], "warning")
            self.assertEqual(result["geometric_alignment_quality_report"]["status"], "warning")
            self.assertIn(
                "geometric_alignment_failed:synthetic_alignment_failure",
                result["geometric_alignment_quality_report"]["notes"],
            )
            self.assertTrue(Path(result["artifacts"]["virtual_imu_npz_path"]).exists())
            self.assertEqual(result["virtual_imu_sequence"].acc.shape[:2], (8, 4))
