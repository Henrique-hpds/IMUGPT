import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pose_module.interfaces import IMUGPT_22_JOINT_NAMES, IMUGPT_22_PARENT_INDICES, PoseSequence3D
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
