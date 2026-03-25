import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pose_module.interfaces import (
    MOTIONBERT_17_JOINT_NAMES,
    MOTIONBERT_17_PARENT_INDICES,
    PoseSequence2D,
    PoseSequence3D,
)
from pose_module.export.debug_video import _project_pose3d_sequence_to_panel
from pose_module.motionbert.adapter import (
    build_motionbert_window_batch,
    merge_motionbert_window_predictions,
)
from pose_module.motionbert.lifter import (
    _canonicalize_backend_prediction_array,
    _fill_missing_keypoints_for_lifter,
    _resolve_backend_joint_names,
    run_motionbert_lifter,
)
from pose_module.pipeline import run_pose3d_pipeline


_MB17_BASE_POINTS = {
    "pelvis": (0.00, 0.00),
    "left_hip": (-0.10, 0.02),
    "right_hip": (0.10, 0.02),
    "spine": (0.00, -0.10),
    "left_knee": (-0.12, 0.26),
    "right_knee": (0.12, 0.26),
    "thorax": (0.00, -0.22),
    "left_ankle": (-0.12, 0.52),
    "right_ankle": (0.12, 0.52),
    "neck": (0.00, -0.32),
    "head": (0.00, -0.44),
    "left_shoulder": (-0.18, -0.22),
    "right_shoulder": (0.18, -0.22),
    "left_elbow": (-0.24, -0.14),
    "right_elbow": (0.24, -0.14),
    "left_wrist": (-0.30, -0.04),
    "right_wrist": (0.30, -0.04),
}

_H36M_17_JOINT_NAMES = [
    "root",
    "right_hip",
    "right_knee",
    "right_foot",
    "left_hip",
    "left_knee",
    "left_foot",
    "spine",
    "thorax",
    "neck_base",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]


def _make_motionbert_sequence(num_frames: int) -> PoseSequence2D:
    keypoints_xy = np.zeros((num_frames, len(MOTIONBERT_17_JOINT_NAMES), 2), dtype=np.float32)
    confidence = np.full((num_frames, len(MOTIONBERT_17_JOINT_NAMES)), 0.95, dtype=np.float32)
    bbox_xywh = np.tile(np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32), (num_frames, 1))

    for frame_index in range(num_frames):
        phase = float(frame_index) / 10.0
        for joint_index, joint_name in enumerate(MOTIONBERT_17_JOINT_NAMES):
            base_x, base_y = _MB17_BASE_POINTS[joint_name]
            x = base_x + (0.02 * np.sin(phase))
            y = base_y + (0.01 * np.cos(phase))
            keypoints_xy[frame_index, joint_index] = np.asarray([x, y], dtype=np.float32)

    return PoseSequence2D(
        clip_id="clip_motionbert",
        fps=20.0,
        fps_original=30.0,
        joint_names_2d=list(MOTIONBERT_17_JOINT_NAMES),
        keypoints_xy=keypoints_xy,
        confidence=confidence,
        bbox_xywh=bbox_xywh,
        frame_indices=np.arange(num_frames, dtype=np.int32),
        timestamps_sec=np.arange(num_frames, dtype=np.float32) / 20.0,
        source="vitpose-b_motionbert17_clean",
    )


def _predictor(batch_inputs: np.ndarray):
    output = np.zeros((batch_inputs.shape[0], batch_inputs.shape[1], batch_inputs.shape[2], 3), dtype=np.float32)
    output[..., 0] = batch_inputs[..., 0]
    output[..., 1] = -batch_inputs[..., 1]
    output[..., 2] = np.arange(batch_inputs.shape[2], dtype=np.float32)[None, None, :]
    return {
        "keypoints_3d": output,
        "joint_names": list(MOTIONBERT_17_JOINT_NAMES),
    }


class MotionBERTAdapterTests(unittest.TestCase):
    def test_build_motionbert_window_batch_includes_confidence_channel(self) -> None:
        sequence = _make_motionbert_sequence(120)

        batch = build_motionbert_window_batch(
            sequence,
            window_size=50,
            window_overlap=0.5,
            include_confidence=True,
        )

        self.assertEqual(batch.inputs.shape[1:], (50, 17, 3))
        self.assertGreater(batch.num_windows, 1)
        self.assertEqual(int(batch.frame_index_map[0, 0]), 0)
        last_valid_position = int(np.flatnonzero(batch.valid_mask[-1])[-1])
        self.assertEqual(int(batch.frame_index_map[-1, last_valid_position]), 119)

    def test_merge_motionbert_window_predictions_blends_overlaps(self) -> None:
        sequence = _make_motionbert_sequence(120)
        batch = build_motionbert_window_batch(
            sequence,
            window_size=50,
            window_overlap=0.5,
            include_confidence=False,
        )
        predictions = np.zeros((batch.num_windows, batch.window_size, 17, 3), dtype=np.float32)
        for window_index in range(batch.num_windows):
            predictions[window_index] = float(window_index)

        fused = merge_motionbert_window_predictions(
            predictions,
            batch,
            num_frames=sequence.num_frames,
        )

        self.assertEqual(fused.shape, (120, 17, 3))
        self.assertAlmostEqual(float(fused[0, 0, 0]), 0.0, places=5)
        self.assertGreater(float(fused[30, 0, 0]), 0.0)
        self.assertLess(float(fused[30, 0, 0]), 1.0)
        self.assertGreater(float(fused[60, 0, 0]), 1.0)
        self.assertLess(float(fused[60, 0, 0]), 2.0)

    def test_canonicalize_backend_prediction_array_reorders_h36m_output_to_mb17(self) -> None:
        raw_prediction = np.zeros((5, len(_H36M_17_JOINT_NAMES), 3), dtype=np.float32)
        for joint_index in range(len(_H36M_17_JOINT_NAMES)):
            raw_prediction[:, joint_index, 0] = float(joint_index)

        canonical_prediction = _canonicalize_backend_prediction_array(
            raw_prediction,
            joint_names=_H36M_17_JOINT_NAMES,
        )

        expected_joint_positions = {
            "pelvis": 0.0,
            "left_hip": 4.0,
            "right_hip": 1.0,
            "spine": 7.0,
            "left_knee": 5.0,
            "right_knee": 2.0,
            "thorax": 8.0,
            "left_ankle": 6.0,
            "right_ankle": 3.0,
            "neck": 9.0,
            "head": 10.0,
            "left_shoulder": 11.0,
            "right_shoulder": 14.0,
            "left_elbow": 12.0,
            "right_elbow": 15.0,
            "left_wrist": 13.0,
            "right_wrist": 16.0,
        }
        for joint_index, joint_name in enumerate(MOTIONBERT_17_JOINT_NAMES):
            self.assertAlmostEqual(
                float(canonical_prediction[0, joint_index, 0]),
                expected_joint_positions[joint_name],
                places=5,
            )

    def test_resolve_backend_joint_names_prefers_dataset_meta_keypoint_order(self) -> None:
        dataset_meta = {
            "dataset_name": "h36m",
            "keypoint_id2name": {index: name for index, name in enumerate(_H36M_17_JOINT_NAMES)},
        }

        resolved_joint_names = _resolve_backend_joint_names(dataset_meta)

        self.assertEqual(resolved_joint_names, _H36M_17_JOINT_NAMES)


class MotionBERTLifterTests(unittest.TestCase):
    def test_run_motionbert_lifter_exports_pose3d_artifacts(self) -> None:
        sequence = _make_motionbert_sequence(90)

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_motionbert_lifter(
                sequence,
                output_dir=tmp_dir,
                window_size=50,
                window_overlap=0.5,
                predictor=_predictor,
                backend_name="unit_test_motionbert",
            )

            pose_sequence = result["pose_sequence"]
            self.assertEqual(pose_sequence.joint_names_3d, list(MOTIONBERT_17_JOINT_NAMES))
            self.assertEqual(pose_sequence.joint_positions_xyz.shape, (90, 17, 3))
            self.assertTrue(Path(result["artifacts"]["pose3d_npz_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["pose3d_raw_keypoints_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["motionbert_run_json_path"]).exists())
            self.assertEqual(result["quality_report"]["backend_name"], "unit_test_motionbert")
            self.assertEqual(result["quality_report"]["status"], "ok")
            self.assertAlmostEqual(float(pose_sequence.joint_positions_xyz[0, 4, 2]), 4.0, places=5)


class Pose3DPipelineTests(unittest.TestCase):
    def test_run_pose3d_pipeline_merges_motionbert_stage(self) -> None:
        sequence = _make_motionbert_sequence(60)
        pose2d_result = {
            "clip_id": "clip_pipeline3d",
            "pose_sequence": sequence,
            "raw_pose_sequence": sequence,
            "quality_report": {
                "clip_id": "clip_pipeline3d",
                "status": "ok",
                "visible_joint_ratio": 1.0,
                "mean_confidence": 0.95,
                "notes": [],
            },
            "track_report": {"status": "ok", "warnings": []},
            "backend_run": {"status": "ok"},
            "artifacts": {
                "quality_report_json_path": None,
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("pose_module.pipeline.run_pose2d_pipeline", return_value=pose2d_result):
                result = run_pose3d_pipeline(
                    clip_id="clip_pipeline3d",
                    video_path=str(Path(tmp_dir) / "video.mp4"),
                    output_dir=tmp_dir,
                    fps_target=20,
                    save_debug=False,
                    env_name="current",
                    motionbert_window_size=40,
                    motionbert_window_overlap=0.5,
                    motionbert_predictor=_predictor,
                    motionbert_backend_name="unit_test_motionbert",
                )

            self.assertEqual(result["quality_report"]["motionbert_backend_name"], "unit_test_motionbert")
            self.assertTrue(Path(result["artifacts"]["pose3d_npz_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["motionbert_run_json_path"]).exists())
            self.assertEqual(result["pose_sequence"].joint_names_3d, list(MOTIONBERT_17_JOINT_NAMES))

    def test_run_pose3d_pipeline_exports_side_by_side_raw_3d_debug_video(self) -> None:
        sequence = _make_motionbert_sequence(24)
        joint_positions_xyz = np.zeros((sequence.num_frames, len(MOTIONBERT_17_JOINT_NAMES), 3), dtype=np.float32)
        joint_positions_xyz[..., 0] = np.asarray(sequence.keypoints_xy[..., 0], dtype=np.float32)
        joint_positions_xyz[..., 1] = np.linspace(
            -0.2,
            0.2,
            sequence.num_frames * len(MOTIONBERT_17_JOINT_NAMES),
            dtype=np.float32,
        ).reshape(sequence.num_frames, len(MOTIONBERT_17_JOINT_NAMES))
        joint_positions_xyz[..., 2] = np.asarray(-sequence.keypoints_xy[..., 1], dtype=np.float32)
        pose3d_sequence = PoseSequence3D(
            clip_id=str(sequence.clip_id),
            fps=sequence.fps,
            fps_original=sequence.fps_original,
            joint_names_3d=list(MOTIONBERT_17_JOINT_NAMES),
            joint_positions_xyz=joint_positions_xyz,
            joint_confidence=np.asarray(sequence.confidence, dtype=np.float32),
            skeleton_parents=list(MOTIONBERT_17_PARENT_INDICES),
            frame_indices=np.asarray(sequence.frame_indices, dtype=np.int32),
            timestamps_sec=np.asarray(sequence.timestamps_sec, dtype=np.float32),
            source="vitpose-b_motionbert17_clean_mmpose_motionbert",
            coordinate_space="pose_lifter_aligned",
        )
        pose2d_result = {
            "clip_id": "clip_pipeline3d_debug",
            "pose_sequence": sequence,
            "raw_pose_sequence": sequence,
            "quality_report": {
                "clip_id": "clip_pipeline3d_debug",
                "status": "ok",
                "visible_joint_ratio": 1.0,
                "mean_confidence": 0.95,
                "notes": [],
            },
            "track_report": {"status": "ok", "warnings": []},
            "backend_run": {"status": "ok"},
            "cleaner_artifacts": {
                "normalization_centers_xy": np.zeros((sequence.num_frames, 2), dtype=np.float32),
                "normalization_scales": np.ones((sequence.num_frames,), dtype=np.float32),
            },
            "artifacts": {
                "quality_report_json_path": None,
            },
        }
        lifter_result = {
            "pose_sequence": pose3d_sequence,
            "quality_report": {
                "clip_id": "clip_pipeline3d_debug",
                "status": "ok",
                "notes": [],
                "coordinate_space": "pose_lifter_aligned",
            },
            "run_report": {"status": "ok"},
            "artifacts": {
                "pose3d_npz_path": "/tmp/fake_pose3d.npz",
                "motionbert_run_json_path": "/tmp/fake_motionbert_run.json",
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            rendered_outputs = []
            render_calls = []

            def _fake_render_pose3d_side_by_side_video(*, output_path, **kwargs):
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"fake mp4")
                rendered_outputs.append(output_path.name)
                render_calls.append(dict(kwargs))
                return output_path.resolve()

            with patch(
                "pose_module.pipeline.run_pose2d_pipeline",
                return_value=pose2d_result,
            ) as mocked_pose2d_pipeline, patch(
                "pose_module.pipeline.run_motionbert_lifter",
                return_value=lifter_result,
            ), patch(
                "pose_module.pipeline.render_pose3d_side_by_side_video",
                side_effect=_fake_render_pose3d_side_by_side_video,
            ) as mocked_render:
                result = run_pose3d_pipeline(
                    clip_id="clip_pipeline3d_debug",
                    video_path=str(Path(tmp_dir) / "video.mp4"),
                    output_dir=tmp_dir,
                    fps_target=20,
                    save_debug_2d=False,
                    save_debug_3d=True,
                    env_name="current",
                    motionbert_window_size=24,
                    motionbert_window_overlap=0.5,
                    motionbert_predictor=_predictor,
                    motionbert_backend_name="unit_test_motionbert",
                )

            self.assertTrue(Path(result["artifacts"]["debug_overlay_pose3d_raw_path"]).exists())
            self.assertEqual(rendered_outputs, ["debug_overlay_pose3d_raw.mp4"])
            self.assertEqual(len(render_calls), 1)
            self.assertEqual(render_calls[0]["coordinate_space"], "pose_lifter_aligned")
            self.assertEqual(mocked_render.call_count, 1)
            self.assertEqual(mocked_pose2d_pipeline.call_args.kwargs["save_debug"], False)

    def test_project_pose3d_pose_lifter_aligned_flips_horizontal_axis_to_match_video_view(self) -> None:
        joint_positions_xyz = np.asarray(
            [[[-1.0, 0.0, 0.2], [1.0, 0.0, 0.2]]],
            dtype=np.float32,
        )
        joint_confidence = np.ones((1, 2), dtype=np.float32)

        projected_points, _ = _project_pose3d_sequence_to_panel(
            joint_positions_xyz,
            joint_confidence,
            width=200,
            height=100,
            coordinate_space="pose_lifter_aligned",
        )

        self.assertGreater(float(projected_points[0, 0, 0]), float(projected_points[0, 1, 0]))

    def test_fill_missing_keypoints_for_lifter_replaces_nan_gaps_with_temporal_track(self) -> None:
        sequence = _make_motionbert_sequence(5)
        pelvis_index = MOTIONBERT_17_JOINT_NAMES.index("pelvis")
        left_wrist_index = MOTIONBERT_17_JOINT_NAMES.index("left_wrist")

        keypoints_xy = np.asarray(sequence.keypoints_xy, dtype=np.float32).copy()
        confidence = np.asarray(sequence.confidence, dtype=np.float32).copy()
        keypoints_xy[2, left_wrist_index] = np.asarray([np.nan, np.nan], dtype=np.float32)
        confidence[2, left_wrist_index] = 0.0

        filled = _fill_missing_keypoints_for_lifter(keypoints_xy, confidence)

        self.assertTrue(np.isfinite(filled[2, left_wrist_index]).all())
        self.assertTrue(np.allclose(filled[2, pelvis_index], keypoints_xy[2, pelvis_index], atol=1e-6))
        self.assertTrue(np.allclose(filled[2, left_wrist_index], 0.5 * (keypoints_xy[1, left_wrist_index] + keypoints_xy[3, left_wrist_index]), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
