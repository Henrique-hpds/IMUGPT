import unittest

import numpy as np

from pose_module.interfaces import IMUGPT_22_JOINT_NAMES, IMUGPT_22_PARENT_INDICES, PoseSequence3D
from pose_module.processing.metric_normalizer import (
    BODY_METRIC_LOCAL_COORDINATE_SPACE,
    run_metric_normalizer,
)


_IMUGPT22_BASE_POINTS = {
    "Pelvis": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    "Left_hip": np.asarray([-0.2, 0.0, 0.0], dtype=np.float32),
    "Right_hip": np.asarray([0.2, 0.0, 0.0], dtype=np.float32),
    "Spine1": np.asarray([0.0, 0.2, 0.0], dtype=np.float32),
    "Left_knee": np.asarray([-0.2, -0.5, 0.0], dtype=np.float32),
    "Right_knee": np.asarray([0.2, -0.5, 0.0], dtype=np.float32),
    "Spine2": np.asarray([0.0, 0.4, 0.0], dtype=np.float32),
    "Left_ankle": np.asarray([-0.2, -0.9, 0.0], dtype=np.float32),
    "Right_ankle": np.asarray([0.2, -0.9, 0.0], dtype=np.float32),
    "Spine3": np.asarray([0.0, 0.6, 0.0], dtype=np.float32),
    "Left_foot": np.asarray([-0.2, -0.9, 0.12], dtype=np.float32),
    "Right_foot": np.asarray([0.2, -0.9, 0.12], dtype=np.float32),
    "Neck": np.asarray([0.0, 0.8, 0.0], dtype=np.float32),
    "Left_collar": np.asarray([-0.125, 0.6, 0.0], dtype=np.float32),
    "Right_collar": np.asarray([0.125, 0.6, 0.0], dtype=np.float32),
    "Head": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    "Left_shoulder": np.asarray([-0.25, 0.6, 0.0], dtype=np.float32),
    "Right_shoulder": np.asarray([0.25, 0.6, 0.0], dtype=np.float32),
    "Left_elbow": np.asarray([-0.45, 0.45, 0.0], dtype=np.float32),
    "Right_elbow": np.asarray([0.45, 0.45, 0.0], dtype=np.float32),
    "Left_wrist": np.asarray([-0.65, 0.3, 0.0], dtype=np.float32),
    "Right_wrist": np.asarray([0.65, 0.3, 0.0], dtype=np.float32),
}


def _rotation_y(angle_rad: float) -> np.ndarray:
    cos_angle = np.float32(np.cos(angle_rad))
    sin_angle = np.float32(np.sin(angle_rad))
    return np.asarray(
        [
            [cos_angle, 0.0, sin_angle],
            [0.0, 1.0, 0.0],
            [-sin_angle, 0.0, cos_angle],
        ],
        dtype=np.float32,
    )


def _make_imugpt22_sequence(
    *,
    num_frames: int = 9,
    yaw_rad: float = 0.55,
    jitter_frame: int | None = None,
    degenerate_body_frame_index: int | None = None,
) -> PoseSequence3D:
    joint_positions_xyz = np.zeros((num_frames, len(IMUGPT_22_JOINT_NAMES), 3), dtype=np.float32)
    joint_confidence = np.full((num_frames, len(IMUGPT_22_JOINT_NAMES)), 0.95, dtype=np.float32)
    rotation = _rotation_y(float(yaw_rad))

    for frame_index in range(num_frames):
        translation = np.asarray(
            [1.25 + (0.03 * frame_index), -0.4 + (0.01 * frame_index), 2.0 - (0.02 * frame_index)],
            dtype=np.float32,
        )
        for joint_index, joint_name in enumerate(IMUGPT_22_JOINT_NAMES):
            point = _IMUGPT22_BASE_POINTS[joint_name].copy()
            if jitter_frame is not None and frame_index == int(jitter_frame) and joint_name == "Right_wrist":
                point[0] += np.float32(0.12)
            rotated = point @ rotation.T
            joint_positions_xyz[frame_index, joint_index] = rotated + translation

    if degenerate_body_frame_index is not None:
        frame_index = int(degenerate_body_frame_index)
        left_hip_index = IMUGPT_22_JOINT_NAMES.index("Left_hip")
        right_hip_index = IMUGPT_22_JOINT_NAMES.index("Right_hip")
        joint_positions_xyz[frame_index, right_hip_index] = joint_positions_xyz[frame_index, left_hip_index]

    return PoseSequence3D(
        clip_id="clip_metric_normalizer",
        fps=20.0,
        fps_original=30.0,
        joint_names_3d=list(IMUGPT_22_JOINT_NAMES),
        joint_positions_xyz=joint_positions_xyz,
        joint_confidence=joint_confidence,
        skeleton_parents=list(IMUGPT_22_PARENT_INDICES),
        frame_indices=np.arange(num_frames, dtype=np.int32),
        timestamps_sec=np.arange(num_frames, dtype=np.float32) / 20.0,
        source="vitpose-b_motionbert17_clean_mmpose_motionbert_imugpt22",
        coordinate_space="camera",
    )


class MetricNormalizerTests(unittest.TestCase):
    def test_run_metric_normalizer_builds_body_frame_and_metric_scale(self) -> None:
        sequence = _make_imugpt22_sequence(jitter_frame=4)

        result = run_metric_normalizer(
            sequence,
            target_femur_length_m=0.45,
            smoothing_window_length=5,
            smoothing_polyorder=2,
        )

        pose_sequence = result["pose_sequence"]
        normalization = result["normalization_result"]
        quality_report = result["quality_report"]

        self.assertEqual(quality_report["status"], "ok")
        self.assertEqual(pose_sequence.coordinate_space, BODY_METRIC_LOCAL_COORDINATE_SPACE)
        self.assertEqual(quality_report["coordinate_space"], BODY_METRIC_LOCAL_COORDINATE_SPACE)
        self.assertEqual(pose_sequence.joint_names_3d, list(IMUGPT_22_JOINT_NAMES))

        pelvis_index = IMUGPT_22_JOINT_NAMES.index("Pelvis")
        left_hip_index = IMUGPT_22_JOINT_NAMES.index("Left_hip")
        right_hip_index = IMUGPT_22_JOINT_NAMES.index("Right_hip")
        left_knee_index = IMUGPT_22_JOINT_NAMES.index("Left_knee")
        right_wrist_index = IMUGPT_22_JOINT_NAMES.index("Right_wrist")

        np.testing.assert_allclose(
            normalization["joint_positions_body_frame"][:, pelvis_index],
            np.zeros((sequence.num_frames, 3), dtype=np.float32),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            pose_sequence.joint_positions_xyz[:, pelvis_index],
            np.zeros((sequence.num_frames, 3), dtype=np.float32),
            atol=1e-5,
        )
        self.assertTrue(np.all(normalization["joint_positions_body_frame"][:, right_hip_index, 0] > 0.0))
        self.assertTrue(np.all(normalization["joint_positions_body_frame"][:, left_hip_index, 0] < 0.0))
        self.assertAlmostEqual(float(normalization["scale_factor"]), 0.9, places=4)
        self.assertAlmostEqual(
            float(pose_sequence.joint_positions_xyz[0, left_knee_index, 1]),
            -0.45,
            places=4,
        )

        expected_right_wrist_x = np.float32(0.65 * 0.9)
        raw_metric_wrist_x = float(normalization["joint_positions_metric_local"][4, right_wrist_index, 0])
        smoothed_wrist_x = float(normalization["joint_positions_smoothed"][4, right_wrist_index, 0])
        self.assertLess(abs(smoothed_wrist_x - expected_right_wrist_x), abs(raw_metric_wrist_x - expected_right_wrist_x))

    def test_run_metric_normalizer_falls_back_to_previous_body_frame_when_axes_degenerate(self) -> None:
        sequence = _make_imugpt22_sequence(degenerate_body_frame_index=3)

        result = run_metric_normalizer(
            sequence,
            target_femur_length_m=0.45,
            smoothing_window_length=5,
            smoothing_polyorder=2,
        )

        quality_report = result["quality_report"]

        self.assertEqual(quality_report["status"], "warning")
        self.assertEqual(quality_report["body_frame_fallback_frames"], 1)
        self.assertIn("body_frame_fallback_frames:1", quality_report["notes"])
        self.assertTrue(np.isfinite(result["pose_sequence"].joint_positions_xyz).all())
        self.assertEqual(int(np.count_nonzero(result["artifacts"]["body_frame_fallback_mask"])), 1)

    def test_run_metric_normalizer_applies_tibia_prior_only_to_corrected_legs(self) -> None:
        sequence = _make_imugpt22_sequence()

        result = run_metric_normalizer(
            sequence,
            target_femur_length_m=0.45,
            target_tibia_length_m=0.40,
            smoothing_window_length=5,
            corrected_smoothing_window_length=3,
            smoothing_polyorder=2,
            lower_limb_correction_masks={
                "left_leg": np.ones((sequence.num_frames,), dtype=bool),
                "right_leg": np.zeros((sequence.num_frames,), dtype=bool),
            },
        )

        pose_sequence = result["pose_sequence"]
        quality_report = result["quality_report"]
        left_knee_index = IMUGPT_22_JOINT_NAMES.index("Left_knee")
        left_ankle_index = IMUGPT_22_JOINT_NAMES.index("Left_ankle")
        right_knee_index = IMUGPT_22_JOINT_NAMES.index("Right_knee")
        right_ankle_index = IMUGPT_22_JOINT_NAMES.index("Right_ankle")

        left_tibia_length = np.linalg.norm(
            pose_sequence.joint_positions_xyz[:, left_ankle_index] - pose_sequence.joint_positions_xyz[:, left_knee_index],
            axis=1,
        )
        right_tibia_length = np.linalg.norm(
            pose_sequence.joint_positions_xyz[:, right_ankle_index] - pose_sequence.joint_positions_xyz[:, right_knee_index],
            axis=1,
        )

        self.assertTrue(np.allclose(left_tibia_length, 0.40, atol=0.02))
        self.assertTrue(np.allclose(right_tibia_length, 0.36, atol=0.02))
        self.assertEqual(quality_report["tibia_prior_applied_frames"], sequence.num_frames)
        self.assertEqual(quality_report["corrected_smoothing_window_length"], 3)
        self.assertTrue(
            np.all(result["artifacts"]["tibia_prior_applied_mask"]["left_leg"])
        )
        self.assertFalse(
            np.any(result["artifacts"]["tibia_prior_applied_mask"]["right_leg"])
        )

    def test_run_metric_normalizer_excludes_imputed_femur_from_scale_estimate(self) -> None:
        sequence = _make_imugpt22_sequence()
        left_hip_index = IMUGPT_22_JOINT_NAMES.index("Left_hip")
        left_knee_index = IMUGPT_22_JOINT_NAMES.index("Left_knee")
        joint_confidence = np.asarray(sequence.joint_confidence, dtype=np.float32).copy()
        imputed_mask = np.zeros((sequence.num_frames, len(IMUGPT_22_JOINT_NAMES)), dtype=bool)
        observed_mask = np.ones((sequence.num_frames, len(IMUGPT_22_JOINT_NAMES)), dtype=bool)
        joint_confidence[:, left_hip_index] = 0.05
        joint_confidence[:, left_knee_index] = 0.05
        observed_mask[:, left_hip_index] = False
        observed_mask[:, left_knee_index] = False
        imputed_mask[:, left_hip_index] = True
        imputed_mask[:, left_knee_index] = True

        sequence = PoseSequence3D(
            clip_id=sequence.clip_id,
            fps=sequence.fps,
            fps_original=sequence.fps_original,
            joint_names_3d=sequence.joint_names_3d,
            joint_positions_xyz=np.asarray(sequence.joint_positions_xyz, dtype=np.float32),
            joint_confidence=joint_confidence,
            skeleton_parents=sequence.skeleton_parents,
            frame_indices=sequence.frame_indices,
            timestamps_sec=sequence.timestamps_sec,
            source=sequence.source,
            coordinate_space=sequence.coordinate_space,
            observed_mask=observed_mask,
            imputed_mask=imputed_mask,
        )

        result = run_metric_normalizer(
            sequence,
            target_femur_length_m=0.45,
            smoothing_window_length=5,
            smoothing_polyorder=2,
        )

        self.assertAlmostEqual(float(result["normalization_result"]["scale_factor"]), 0.9, places=4)
        self.assertNotIn("scale_factor_fallback_to_identity", result["quality_report"]["notes"])


if __name__ == "__main__":
    unittest.main()
