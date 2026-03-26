import unittest

import numpy as np

from pose_module.interfaces import (
    IMUGPT_22_JOINT_NAMES,
    IMUGPT_22_PARENT_INDICES,
    MOTIONBERT_17_JOINT_NAMES,
    MOTIONBERT_17_PARENT_INDICES,
    PoseSequence3D,
)
from pose_module.processing.skeleton_mapper import map_pose_sequence_to_imugpt22


_MB17_POINTS = {
    "pelvis": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    "left_hip": np.asarray([-0.2, 0.0, 0.0], dtype=np.float32),
    "right_hip": np.asarray([0.2, 0.0, 0.0], dtype=np.float32),
    "spine": np.asarray([0.0, 0.2, 0.0], dtype=np.float32),
    "left_knee": np.asarray([-0.2, -0.4, 0.0], dtype=np.float32),
    "right_knee": np.asarray([0.2, -0.4, 0.0], dtype=np.float32),
    "thorax": np.asarray([0.0, 0.6, 0.0], dtype=np.float32),
    "left_ankle": np.asarray([-0.2, -0.8, 0.0], dtype=np.float32),
    "right_ankle": np.asarray([0.2, -0.8, 0.0], dtype=np.float32),
    "neck": np.asarray([0.0, 0.8, 0.0], dtype=np.float32),
    "head": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    "left_shoulder": np.asarray([-0.35, 0.6, 0.0], dtype=np.float32),
    "right_shoulder": np.asarray([0.35, 0.6, 0.0], dtype=np.float32),
    "left_elbow": np.asarray([-0.55, 0.45, 0.0], dtype=np.float32),
    "right_elbow": np.asarray([0.55, 0.45, 0.0], dtype=np.float32),
    "left_wrist": np.asarray([-0.75, 0.3, 0.0], dtype=np.float32),
    "right_wrist": np.asarray([0.75, 0.3, 0.0], dtype=np.float32),
}


def _make_mb17_pose_sequence(*, num_frames: int = 3, flip_handedness: bool = False) -> PoseSequence3D:
    joint_positions_xyz = np.zeros((num_frames, len(MOTIONBERT_17_JOINT_NAMES), 3), dtype=np.float32)
    joint_confidence = np.full((num_frames, len(MOTIONBERT_17_JOINT_NAMES)), 0.9, dtype=np.float32)

    for frame_index in range(num_frames):
        depth_offset = np.float32(frame_index) * np.float32(0.01)
        for joint_index, joint_name in enumerate(MOTIONBERT_17_JOINT_NAMES):
            point = _MB17_POINTS[joint_name].copy()
            point[2] += depth_offset
            joint_positions_xyz[frame_index, joint_index] = point

    if flip_handedness:
        for left_name, right_name in (
            ("left_hip", "right_hip"),
            ("left_knee", "right_knee"),
            ("left_ankle", "right_ankle"),
            ("left_shoulder", "right_shoulder"),
            ("left_elbow", "right_elbow"),
            ("left_wrist", "right_wrist"),
        ):
            left_index = MOTIONBERT_17_JOINT_NAMES.index(left_name)
            right_index = MOTIONBERT_17_JOINT_NAMES.index(right_name)
            left_points = joint_positions_xyz[:, left_index].copy()
            right_points = joint_positions_xyz[:, right_index].copy()
            joint_positions_xyz[:, left_index] = right_points
            joint_positions_xyz[:, right_index] = left_points

    return PoseSequence3D(
        clip_id="clip_mapper",
        fps=20.0,
        fps_original=30.0,
        joint_names_3d=list(MOTIONBERT_17_JOINT_NAMES),
        joint_positions_xyz=joint_positions_xyz,
        joint_confidence=joint_confidence,
        skeleton_parents=list(MOTIONBERT_17_PARENT_INDICES),
        frame_indices=np.arange(num_frames, dtype=np.int32),
        timestamps_sec=np.arange(num_frames, dtype=np.float32) / 20.0,
        source="vitpose-b_motionbert17_clean_mmpose_motionbert",
        coordinate_space="camera",
    )


class SkeletonMapperTests(unittest.TestCase):
    def test_map_pose_sequence_to_imugpt22_expands_the_fixed_contract(self) -> None:
        sequence = _make_mb17_pose_sequence()

        mapped_sequence, quality_report, artifacts = map_pose_sequence_to_imugpt22(sequence)

        self.assertEqual(mapped_sequence.joint_names_3d, list(IMUGPT_22_JOINT_NAMES))
        self.assertEqual(mapped_sequence.skeleton_parents, list(IMUGPT_22_PARENT_INDICES))
        self.assertEqual(mapped_sequence.joint_positions_xyz.shape, (3, 22, 3))
        self.assertEqual(mapped_sequence.joint_confidence.shape, (3, 22))
        self.assertEqual(quality_report["status"], "ok")
        self.assertTrue(quality_report["skeleton_mapping_ok"])
        self.assertEqual(int(np.count_nonzero(artifacts["handedness_swap_mask"])), 0)

        spine1 = mapped_sequence.joint_positions_xyz[0, IMUGPT_22_JOINT_NAMES.index("Spine1")]
        spine2 = mapped_sequence.joint_positions_xyz[0, IMUGPT_22_JOINT_NAMES.index("Spine2")]
        spine3 = mapped_sequence.joint_positions_xyz[0, IMUGPT_22_JOINT_NAMES.index("Spine3")]
        left_collar = mapped_sequence.joint_positions_xyz[0, IMUGPT_22_JOINT_NAMES.index("Left_collar")]
        left_foot = mapped_sequence.joint_positions_xyz[0, IMUGPT_22_JOINT_NAMES.index("Left_foot")]

        np.testing.assert_allclose(spine1, np.asarray([0.0, 0.2, 0.0], dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(spine2, np.asarray([0.0, 0.4, 0.0], dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(spine3, np.asarray([0.0, 0.6, 0.0], dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(left_collar, np.asarray([-0.175, 0.6, 0.0], dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(left_foot, np.asarray([-0.2, -0.8, 0.12], dtype=np.float32), atol=1e-5)

        self.assertAlmostEqual(
            float(mapped_sequence.joint_confidence[0, IMUGPT_22_JOINT_NAMES.index("Spine1")]),
            0.9,
            places=5,
        )
        self.assertAlmostEqual(
            float(mapped_sequence.joint_confidence[0, IMUGPT_22_JOINT_NAMES.index("Left_collar")]),
            0.9,
            places=5,
        )
        self.assertAlmostEqual(
            float(mapped_sequence.joint_confidence[0, IMUGPT_22_JOINT_NAMES.index("Left_foot")]),
            0.72,
            places=5,
        )
        self.assertFalse(bool(mapped_sequence.observed_mask[0, IMUGPT_22_JOINT_NAMES.index("Left_foot")]))
        self.assertTrue(bool(mapped_sequence.imputed_mask[0, IMUGPT_22_JOINT_NAMES.index("Left_foot")]))

    def test_map_pose_sequence_to_imugpt22_corrects_flipped_handedness(self) -> None:
        sequence = _make_mb17_pose_sequence(flip_handedness=True)

        mapped_sequence, quality_report, artifacts = map_pose_sequence_to_imugpt22(sequence)

        self.assertEqual(quality_report["status"], "warning")
        self.assertTrue(quality_report["skeleton_mapping_ok"])
        self.assertEqual(quality_report["handedness_swapped_frames"], sequence.num_frames)
        self.assertEqual(int(np.count_nonzero(artifacts["handedness_swap_mask"])), sequence.num_frames)
        self.assertIn(
            f"handedness_swapped_frames:{sequence.num_frames}",
            quality_report["notes"],
        )

        left_hip_x = mapped_sequence.joint_positions_xyz[:, IMUGPT_22_JOINT_NAMES.index("Left_hip"), 0]
        right_hip_x = mapped_sequence.joint_positions_xyz[:, IMUGPT_22_JOINT_NAMES.index("Right_hip"), 0]
        self.assertTrue(np.all(right_hip_x > left_hip_x))

    def test_map_pose_sequence_to_imugpt22_propagates_low_confidence_ankle_to_synthetic_foot(self) -> None:
        sequence = _make_mb17_pose_sequence()
        left_ankle_index = MOTIONBERT_17_JOINT_NAMES.index("left_ankle")
        observed_mask = np.ones((sequence.num_frames, len(MOTIONBERT_17_JOINT_NAMES)), dtype=bool)
        imputed_mask = np.zeros((sequence.num_frames, len(MOTIONBERT_17_JOINT_NAMES)), dtype=bool)
        joint_confidence = np.asarray(sequence.joint_confidence, dtype=np.float32).copy()
        joint_confidence[:, left_ankle_index] = 0.1
        observed_mask[:, left_ankle_index] = False
        imputed_mask[:, left_ankle_index] = True
        sequence = PoseSequence3D(
            clip_id=sequence.clip_id,
            fps=sequence.fps,
            fps_original=sequence.fps_original,
            joint_names_3d=sequence.joint_names_3d,
            joint_positions_xyz=sequence.joint_positions_xyz,
            joint_confidence=joint_confidence,
            skeleton_parents=sequence.skeleton_parents,
            frame_indices=sequence.frame_indices,
            timestamps_sec=sequence.timestamps_sec,
            source=sequence.source,
            coordinate_space=sequence.coordinate_space,
            observed_mask=observed_mask,
            imputed_mask=imputed_mask,
        )

        mapped_sequence, _, _ = map_pose_sequence_to_imugpt22(sequence)

        left_foot_index = IMUGPT_22_JOINT_NAMES.index("Left_foot")
        self.assertTrue(np.allclose(mapped_sequence.joint_confidence[:, left_foot_index], 0.08, atol=1e-6))
        self.assertTrue(np.all(mapped_sequence.imputed_mask[:, left_foot_index]))
        self.assertTrue(np.all(~mapped_sequence.observed_mask[:, left_foot_index]))


if __name__ == "__main__":
    unittest.main()
