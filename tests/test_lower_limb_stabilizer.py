import unittest

import numpy as np

from pose_module.interfaces import MOTIONBERT_17_JOINT_NAMES, MOTIONBERT_17_PARENT_INDICES, PoseSequence3D
from pose_module.processing.lower_limb_stabilizer import run_lower_limb_stabilizer


_BASE_POINTS = {
    "pelvis": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    "left_hip": np.asarray([-0.15, 0.0, 0.0], dtype=np.float32),
    "right_hip": np.asarray([0.15, 0.0, 0.0], dtype=np.float32),
    "spine": np.asarray([0.0, 0.2, 0.0], dtype=np.float32),
    "left_knee": np.asarray([-0.15, -0.35, 0.0], dtype=np.float32),
    "right_knee": np.asarray([0.15, -0.35, 0.0], dtype=np.float32),
    "thorax": np.asarray([0.0, 0.45, 0.0], dtype=np.float32),
    "left_ankle": np.asarray([0.05, -0.55, 0.0], dtype=np.float32),
    "right_ankle": np.asarray([-0.05, -0.55, 0.0], dtype=np.float32),
    "neck": np.asarray([0.0, 0.65, 0.0], dtype=np.float32),
    "head": np.asarray([0.0, 0.82, 0.0], dtype=np.float32),
    "left_shoulder": np.asarray([-0.2, 0.45, 0.0], dtype=np.float32),
    "right_shoulder": np.asarray([0.2, 0.45, 0.0], dtype=np.float32),
    "left_elbow": np.asarray([-0.35, 0.3, 0.0], dtype=np.float32),
    "right_elbow": np.asarray([0.35, 0.3, 0.0], dtype=np.float32),
    "left_wrist": np.asarray([-0.45, 0.15, 0.0], dtype=np.float32),
    "right_wrist": np.asarray([0.45, 0.15, 0.0], dtype=np.float32),
}

_STANDING_ANKLES = {
    "left_ankle": np.asarray([-0.15, -0.72, 0.0], dtype=np.float32),
    "right_ankle": np.asarray([0.15, -0.72, 0.0], dtype=np.float32),
}


def _make_sequence(*, seated_history_frames: int, uncertain_frames: int) -> PoseSequence3D:
    num_frames = seated_history_frames + uncertain_frames
    joint_positions_xyz = np.zeros((num_frames, len(MOTIONBERT_17_JOINT_NAMES), 3), dtype=np.float32)
    joint_confidence = np.full((num_frames, len(MOTIONBERT_17_JOINT_NAMES)), 0.95, dtype=np.float32)
    observed_mask = np.ones((num_frames, len(MOTIONBERT_17_JOINT_NAMES)), dtype=bool)
    imputed_mask = np.zeros((num_frames, len(MOTIONBERT_17_JOINT_NAMES)), dtype=bool)

    for frame_index in range(num_frames):
        for joint_index, joint_name in enumerate(MOTIONBERT_17_JOINT_NAMES):
            point = _BASE_POINTS[joint_name].copy()
            point[2] += np.float32(frame_index) * np.float32(0.01)
            joint_positions_xyz[frame_index, joint_index] = point

    if uncertain_frames > 0:
        lower_leg_indices = [
            MOTIONBERT_17_JOINT_NAMES.index("left_knee"),
            MOTIONBERT_17_JOINT_NAMES.index("right_knee"),
            MOTIONBERT_17_JOINT_NAMES.index("left_ankle"),
            MOTIONBERT_17_JOINT_NAMES.index("right_ankle"),
        ]
        for frame_index in range(seated_history_frames, num_frames):
            joint_positions_xyz[frame_index, MOTIONBERT_17_JOINT_NAMES.index("left_ankle")] = _STANDING_ANKLES["left_ankle"]
            joint_positions_xyz[frame_index, MOTIONBERT_17_JOINT_NAMES.index("right_ankle")] = _STANDING_ANKLES["right_ankle"]
            joint_positions_xyz[frame_index, MOTIONBERT_17_JOINT_NAMES.index("left_ankle"), 2] += (
                np.float32(frame_index) * np.float32(0.01)
            )
            joint_positions_xyz[frame_index, MOTIONBERT_17_JOINT_NAMES.index("right_ankle"), 2] += (
                np.float32(frame_index) * np.float32(0.01)
            )
            joint_confidence[frame_index, lower_leg_indices] = 0.05
            observed_mask[frame_index, lower_leg_indices] = False
            imputed_mask[frame_index, lower_leg_indices] = True

    return PoseSequence3D(
        clip_id="clip_lower_limb_stabilizer",
        fps=20.0,
        fps_original=30.0,
        joint_names_3d=list(MOTIONBERT_17_JOINT_NAMES),
        joint_positions_xyz=joint_positions_xyz,
        joint_confidence=joint_confidence,
        skeleton_parents=list(MOTIONBERT_17_PARENT_INDICES),
        frame_indices=np.arange(num_frames, dtype=np.int32),
        timestamps_sec=np.arange(num_frames, dtype=np.float32) / 20.0,
        source="unit_test_motionbert",
        coordinate_space="pose_lifter_aligned",
        observed_mask=observed_mask,
        imputed_mask=imputed_mask,
    )


class LowerLimbStabilizerTests(unittest.TestCase):
    def test_stabilizer_uses_last_reliable_seated_history_under_occlusion(self) -> None:
        sequence = _make_sequence(seated_history_frames=4, uncertain_frames=3)

        result = run_lower_limb_stabilizer(sequence)
        report = result["quality_report"]["lower_body_report"]

        left_tail_angles = np.asarray(report["left_leg"]["knee_angle_deg"][-3:], dtype=np.float32)
        right_tail_angles = np.asarray(report["right_leg"]["knee_angle_deg"][-3:], dtype=np.float32)
        left_tail_posture = report["left_leg"]["posture_state"][-3:]
        right_tail_posture = report["right_leg"]["posture_state"][-3:]

        self.assertTrue(np.all(left_tail_angles < 140.0))
        self.assertTrue(np.all(right_tail_angles < 140.0))
        self.assertEqual(left_tail_posture, ["seated", "seated", "seated"])
        self.assertEqual(right_tail_posture, ["seated", "seated", "seated"])
        self.assertEqual(result["quality_report"]["left_leg_correction_frames"], 3)
        self.assertEqual(result["quality_report"]["right_leg_correction_frames"], 3)

    def test_stabilizer_keeps_uncertain_leg_below_full_extension_without_history(self) -> None:
        sequence = _make_sequence(seated_history_frames=0, uncertain_frames=4)

        result = run_lower_limb_stabilizer(sequence)
        report = result["quality_report"]["lower_body_report"]

        left_angles = np.asarray(report["left_leg"]["knee_angle_deg"], dtype=np.float32)
        right_angles = np.asarray(report["right_leg"]["knee_angle_deg"], dtype=np.float32)

        self.assertTrue(np.all(left_angles < 160.0))
        self.assertTrue(np.all(right_angles < 160.0))
        self.assertEqual(report["left_leg"]["posture_state"], ["uncertain"] * 4)
        self.assertEqual(report["right_leg"]["posture_state"], ["uncertain"] * 4)
        self.assertTrue(np.all(result["pose_sequence"].joint_confidence[:, MOTIONBERT_17_JOINT_NAMES.index("left_ankle")] <= 0.10))
        self.assertTrue(np.all(result["pose_sequence"].joint_confidence[:, MOTIONBERT_17_JOINT_NAMES.index("right_ankle")] <= 0.10))


if __name__ == "__main__":
    unittest.main()
