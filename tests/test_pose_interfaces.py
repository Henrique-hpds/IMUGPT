import unittest

import numpy as np

from pose_module.interfaces import (
    IMUGPT_22_JOINT_NAMES,
    IMUGPT_22_PARENT_INDICES,
    MOTIONBERT_17_JOINT_NAMES,
    PoseSequence2D,
    PoseSequence3D,
)


class PoseInterfaceTests(unittest.TestCase):
    def test_pose_sequence2d_npz_roundtrip_preserves_masks(self) -> None:
        observed_mask = np.asarray([[True] * len(MOTIONBERT_17_JOINT_NAMES)], dtype=bool)
        imputed_mask = np.asarray([[False] * len(MOTIONBERT_17_JOINT_NAMES)], dtype=bool)
        imputed_mask[0, MOTIONBERT_17_JOINT_NAMES.index("left_ankle")] = True
        observed_mask[0, MOTIONBERT_17_JOINT_NAMES.index("left_ankle")] = False
        sequence = PoseSequence2D(
            clip_id="clip_interfaces_2d",
            fps=20.0,
            fps_original=30.0,
            joint_names_2d=list(MOTIONBERT_17_JOINT_NAMES),
            keypoints_xy=np.zeros((1, len(MOTIONBERT_17_JOINT_NAMES), 2), dtype=np.float32),
            confidence=np.ones((1, len(MOTIONBERT_17_JOINT_NAMES)), dtype=np.float32),
            bbox_xywh=np.ones((1, 4), dtype=np.float32),
            frame_indices=np.asarray([0], dtype=np.int32),
            timestamps_sec=np.asarray([0.0], dtype=np.float32),
            source="unit_test",
            observed_mask=observed_mask,
            imputed_mask=imputed_mask,
        )

        roundtrip = PoseSequence2D.from_npz_payload(sequence.to_npz_payload())

        np.testing.assert_array_equal(roundtrip.observed_mask, observed_mask)
        np.testing.assert_array_equal(roundtrip.imputed_mask, imputed_mask)

    def test_pose_sequence3d_from_legacy_npz_payload_defaults_masks_from_confidence(self) -> None:
        payload = {
            "clip_id": np.asarray("clip_interfaces_3d"),
            "fps": np.asarray(20.0, dtype=np.float32),
            "fps_original": np.asarray(30.0, dtype=np.float32),
            "joint_names_3d": np.asarray(list(IMUGPT_22_JOINT_NAMES)),
            "joint_positions_xyz": np.zeros((1, len(IMUGPT_22_JOINT_NAMES), 3), dtype=np.float32),
            "joint_confidence": np.ones((1, len(IMUGPT_22_JOINT_NAMES)), dtype=np.float32),
            "skeleton_parents": np.asarray(list(IMUGPT_22_PARENT_INDICES), dtype=np.int32),
            "frame_indices": np.asarray([0], dtype=np.int32),
            "timestamps_sec": np.asarray([0.0], dtype=np.float32),
            "source": np.asarray("unit_test"),
            "coordinate_space": np.asarray("camera"),
        }
        payload["joint_confidence"][0, IMUGPT_22_JOINT_NAMES.index("Left_ankle")] = 0.0

        sequence = PoseSequence3D.from_npz_payload(payload)

        self.assertIsNone(sequence.observed_mask)
        self.assertIsNone(sequence.imputed_mask)
        self.assertFalse(bool(sequence.resolved_observed_mask()[0, IMUGPT_22_JOINT_NAMES.index("Left_ankle")]))
        self.assertFalse(bool(sequence.resolved_imputed_mask()[0, IMUGPT_22_JOINT_NAMES.index("Left_ankle")]))


if __name__ == "__main__":
    unittest.main()
