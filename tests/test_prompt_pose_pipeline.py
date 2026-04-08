import tempfile
import unittest
from pathlib import Path

import numpy as np

from pose_module.interfaces import IMUGPT_22_JOINT_NAMES
from pose_module.pipeline import generate_pose_from_prompt, run_pose3d_from_prompt


def _make_prompt_xyz(num_frames: int = 12) -> np.ndarray:
    positions = np.zeros((num_frames, len(IMUGPT_22_JOINT_NAMES), 3), dtype=np.float32)
    base_pose = {
        "Pelvis": [0.00, 1.00, 0.00],
        "Left_hip": [-0.10, 0.95, 0.00],
        "Right_hip": [0.10, 0.95, 0.00],
        "Spine1": [0.00, 1.10, 0.00],
        "Left_knee": [-0.10, 0.55, 0.04],
        "Right_knee": [0.10, 0.55, -0.04],
        "Spine2": [0.00, 1.22, 0.00],
        "Left_ankle": [-0.10, 0.12, 0.06],
        "Right_ankle": [0.10, 0.12, -0.06],
        "Spine3": [0.00, 1.34, 0.00],
        "Left_foot": [-0.10, 0.02, 0.18],
        "Right_foot": [0.10, 0.02, 0.08],
        "Neck": [0.00, 1.48, 0.00],
        "Left_collar": [-0.08, 1.44, 0.00],
        "Right_collar": [0.08, 1.44, 0.00],
        "Head": [0.00, 1.68, 0.00],
        "Left_shoulder": [-0.20, 1.42, 0.00],
        "Right_shoulder": [0.20, 1.42, 0.00],
        "Left_elbow": [-0.42, 1.22, 0.05],
        "Right_elbow": [0.42, 1.22, 0.05],
        "Left_wrist": [-0.58, 1.02, 0.08],
        "Right_wrist": [0.58, 1.02, 0.08],
    }
    joint_index = {name: index for index, name in enumerate(IMUGPT_22_JOINT_NAMES)}
    for frame_index in range(num_frames):
        forward = 0.05 * frame_index
        arm_phase = np.float32(np.sin(frame_index / 2.0))
        for joint_name, xyz in base_pose.items():
            positions[frame_index, joint_index[joint_name]] = np.asarray(
                [xyz[0], xyz[1], xyz[2] + forward],
                dtype=np.float32,
            )
        positions[frame_index, joint_index["Left_wrist"], 1] += 0.10 * arm_phase
        positions[frame_index, joint_index["Right_wrist"], 1] -= 0.08 * arm_phase
        positions[frame_index, joint_index["Left_elbow"], 1] += 0.06 * arm_phase
        positions[frame_index, joint_index["Right_elbow"], 1] -= 0.05 * arm_phase
        positions[frame_index, joint_index["Left_ankle"], 2] += 0.03 * arm_phase
        positions[frame_index, joint_index["Right_ankle"], 2] -= 0.03 * arm_phase
        positions[frame_index, joint_index["Left_foot"], 2] += 0.05 * arm_phase
        positions[frame_index, joint_index["Right_foot"], 2] -= 0.05 * arm_phase
    return positions


class _FakePromptBackend:
    def generate(self, *, prompt_text: str, seed: int, fps: float, duration_hint_sec=None, output_dir=None):
        del prompt_text, seed, fps, duration_hint_sec, output_dir
        return {
            "joint_positions_xyz": _make_prompt_xyz(),
            "generation_backend": "fake_t2mgpt",
            "backend_report": {"backend_name": "fake_t2mgpt"},
            "artifacts": {},
        }


class PromptPosePipelineTests(unittest.TestCase):
    def test_run_pose3d_from_prompt_writes_phase1_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_pose3d_from_prompt(
                prompt_id="happy_walk_001",
                prompt_text="A person walks happily with energetic arm swings.",
                output_dir=tmp_dir,
                labels={
                    "emotion": "happiness",
                    "modality": "walking",
                    "stimulus": "music",
                },
                seed=7,
                fps=20.0,
                prompt_backend=_FakePromptBackend(),
            )

            self.assertEqual(result["clip_id"], "happy_walk_001")
            self.assertEqual(result["quality_report"]["source_kind"], "prompt")
            self.assertEqual(result["quality_report"]["modality_domain"], "synthetic")
            self.assertEqual(result["pose_sequence"].coordinate_space, "pseudo_global_metric")
            self.assertTrue(Path(result["artifacts"]["prompt_metadata_json_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["pose3d_prompt_raw_npz_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["pose3d_metric_local_npz_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["pose3d_npz_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["quality_report_json_path"]).exists())
            self.assertEqual(result["prompt_metadata"]["generation_backend"], "fake_t2mgpt")

    def test_generate_pose_from_prompt_returns_final_pose_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            sequence = generate_pose_from_prompt(
                "A person stands happily with animated gestures.",
                "happy_standing_001",
                output_dir=tmp_dir,
                prompt_backend=_FakePromptBackend(),
            )
            self.assertEqual(sequence.clip_id, "happy_standing_001")
            self.assertEqual(list(sequence.joint_names_3d), list(IMUGPT_22_JOINT_NAMES))
            self.assertEqual(sequence.coordinate_space, "pseudo_global_metric")
