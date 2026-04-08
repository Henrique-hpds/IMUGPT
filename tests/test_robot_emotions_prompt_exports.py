import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pose_module.interfaces import IMUGPT_22_JOINT_NAMES, IMUGPT_22_PARENT_INDICES, PoseSequence3D
from pose_module.motionbert.adapter import write_pose_sequence3d_npz
from pose_module.robot_emotions.cli import main as robot_emotions_cli_main
from pose_module.robot_emotions.extractor import RobotEmotionsClipRecord
from pose_module.robot_emotions.prompt_exports import (
    build_robot_emotions_prompt_catalog,
    run_robot_emotions_prompt_pose3d,
)


def _make_record(*, clip_id: str, domain: str, tag_number: int) -> RobotEmotionsClipRecord:
    return RobotEmotionsClipRecord(
        clip_id=clip_id,
        domain=domain,
        user_id=2,
        tag_number=tag_number,
        tag_dir=Path(f"data/RobotEmotions/{domain}/User2/Tag{tag_number}"),
        imu_csv_path=Path(f"data/RobotEmotions/{domain}/User2/Tag{tag_number}/ESP_2_{tag_number}.csv"),
        video_path=Path(f"data/RobotEmotions/{domain}/User2/Tag{tag_number}/TAG_2_{tag_number}.mp4"),
        source_rel_dir=f"{domain}/User2/Tag{tag_number}",
        take_id=None,
        participant={"name": "Test User", "age": 20, "gender": "F"},
        protocol={},
    )


def _make_pose_sequence(clip_id: str) -> PoseSequence3D:
    num_frames = 10
    num_joints = len(IMUGPT_22_JOINT_NAMES)
    joint_positions_xyz = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    root_translation = np.zeros((num_frames, 3), dtype=np.float32)
    for frame_index in range(num_frames):
        root_translation[frame_index] = np.asarray([0.0, 0.0, 0.03 * frame_index], dtype=np.float32)
        joint_positions_xyz[frame_index, :, 0] = np.linspace(-0.4, 0.4, num_joints, dtype=np.float32)
        joint_positions_xyz[frame_index, :, 1] = np.linspace(1.0, 0.1, num_joints, dtype=np.float32)
        joint_positions_xyz[frame_index, :, 2] = root_translation[frame_index, 2]
        joint_positions_xyz[frame_index, 0] = root_translation[frame_index]
        joint_positions_xyz[frame_index, IMUGPT_22_JOINT_NAMES.index("Left_shoulder")] = [-0.20, 1.35, root_translation[frame_index, 2]]
        joint_positions_xyz[frame_index, IMUGPT_22_JOINT_NAMES.index("Right_shoulder")] = [0.20, 1.35, root_translation[frame_index, 2]]
        joint_positions_xyz[frame_index, IMUGPT_22_JOINT_NAMES.index("Left_elbow")] = [-0.22, 1.55, root_translation[frame_index, 2]]
        joint_positions_xyz[frame_index, IMUGPT_22_JOINT_NAMES.index("Right_elbow")] = [0.22, 1.55, root_translation[frame_index, 2]]
        joint_positions_xyz[frame_index, IMUGPT_22_JOINT_NAMES.index("Left_wrist")] = [-0.18, 1.78, root_translation[frame_index, 2]]
        joint_positions_xyz[frame_index, IMUGPT_22_JOINT_NAMES.index("Right_wrist")] = [0.18, 1.78, root_translation[frame_index, 2]]
    return PoseSequence3D(
        clip_id=clip_id,
        fps=20.0,
        fps_original=20.0,
        joint_names_3d=list(IMUGPT_22_JOINT_NAMES),
        joint_positions_xyz=joint_positions_xyz,
        joint_confidence=np.ones((num_frames, num_joints), dtype=np.float32),
        skeleton_parents=list(IMUGPT_22_PARENT_INDICES),
        frame_indices=np.arange(num_frames, dtype=np.int32),
        timestamps_sec=np.arange(num_frames, dtype=np.float32) / np.float32(20.0),
        source="unit_test_prompt_source",
        coordinate_space="pseudo_global_metric",
        root_translation_m=root_translation,
    )


class _FakeExtractor:
    def __init__(self, dataset_root: str, *, domains: tuple[str, ...]) -> None:
        self.dataset_root = dataset_root
        self.domains = domains

    def scan(self):
        return [
            _make_record(
                clip_id="robot_emotions_10ms_u02_tag11",
                domain="10ms",
                tag_number=11,
            )
        ]


class RobotEmotionsPromptExportTests(unittest.TestCase):
    def test_build_prompt_catalog_uses_real_pose_stats_to_enrich_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pose_path = Path(tmp_dir) / "pose3d.npz"
            write_pose_sequence3d_npz(_make_pose_sequence("robot_emotions_10ms_u02_tag11"), pose_path)
            manifest_path = Path(tmp_dir) / "pose3d_manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "clip_id": "robot_emotions_10ms_u02_tag11",
                        "status": "ok",
                        "artifacts": {"pose3d_npz_path": str(pose_path.resolve())},
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            catalog_path = Path(tmp_dir) / "robot_emotions_prompts.jsonl"
            with patch("pose_module.robot_emotions.prompt_exports.RobotEmotionsExtractor", _FakeExtractor):
                summary = build_robot_emotions_prompt_catalog(
                    dataset_root="data/RobotEmotions",
                    output_path=catalog_path,
                    real_pose3d_manifest_path=manifest_path,
                    domains=("10ms",),
                )

            self.assertEqual(summary["num_conditions"], 18)
            lines = [json.loads(line) for line in catalog_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            enriched_entry = next(
                line for line in lines if line["action_detail"] == "happy_standing_arms_raised"
            )
            self.assertEqual(enriched_entry["labels"]["emotion"], "happiness")
            self.assertEqual(enriched_entry["labels"]["modality"], "standing")
            self.assertIn("arms above shoulder level", enriched_entry["prompt_text"])

    def test_run_robot_emotions_prompt_pose3d_writes_manifest(self) -> None:
        fake_pose_sequence = _make_pose_sequence("happy_prompt_sample")
        fake_pipeline_result = {
            "clip_id": "happy_prompt_sample",
            "pose_sequence": fake_pose_sequence,
            "quality_report": {"clip_id": "happy_prompt_sample", "status": "ok"},
            "prompt_adapter_quality_report": {"clip_id": "happy_prompt_sample", "status": "ok"},
            "prompt_source_quality_report": {"clip_id": "happy_prompt_sample", "status": "ok"},
            "metric_normalization_quality_report": {"clip_id": "happy_prompt_sample", "status": "ok"},
            "root_trajectory_quality_report": {"clip_id": "happy_prompt_sample", "status": "ok"},
            "prompt_metadata": {"generation_backend": "fake_t2mgpt"},
            "artifacts": {
                "pose3d_prompt_raw_npz_path": "/tmp/fake_pose3d_prompt_raw.npz",
                "pose3d_metric_local_npz_path": "/tmp/fake_pose3d_metric_local.npz",
                "pose3d_npz_path": "/tmp/fake_pose3d.npz",
                "pose3d_bvh_path": "/tmp/fake_pose3d.bvh",
                "quality_report_json_path": "/tmp/fake_quality_report.json",
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            catalog_path = Path(tmp_dir) / "prompt_catalog.jsonl"
            catalog_path.write_text(
                json.dumps(
                    {
                        "prompt_id": "happy_prompt",
                        "prompt_text": "A person is standing and expressing happiness with expansive movement.",
                        "labels": {
                            "emotion": "happiness",
                            "modality": "standing",
                            "stimulus": "autobiographical_recall",
                        },
                        "seed": 123,
                        "num_samples": 1,
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            with patch(
                "pose_module.robot_emotions.prompt_exports.run_pose3d_from_prompt",
                return_value=fake_pipeline_result,
            ):
                summary = run_robot_emotions_prompt_pose3d(
                    prompt_catalog_path=catalog_path,
                    output_dir=tmp_dir,
                )

            self.assertEqual(summary["num_ok"], 1)
            manifest_entries = [
                json.loads(line)
                for line in Path(summary["prompt_pose3d_manifest_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(manifest_entries[0]["source_kind"], "prompt")
            self.assertEqual(manifest_entries[0]["modality_domain"], "synthetic")
            self.assertEqual(manifest_entries[0]["generation_backend"], "fake_t2mgpt")

    def test_cli_build_prompt_catalog_dispatches_wrapper(self) -> None:
        with patch(
            "pose_module.robot_emotions.cli.build_robot_emotions_prompt_catalog",
            return_value={"status": "ok"},
        ) as mocked_wrapper:
            with patch("builtins.print") as mocked_print:
                exit_code = robot_emotions_cli_main(["build-prompt-catalog"])
        self.assertEqual(exit_code, 0)
        mocked_wrapper.assert_called_once()
        mocked_print.assert_called_once()

    def test_cli_export_prompt_pose3d_dispatches_wrapper(self) -> None:
        with patch(
            "pose_module.robot_emotions.cli.run_robot_emotions_prompt_pose3d",
            return_value={"status": "ok"},
        ) as mocked_wrapper:
            with patch("builtins.print") as mocked_print:
                exit_code = robot_emotions_cli_main(
                    [
                        "export-prompt-pose3d",
                        "--prompt-catalog",
                        "/tmp/prompts.jsonl",
                        "--output-dir",
                        "/tmp/prompt_pose3d",
                    ]
                )
        self.assertEqual(exit_code, 0)
        mocked_wrapper.assert_called_once()
        mocked_print.assert_called_once()
