from fractions import Fraction
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pose_module.interfaces import IKSequence, IMUGPT_22_JOINT_NAMES, IMUGPT_22_PARENT_INDICES, PoseSequence3D
from pose_module.motionbert.adapter import write_pose_sequence3d_npz
from robot_emotions_vlm.anchor_catalog import SMPLX22_JOINT_NAMES, build_anchor_catalog
from robot_emotions_vlm.cli import describe_videos, main as robot_emotions_vlm_main
from robot_emotions_vlm.dataset import RobotEmotionsDataset
from robot_emotions_vlm.kimodo_generation import (
    _save_smplx_amass_outputs,
    generate_kimodo_from_catalog,
    load_catalog_entries,
)
from robot_emotions_vlm.prompts import render_prompts, sanitize_kimodo_prompt_text
from robot_emotions_vlm.qwen_backend import QwenGenerationConfig, QwenVideoBackend
from robot_emotions_vlm.schemas import DescriptionValidationError, parse_model_response
from robot_emotions_vlm.window_descriptions import _write_video_window_subclip, describe_windows


def _build_dataset_tree(root: Path) -> None:
    clip_root = root / "30ms" / "User6" / "Tag7"
    clip_root.mkdir(parents=True, exist_ok=True)
    (clip_root / "ESP_6_7_2.csv").write_text("timestamp,ax\n0,0\n", encoding="utf-8")
    (clip_root / "TAG_6_7_2.mp4").write_bytes(b"")


def _make_pose_sequence(clip_id: str, *, x_step: float = 0.02, z_step: float = 0.03) -> PoseSequence3D:
    num_frames = 10
    num_joints = len(IMUGPT_22_JOINT_NAMES)
    joint_positions_xyz = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    root_translation = np.zeros((num_frames, 3), dtype=np.float32)
    for frame_index in range(num_frames):
        root_translation[frame_index] = np.asarray([x_step * frame_index, 0.0, z_step * frame_index], dtype=np.float32)
        joint_positions_xyz[frame_index, :, 0] = np.linspace(-0.4, 0.4, num_joints, dtype=np.float32)
        joint_positions_xyz[frame_index, :, 1] = np.linspace(1.0, 0.1, num_joints, dtype=np.float32)
        joint_positions_xyz[frame_index, :, 2] = root_translation[frame_index, 2]
        joint_positions_xyz[frame_index, 0] = root_translation[frame_index]
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
        source="unit_test",
        coordinate_space="pseudo_global_metric",
        root_translation_m=root_translation,
    )


def _make_ik_sequence(sequence: PoseSequence3D) -> IKSequence:
    num_frames = sequence.num_frames
    num_joints = len(IMUGPT_22_JOINT_NAMES)
    local_joint_rotations = np.zeros((num_frames, num_joints, 4), dtype=np.float32)
    local_joint_rotations[..., 0] = 1.0
    return IKSequence(
        clip_id=sequence.clip_id,
        fps=sequence.fps,
        fps_original=sequence.fps_original,
        joint_names_3d=list(IMUGPT_22_JOINT_NAMES),
        local_joint_rotations=local_joint_rotations,
        root_translation_m=np.asarray(sequence.root_translation_m, dtype=np.float32),
        joint_offsets_m=np.zeros((num_joints, 3), dtype=np.float32),
        skeleton_parents=list(IMUGPT_22_PARENT_INDICES),
        frame_indices=np.asarray(sequence.frame_indices, dtype=np.int32),
        timestamps_sec=np.asarray(sequence.timestamps_sec, dtype=np.float32),
        source="unit_test_ik",
    )


def _write_ik_sequence_npz(sequence: IKSequence, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **sequence.to_npz_payload())


def _write_anchor_pose_manifest(path: Path, *, clip_id: str, pose_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "status": "warning",
                "domain": "10ms",
                "user_id": 2,
                "tag_number": 11,
                "artifacts": {"pose3d_npz_path": str(pose_path.resolve())},
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_anchor_qwen_catalog(path: Path, *, prompt_id: str, reference_clip_id: str, window_payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(
            {
                "prompt_id": prompt_id,
                "window_id": prompt_id,
                "prompt_text": "A person stands with open posture, expressive arm motion, upright trunk alignment, lively head orientation, and stable leg support",
                "labels": {"emotion": "happiness", "modality": "standing"},
                "seed": 123,
                "num_samples": 4,
                "reference_clip_id": reference_clip_id,
                "duration_hint_sec": float(window_payload["duration_sec"]),
                "window": window_payload,
                "source_metadata": {"dataset": "RobotEmotions"},
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )


class _FakeBackend:
    def __init__(self, raw_response: str) -> None:
        self.raw_response = raw_response
        self.calls: list[dict[str, str]] = []

    def describe_video(self, video_path: str | Path, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(
            {
                "video_path": str(video_path),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return self.raw_response


class _FakeProcessor:
    def __init__(self) -> None:
        self.video_processor = type("VideoProcessor", (), {"fps": 2})()
        self.apply_calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.apply_calls.append({"messages": messages, **kwargs})
        return {"input_ids": [[1, 2, 3]]}

    def batch_decode(self, generated_ids_trimmed, **kwargs):
        return ['{"prompt_text":"A person moves with compact posture, measured arm motion, steady trunk alignment, attentive head orientation, and stable leg support","dominant_behaviors":["measured motion"],"body_parts":{"arms":"Measured arm motion.","trunk":"Steady trunk alignment.","head":"Attentive head orientation.","legs":"Stable leg support."},"clip_notes":""}']


class _FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        return [[1, 2, 3, 4]]


class _FakeKimodoModel:
    def __init__(self) -> None:
        self.fps = 20.0
        self.skeleton = type("Skeleton", (), {"name": "smplx"})()
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        prompt_text,
        num_frames,
        *,
        num_denoising_steps,
        constraint_lst,
        num_samples,
        multi_prompt,
        post_processing,
        return_numpy,
        **kwargs,
    ):
        self.calls.append(
            {
                "prompt_text": prompt_text,
                "num_frames": num_frames,
                "num_denoising_steps": num_denoising_steps,
                "constraint_lst": constraint_lst,
                "num_samples": num_samples,
                "post_processing": post_processing,
                "kwargs": kwargs,
            }
        )
        return {
            "posed_joints": np.zeros((num_samples, num_frames, 3, 3), dtype=np.float32),
            "global_rot_mats": np.zeros((num_samples, num_frames, 3, 3, 3), dtype=np.float32),
            "local_rot_mats": np.zeros((num_samples, num_frames, 3, 3, 3), dtype=np.float32),
            "root_positions": np.zeros((num_samples, num_frames, 3), dtype=np.float32),
        }


class _FakeKimodoRuntime:
    def __init__(self) -> None:
        self.default_model = "Kimodo-SMPLX-RP-v1"
        self.model = _FakeKimodoModel()
        self.seed_calls: list[int] = []
        self.save_calls: list[dict[str, object]] = []
        self.constraint_load_calls: list[dict[str, object]] = []
        self.loaded_model_name = None

    def resolve_device(self) -> str:
        return "cpu"

    def load_model(self, model_name, **kwargs):
        self.loaded_model_name = model_name
        return self.model, "kimodo-smplx-rp"

    def get_model_info(self, resolved_model):
        return type("Info", (), {"display_name": "Kimodo-SMPLX-RP-v1", "skeleton": "SMPLX"})()

    def seed_everything(self, seed: int) -> None:
        self.seed_calls.append(seed)

    def load_constraints(self, path: str | Path, *, skeleton, device: str) -> list[dict[str, object]]:
        self.constraint_load_calls.append(
            {
                "path": str(path),
                "skeleton_name": getattr(skeleton, "name", None),
                "device": device,
            }
        )
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = [payload]
        return payload

    def save_outputs(self, **kwargs):
        self.save_calls.append(kwargs)
        output_stem = Path(kwargs["output_stem"])
        npz_path = output_stem.with_suffix(".npz")
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        npz_path.write_bytes(b"fake_npz")
        return {"kimodo_npz_path": str(npz_path.resolve())}


class _FakeAnchorRuntime:
    def __init__(self, *, fps: float = 20.0) -> None:
        self.default_model = "Kimodo-SMPLX-RP-v1"
        self.model = type(
            "Model",
            (),
            {
                "fps": fps,
                "skeleton": type("Skeleton", (), {"name": "smplx"})(),
            },
        )()
        self.loaded_model_name = None

    def resolve_device(self) -> str:
        return "cpu"

    def load_model(self, model_name, **kwargs):
        self.loaded_model_name = model_name
        return self.model, "kimodo-smplx-rp"

    def get_model_info(self, resolved_model):
        return type("Info", (), {"display_name": "Kimodo-SMPLX-RP-v1", "skeleton": "SMPLX"})()


class _FakeAMASSConverter:
    saved_paths: list[str] = []

    def __init__(self, skeleton, fps) -> None:
        self.skeleton = skeleton
        self.fps = fps

    def convert_save_npz(self, output, path: str) -> None:
        self.__class__.saved_paths.append(path)
        Path(path).write_bytes(b"fake_amass")


class RobotEmotionsVLMTests(unittest.TestCase):
    def test_dataset_scan_discovers_multi_take_clip_id_and_supports_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "RobotEmotions"
            _build_dataset_tree(dataset_root)

            dataset = RobotEmotionsDataset(dataset_root)
            records = dataset.scan()

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].clip_id, "robot_emotions_30ms_u06_tag07_2")
            filtered = dataset.select_records(clip_ids=["robot_emotions_30ms_u06_tag07_2"])
            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0].labels["emotion"], "sadness")

    def test_render_prompts_includes_current_clip_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "RobotEmotions"
            _build_dataset_tree(dataset_root)
            record = RobotEmotionsDataset(dataset_root).scan()[0]

            bundle = render_prompts(
                record,
                {
                    "available": True,
                    "video_path": str(record.video_path),
                    "fps": 30.0,
                    "num_frames": 300,
                    "duration_sec": 10.0,
                    "width": 1920,
                    "height": 1080,
                },
            )

            self.assertIn("robot_emotions_30ms_u06_tag07_2", bundle["user_prompt"])
            self.assertIn('"emotion": "sadness"', bundle["user_prompt"])
            self.assertIn("Return exactly one JSON object", bundle["system_prompt"])

    def test_parse_model_response_normalizes_prompt_text_and_reads_body_parts(self) -> None:
        raw_response = """
        ```json
        {
          "prompt_text": "A person stands with tense posture, sharp arm gestures, slight forward trunk lean, alert head orientation, and grounded leg support.",
          "dominant_behaviors": ["tense standing", "sharp gestures"],
          "body_parts": {
            "arms": "Arms move with quick sharp gestures.",
            "trunk": "The trunk keeps a slight forward lean.",
            "head": "The head stays alert and forward oriented.",
            "legs": "The legs remain planted with grounded support."
          },
          "clip_notes": "The movement feels concentrated and tense."
        }
        ```
        """

        parsed = parse_model_response(raw_response)

        self.assertEqual(
            parsed.description.prompt_text,
            "A person stands with tense posture, sharp arm gestures, slight forward trunk lean, alert head orientation, and grounded leg support",
        )
        self.assertIn("prompt_text_trailing_period_removed", parsed.warnings)
        self.assertEqual(parsed.description.body_parts.arms, "Arms move with quick sharp gestures.")

    def test_parse_model_response_requires_all_body_parts(self) -> None:
        raw_response = json.dumps(
            {
                "prompt_text": "A person walks with a measured gait, low arm swing, compact trunk motion, steady head orientation, and careful steps",
                "dominant_behaviors": ["measured walking"],
                "body_parts": {
                    "arms": "The arms swing with low amplitude.",
                    "trunk": "The trunk remains compact.",
                    "head": "The head stays steady.",
                    "legs": "",
                },
                "clip_notes": "",
            }
        )

        with self.assertRaises(DescriptionValidationError):
            parse_model_response(raw_response)

    def test_sanitize_kimodo_prompt_text_removes_artifacts_and_temporal_tail(self) -> None:
        sanitized, warnings = sanitize_kimodo_prompt_text(
            "A person sits on a chair, raises one arm sharply, then turns toward the camera",
            body_parts={
                "arms": "The arms lift sharply away from the torso.",
                "trunk": "The trunk stays compact and steady.",
                "head": "The head remains forward and controlled.",
                "legs": "The legs stay quiet and grounded.",
            },
        )

        self.assertTrue(sanitized.startswith("A person"))
        self.assertNotIn("chair", sanitized.lower())
        self.assertNotIn("camera", sanitized.lower())
        self.assertNotIn("then", sanitized.lower())
        self.assertGreaterEqual(len(sanitized.split()), 12)
        self.assertLessEqual(len(sanitized.split()), 22)
        self.assertIn("prompt_text_temporal_tail_removed", warnings)

    def test_describe_videos_writes_manifest_and_kimodo_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "RobotEmotions"
            output_dir = Path(tmp_dir) / "output"
            _build_dataset_tree(dataset_root)
            backend = _FakeBackend(
                json.dumps(
                    {
                        "prompt_text": "A person sits with tense posture, sharp arm gestures, slight forward trunk lean, alert head orientation, and grounded leg support",
                        "dominant_behaviors": ["tense sitting", "sharp gestures"],
                        "body_parts": {
                            "arms": "Arms move in short sharp gestures.",
                            "trunk": "The trunk leans slightly forward.",
                            "head": "The head stays alert and oriented ahead.",
                            "legs": "The legs stay grounded with stable support.",
                        },
                        "clip_notes": "Overall movement is focused and controlled.",
                    }
                )
            )

            summary = describe_videos(
                dataset_root=dataset_root,
                output_dir=output_dir,
                domains=("30ms",),
                backend=backend,
            )

            self.assertEqual(summary["num_total"], 1)
            self.assertEqual(summary["num_warning"], 1)
            self.assertEqual(summary["num_fail"], 0)
            manifest_entries = [
                json.loads(line)
                for line in Path(summary["video_description_manifest_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            catalog_entries = [
                json.loads(line)
                for line in Path(summary["kimodo_prompt_catalog_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(manifest_entries[0]["clip_id"], "robot_emotions_30ms_u06_tag07_2")
            self.assertEqual(manifest_entries[0]["status"], "warning")
            self.assertEqual(catalog_entries[0]["prompt_id"], "robot_emotions_30ms_u06_tag07_2")
            self.assertEqual(catalog_entries[0]["labels"]["stimulus"], "visual_methods")
            self.assertIn("robot_emotions_30ms_u06_tag07_2", backend.calls[0]["user_prompt"])

    def test_cli_dispatches_describe_videos(self) -> None:
        with patch("robot_emotions_vlm.cli.describe_videos", return_value={"status": "ok"}) as mocked_runner:
            with patch("builtins.print") as mocked_print:
                exit_code = robot_emotions_vlm_main(
                    [
                        "describe-videos",
                        "--dataset-root",
                        "/tmp/RobotEmotions",
                        "--output-dir",
                        "/tmp/output",
                    ]
                )

        self.assertEqual(exit_code, 0)
        mocked_runner.assert_called_once()
        mocked_print.assert_called_once()

    def test_load_catalog_entries_reads_reference_clip_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            catalog_path = Path(tmp_dir) / "catalog.jsonl"
            catalog_path.write_text(
                json.dumps(
                    {
                        "prompt_id": "prompt_a",
                        "prompt_text": "A person walks with calm posture, measured arm motion, upright trunk alignment, attentive head orientation, and stable leg support",
                        "labels": {"emotion": "neutrality"},
                        "seed": 7,
                        "num_samples": 2,
                        "reference_clip_id": "robot_emotions_10ms_u02_tag11",
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

            entries = load_catalog_entries(catalog_path)

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].clip_id, "robot_emotions_10ms_u02_tag11")
            self.assertEqual(entries[0].num_samples, 2)

    def test_generate_kimodo_from_catalog_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            catalog_path = Path(tmp_dir) / "catalog.jsonl"
            output_dir = Path(tmp_dir) / "kimodo_output"
            catalog_path.write_text(
                json.dumps(
                    {
                        "prompt_id": "robot_emotions_10ms_u02_tag11",
                        "prompt_text": "A person stands with open posture, expressive arm motion, upright trunk alignment, lively head orientation, and stable leg support",
                        "labels": {
                            "emotion": "happiness",
                            "modality": "standing",
                            "stimulus": "autobiographical_recall",
                        },
                        "seed": 123,
                        "num_samples": 1,
                        "reference_clip_id": "robot_emotions_10ms_u02_tag11",
                        "source_metadata": {"dataset": "RobotEmotions"},
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            runtime = _FakeKimodoRuntime()

            summary = generate_kimodo_from_catalog(
                catalog_path=catalog_path,
                output_dir=output_dir,
                runtime=runtime,
                duration_sec=6.0,
            )

            self.assertEqual(summary["num_ok"], 1)
            self.assertEqual(summary["num_fail"], 0)
            self.assertEqual(runtime.loaded_model_name, "Kimodo-SMPLX-RP-v1")
            self.assertEqual(runtime.model.calls[0]["num_frames"], 120)
            manifest_entries = [
                json.loads(line)
                for line in Path(summary["manifest_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(manifest_entries[0]["clip_id"], "robot_emotions_10ms_u02_tag11")
            self.assertEqual(manifest_entries[0]["status"], "ok")
            self.assertEqual(manifest_entries[0]["sample_id"], "robot_emotions_10ms_u02_tag11__s000")
            self.assertIn("kimodo_npz_path", manifest_entries[0]["artifacts"])

    def test_generate_kimodo_from_catalog_loads_constraints_and_filters_by_prompt_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            catalog_path = Path(tmp_dir) / "catalog.jsonl"
            output_dir = Path(tmp_dir) / "kimodo_output"
            constraints_path = Path(tmp_dir) / "constraints.json"
            constraints_path.write_text(
                json.dumps(
                    [
                        {
                            "type": "fullbody",
                            "frame_indices": [0, 2],
                            "local_joints_rot": [[[0.0, 0.0, 0.0]] * len(IMUGPT_22_JOINT_NAMES)] * 2,
                            "root_positions": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
                        },
                        {
                            "type": "root2d",
                            "frame_indices": [0, 1, 2],
                            "smooth_root_2d": [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]],
                        }
                    ],
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )
            catalog_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "prompt_id": "robot_emotions_10ms_u02_tag11__w000",
                                "window_id": "robot_emotions_10ms_u02_tag11__w000",
                                "prompt_text": "A person stands with open posture, expressive arm motion, upright trunk alignment, lively head orientation, and stable leg support",
                                "labels": {"emotion": "happiness"},
                                "seed": 123,
                                "num_samples": 1,
                                "reference_clip_id": "robot_emotions_10ms_u02_tag11",
                                "duration_hint_sec": 5.0,
                                "constraints_path": str(constraints_path.resolve()),
                                "constraint_summary": {
                                    "constraint_types": ["fullbody", "root2d"],
                                    "constraint_frame_counts": {"fullbody": 2, "root2d": 3},
                                },
                            },
                            ensure_ascii=True,
                        ),
                        json.dumps(
                            {
                                "prompt_id": "robot_emotions_10ms_u02_tag11__w001",
                                "window_id": "robot_emotions_10ms_u02_tag11__w001",
                                "prompt_text": "A person stands with compact posture, gentle arm motion, steady trunk alignment, calm head orientation, and stable leg support",
                                "labels": {"emotion": "neutrality"},
                                "seed": 9,
                                "num_samples": 1,
                                "reference_clip_id": "robot_emotions_10ms_u02_tag11",
                            },
                            ensure_ascii=True,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            runtime = _FakeKimodoRuntime()

            summary = generate_kimodo_from_catalog(
                catalog_path=catalog_path,
                output_dir=output_dir,
                runtime=runtime,
                prompt_ids=["robot_emotions_10ms_u02_tag11__w000"],
            )

            self.assertEqual(summary["num_ok"], 1)
            self.assertEqual(len(runtime.constraint_load_calls), 1)
            self.assertEqual(runtime.constraint_load_calls[0]["path"], str(constraints_path.resolve()))
            self.assertEqual(
                [constraint["type"] for constraint in runtime.model.calls[0]["constraint_lst"]],
                ["fullbody", "root2d"],
            )
            manifest_entries = [
                json.loads(line)
                for line in Path(summary["manifest_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(manifest_entries), 1)
            self.assertEqual(manifest_entries[0]["prompt_id"], "robot_emotions_10ms_u02_tag11__w000")
            self.assertEqual(manifest_entries[0]["constraints_path"], str(constraints_path.resolve()))
            self.assertEqual(manifest_entries[0]["loaded_constraint_types"], ["fullbody", "root2d"])
            self.assertEqual(manifest_entries[0]["num_constraints_loaded"], 2)

    def test_describe_windows_writes_window_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_video_path = Path(tmp_dir) / "source.mp4"
            source_video_path.write_bytes(b"fake_source")
            pose_path = Path(tmp_dir) / "pose3d.npz"
            sequence = _make_pose_sequence("robot_emotions_10ms_u02_tag11")
            write_pose_sequence3d_npz(sequence, pose_path)
            pose_manifest_path = Path(tmp_dir) / "pose3d_manifest.jsonl"
            pose_manifest_path.write_text(
                json.dumps(
                    {
                        "clip_id": "robot_emotions_10ms_u02_tag11",
                        "status": "ok",
                        "domain": "10ms",
                        "user_id": 2,
                        "tag_number": 11,
                        "take_id": None,
                        "labels": {
                            "emotion": "happiness",
                            "modality": "standing",
                            "stimulus": "autobiographical_recall",
                        },
                        "source": {
                            "video_path": str(source_video_path.resolve()),
                            "source_rel_dir": "10ms/User2/Tag11",
                        },
                        "video": {"fps": 20.0},
                        "artifacts": {"pose3d_npz_path": str(pose_path.resolve())},
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            backend = _FakeBackend(
                json.dumps(
                    {
                        "prompt_text": "A person stands with open posture, expressive arm motion, upright trunk alignment, lively head orientation, and stable leg support",
                        "dominant_behaviors": ["expressive standing"],
                        "body_parts": {
                            "arms": "Arms move with expressive elevation.",
                            "trunk": "The trunk stays upright and open.",
                            "head": "The head remains lively and oriented forward.",
                            "legs": "The legs provide stable support.",
                        },
                        "clip_notes": "Window-level motion stays energetic and open.",
                    }
                )
            )

            def _fake_write_subclip(*, source_video_path, output_path, start_sec, end_sec):
                Path(output_path).write_bytes(b"fake_window_video")

            fake_video_metadata = {
                "available": True,
                "video_path": str(source_video_path.resolve()),
                "fps": 20.0,
                "num_frames": 4,
                "duration_sec": 0.2,
                "width": 320,
                "height": 240,
            }
            with patch(
                "robot_emotions_vlm.window_descriptions._write_video_window_subclip",
                side_effect=_fake_write_subclip,
            ), patch(
                "robot_emotions_vlm.window_descriptions.read_video_metadata",
                return_value=fake_video_metadata,
            ):
                summary = describe_windows(
                    pose3d_manifest_path=pose_manifest_path,
                    output_dir=Path(tmp_dir) / "qwen_windows",
                    window_sec=0.2,
                    window_hop_sec=0.2,
                    max_windows_per_clip=1,
                    backend=backend,
                )

            self.assertEqual(summary["num_ok"], 1)
            catalog_entries = [
                json.loads(line)
                for line in Path(summary["kimodo_window_prompt_catalog_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(catalog_entries[0]["prompt_id"], "robot_emotions_10ms_u02_tag11__w000")
            self.assertEqual(catalog_entries[0]["reference_clip_id"], "robot_emotions_10ms_u02_tag11")
            self.assertEqual(catalog_entries[0]["window"]["prompt_id"], "robot_emotions_10ms_u02_tag11__w000")
            self.assertEqual(catalog_entries[0]["duration_hint_sec"], 0.2)

    def test_describe_windows_keeps_cosmetic_prompt_warnings_as_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_video_path = Path(tmp_dir) / "source.mp4"
            source_video_path.write_bytes(b"fake_source")
            pose_path = Path(tmp_dir) / "pose3d.npz"
            sequence = _make_pose_sequence("robot_emotions_10ms_u02_tag11")
            write_pose_sequence3d_npz(sequence, pose_path)
            pose_manifest_path = Path(tmp_dir) / "pose3d_manifest.jsonl"
            pose_manifest_path.write_text(
                json.dumps(
                    {
                        "clip_id": "robot_emotions_10ms_u02_tag11",
                        "status": "ok",
                        "domain": "10ms",
                        "user_id": 2,
                        "tag_number": 11,
                        "take_id": None,
                        "labels": {"emotion": "neutrality"},
                        "source": {
                            "video_path": str(source_video_path.resolve()),
                            "source_rel_dir": "10ms/User2/Tag11",
                        },
                        "video": {"fps": 20.0},
                        "artifacts": {"pose3d_npz_path": str(pose_path.resolve())},
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            backend = _FakeBackend(
                json.dumps(
                    {
                        "prompt_text": "A person stands with compact posture, measured arm motion, steady trunk alignment, attentive head orientation, and stable leg support.",
                        "dominant_behaviors": ["measured standing"],
                        "body_parts": {
                            "arms": "Arms move with measured motion.",
                            "trunk": "The trunk stays steady.",
                            "head": "The head remains attentive.",
                            "legs": "The legs provide stable support.",
                        },
                        "clip_notes": "",
                    }
                )
            )

            def _fake_write_subclip(*, source_video_path, output_path, start_sec, end_sec):
                Path(output_path).write_bytes(b"fake_window_video")

            fake_video_metadata = {
                "available": True,
                "video_path": str(source_video_path.resolve()),
                "fps": 20.0,
                "num_frames": 4,
                "duration_sec": 0.2,
                "width": 320,
                "height": 240,
            }
            with patch(
                "robot_emotions_vlm.window_descriptions._write_video_window_subclip",
                side_effect=_fake_write_subclip,
            ), patch(
                "robot_emotions_vlm.window_descriptions.read_video_metadata",
                return_value=fake_video_metadata,
            ):
                summary = describe_windows(
                    pose3d_manifest_path=pose_manifest_path,
                    output_dir=Path(tmp_dir) / "qwen_windows",
                    window_sec=0.2,
                    window_hop_sec=0.2,
                    max_windows_per_clip=1,
                    backend=backend,
                )

            self.assertEqual(summary["num_ok"], 1)
            self.assertEqual(summary["num_warning"], 0)
            quality_report = json.loads(
                (
                    Path(summary["output_dir"])
                    / "robot_emotions_10ms_u02_tag11__w000"
                    / "quality_report.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(quality_report["status"], "ok")
            self.assertIn("prompt_text_trailing_period_removed", quality_report["warnings"])

    def test_describe_windows_accepts_warning_pose_manifest_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_video_path = Path(tmp_dir) / "source.mp4"
            source_video_path.write_bytes(b"fake_source")
            pose_path = Path(tmp_dir) / "pose3d.npz"
            sequence = _make_pose_sequence("robot_emotions_10ms_u02_tag11")
            write_pose_sequence3d_npz(sequence, pose_path)
            pose_manifest_path = Path(tmp_dir) / "pose3d_manifest.jsonl"
            pose_manifest_path.write_text(
                json.dumps(
                    {
                        "clip_id": "robot_emotions_10ms_u02_tag11",
                        "status": "warning",
                        "domain": "10ms",
                        "user_id": 2,
                        "tag_number": 11,
                        "take_id": None,
                        "labels": {
                            "emotion": "happiness",
                            "modality": "standing",
                            "stimulus": "autobiographical_recall",
                        },
                        "source": {
                            "video_path": str(source_video_path.resolve()),
                            "source_rel_dir": "10ms/User2/Tag11",
                        },
                        "video": {"fps": 20.0},
                        "artifacts": {"pose3d_npz_path": str(pose_path.resolve())},
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            backend = _FakeBackend(
                json.dumps(
                    {
                        "prompt_text": "A person stands with open posture, expressive arm motion, upright trunk alignment, lively head orientation, and stable leg support",
                        "dominant_behaviors": ["expressive standing"],
                        "body_parts": {
                            "arms": "Arms move with expressive elevation.",
                            "trunk": "The trunk stays upright and open.",
                            "head": "The head remains lively and oriented forward.",
                            "legs": "The legs provide stable support.",
                        },
                        "clip_notes": "Window-level motion stays energetic and open.",
                    }
                )
            )

            def _fake_write_subclip(*, source_video_path, output_path, start_sec, end_sec):
                Path(output_path).write_bytes(b"fake_window_video")

            fake_video_metadata = {
                "available": True,
                "video_path": str(source_video_path.resolve()),
                "fps": 20.0,
                "num_frames": 4,
                "duration_sec": 0.2,
                "width": 320,
                "height": 240,
            }
            with patch(
                "robot_emotions_vlm.window_descriptions._write_video_window_subclip",
                side_effect=_fake_write_subclip,
            ), patch(
                "robot_emotions_vlm.window_descriptions.read_video_metadata",
                return_value=fake_video_metadata,
            ):
                summary = describe_windows(
                    pose3d_manifest_path=pose_manifest_path,
                    output_dir=Path(tmp_dir) / "qwen_windows",
                    window_sec=0.2,
                    window_hop_sec=0.2,
                    max_windows_per_clip=1,
                    backend=backend,
                )

            self.assertEqual(summary["num_total_windows"], 1)
            self.assertEqual(summary["num_ok"], 1)

    def test_write_video_window_subclip_passes_rational_rate_to_av(self) -> None:
        captured: dict[str, object] = {}
        encoded_frames: list[object] = []

        class _FakeFrame:
            def __init__(self, time_value: float) -> None:
                self.time = time_value
                self.pts = None
                self.time_base = None

            def reformat(self, **kwargs):
                return self

        class _FakeInputContainer:
            def __init__(self) -> None:
                self.streams = [
                    type(
                        "Stream",
                        (),
                        {
                            "type": "video",
                            "average_rate": 29.97,
                            "codec_context": type("CodecContext", (), {"width": 320, "height": 240})(),
                        },
                    )()
                ]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def decode(self, stream):
                return [_FakeFrame(0.0), _FakeFrame(0.02), _FakeFrame(0.04)]

        class _FakeOutputStream:
            def __init__(self) -> None:
                self.width = None
                self.height = None
                self.pix_fmt = None

            def encode(self, frame):
                if frame is not None:
                    encoded_frames.append(frame)
                return []

        class _FakeOutputContainer:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def add_stream(self, codec: str, *, rate):
                captured["codec"] = codec
                captured["rate"] = rate
                if isinstance(rate, float):
                    raise AssertionError("rate must not be passed to PyAV as float")
                return _FakeOutputStream()

            def mux(self, packet) -> None:
                return None

        def _fake_av_open(path: str, mode: str = "r"):
            if mode == "w":
                return _FakeOutputContainer()
            return _FakeInputContainer()

        fake_av_module = types.SimpleNamespace(open=_fake_av_open)
        with patch.dict(sys.modules, {"av": fake_av_module}):
            _write_video_window_subclip(
                source_video_path="/tmp/source.mp4",
                output_path="/tmp/window.mp4",
                start_sec=0.0,
                end_sec=0.03,
            )

        self.assertEqual(captured["codec"], "libx264")
        self.assertIsInstance(captured["rate"], Fraction)
        self.assertEqual([frame.pts for frame in encoded_frames], [0, 1])
        expected_time_base = Fraction(captured["rate"].denominator, captured["rate"].numerator)
        self.assertTrue(all(frame.time_base == expected_time_base for frame in encoded_frames))

    def test_build_anchor_catalog_writes_window_catalog_and_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pose_path = Path(tmp_dir) / "pose3d.npz"
            sequence = _make_pose_sequence("robot_emotions_10ms_u02_tag11")
            write_pose_sequence3d_npz(sequence, pose_path)
            ik_path = Path(tmp_dir) / "ik_sequence.npz"
            _write_ik_sequence_npz(_make_ik_sequence(sequence), ik_path)
            pose_manifest_path = Path(tmp_dir) / "pose3d_manifest.jsonl"
            _write_anchor_pose_manifest(
                pose_manifest_path,
                clip_id="robot_emotions_10ms_u02_tag11",
                pose_path=pose_path,
            )
            qwen_catalog_path = Path(tmp_dir) / "kimodo_window_prompt_catalog.jsonl"
            window_payload = {
                "window_index": 0,
                "prompt_id": "robot_emotions_10ms_u02_tag11__w000",
                "start_sec": 0.0,
                "end_sec": 0.4,
                "duration_sec": 0.4,
                "source_start_index": 0,
                "source_end_index": 8,
            }
            _write_anchor_qwen_catalog(
                qwen_catalog_path,
                prompt_id="robot_emotions_10ms_u02_tag11__w000",
                reference_clip_id="robot_emotions_10ms_u02_tag11",
                window_payload=window_payload,
            )

            summary = build_anchor_catalog(
                pose3d_manifest_path=pose_manifest_path,
                qwen_window_catalog_path=qwen_catalog_path,
                output_dir=Path(tmp_dir) / "anchors",
                constraint_keyframes=5,
                runtime=_FakeAnchorRuntime(fps=20.0),
            )

            self.assertEqual(summary["num_ok"], 1)
            catalog_entries = [
                json.loads(line)
                for line in Path(summary["catalog_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(catalog_entries[0]["prompt_id"], "robot_emotions_10ms_u02_tag11__w000")
            self.assertEqual(catalog_entries[0]["window_id"], "robot_emotions_10ms_u02_tag11__w000")
            self.assertEqual(catalog_entries[0]["num_samples"], 4)
            self.assertIn("constraints_path", catalog_entries[0])
            self.assertEqual(catalog_entries[0]["window"]["prompt_id"], "robot_emotions_10ms_u02_tag11__w000")
            constraints_payload = json.loads(
                Path(catalog_entries[0]["constraints_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual([constraint["type"] for constraint in constraints_payload], ["end-effector", "root2d"])
            self.assertEqual(constraints_payload[0]["frame_indices"], [0, 2, 4, 5, 7])
            self.assertEqual(len(constraints_payload[0]["root_positions"]), 5)
            self.assertEqual(
                constraints_payload[0]["joint_names"],
                ["LeftFoot", "RightFoot", "LeftHand", "RightHand", "Hips"],
            )
            self.assertEqual(len(constraints_payload[1]["frame_indices"]), 8)
            self.assertIn("global_root_heading", constraints_payload[1])
            self.assertEqual(
                catalog_entries[0]["constraint_summary"]["constraint_types"],
                ["end-effector", "root2d"],
            )
            self.assertEqual(
                catalog_entries[0]["constraint_summary"]["constraint_frame_counts"],
                {"end-effector": 5, "root2d": 8},
            )
            self.assertTrue(catalog_entries[0]["constraint_summary"]["root2d_enabled"])
            self.assertTrue(catalog_entries[0]["constraint_summary"]["heading_enabled"])
            self.assertEqual(catalog_entries[0]["constraint_summary"]["ik_sequence_path"], str(ik_path.resolve()))

    def test_build_anchor_catalog_grounds_fullbody_root_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pose_path = Path(tmp_dir) / "pose3d.npz"
            sequence = _make_pose_sequence("robot_emotions_10ms_u02_tag11")
            sequence.joint_positions_xyz[:, [7, 8, 10, 11], 1] = np.asarray(
                [-0.5, -0.45, -0.55, -0.52],
                dtype=np.float32,
            )
            write_pose_sequence3d_npz(sequence, pose_path)
            _write_ik_sequence_npz(_make_ik_sequence(sequence), Path(tmp_dir) / "ik_sequence.npz")
            pose_manifest_path = Path(tmp_dir) / "pose3d_manifest.jsonl"
            _write_anchor_pose_manifest(
                pose_manifest_path,
                clip_id="robot_emotions_10ms_u02_tag11",
                pose_path=pose_path,
            )
            qwen_catalog_path = Path(tmp_dir) / "kimodo_window_prompt_catalog.jsonl"
            _write_anchor_qwen_catalog(
                qwen_catalog_path,
                prompt_id="robot_emotions_10ms_u02_tag11__w000",
                reference_clip_id="robot_emotions_10ms_u02_tag11",
                window_payload={
                    "window_index": 0,
                    "prompt_id": "robot_emotions_10ms_u02_tag11__w000",
                    "start_sec": 0.0,
                    "end_sec": 0.4,
                    "duration_sec": 0.4,
                    "source_start_index": 0,
                    "source_end_index": 8,
                },
            )

            summary = build_anchor_catalog(
                pose3d_manifest_path=pose_manifest_path,
                qwen_window_catalog_path=qwen_catalog_path,
                output_dir=Path(tmp_dir) / "anchors",
                constraint_keyframes=4,
                runtime=_FakeAnchorRuntime(fps=20.0),
            )

            catalog_entry = json.loads(Path(summary["catalog_path"]).read_text(encoding="utf-8").strip())
            constraints_payload = json.loads(Path(catalog_entry["constraints_path"]).read_text(encoding="utf-8"))
            grounded_root_positions = np.asarray(constraints_payload[0]["root_positions"], dtype=np.float32)
            self.assertTrue(np.allclose(grounded_root_positions[:, 1], 0.55, atol=1e-6))
            self.assertTrue(np.all(grounded_root_positions[:, 1] > 0.0))

    def test_build_anchor_catalog_retargets_fullbody_rotations_to_smplx_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pose_path = Path(tmp_dir) / "pose3d.npz"
            sequence = _make_pose_sequence("robot_emotions_10ms_u02_tag11", x_step=0.0, z_step=0.0)
            joint_index = {name: idx for idx, name in enumerate(IMUGPT_22_JOINT_NAMES)}
            reference_offsets = {
                "Pelvis": [0.0, 0.0, 0.0],
                "Left_hip": [0.09, -0.03, 0.0],
                "Right_hip": [-0.09, -0.03, 0.0],
                "Spine1": [0.0, 0.18, 0.0],
                "Left_knee": [0.09, -0.42, 0.02],
                "Right_knee": [-0.09, -0.42, 0.02],
                "Spine2": [0.0, 0.34, 0.0],
                "Left_ankle": [0.09, -0.78, 0.04],
                "Right_ankle": [-0.09, -0.78, 0.04],
                "Spine3": [0.0, 0.52, 0.0],
                "Left_foot": [0.09, -0.84, 0.16],
                "Right_foot": [-0.09, -0.84, 0.16],
                "Neck": [0.0, 0.68, 0.0],
                "Left_collar": [0.08, 0.56, 0.0],
                "Right_collar": [-0.08, 0.56, 0.0],
                "Head": [0.0, 0.84, 0.02],
                "Left_shoulder": [0.12, 0.49, 0.01],
                "Right_shoulder": [-0.12, 0.49, 0.01],
                "Left_elbow": [0.11, 0.27, 0.03],
                "Right_elbow": [-0.11, 0.27, 0.03],
                "Left_wrist": [0.08, 0.08, 0.05],
                "Right_wrist": [-0.08, 0.08, 0.05],
            }
            for frame_index in range(sequence.num_frames):
                root = np.asarray(sequence.root_translation_m[frame_index], dtype=np.float32)
                for joint_name, offset in reference_offsets.items():
                    sequence.joint_positions_xyz[frame_index, joint_index[joint_name]] = (
                        root + np.asarray(offset, dtype=np.float32)
                    )
            write_pose_sequence3d_npz(sequence, pose_path)
            _write_ik_sequence_npz(_make_ik_sequence(sequence), Path(tmp_dir) / "ik_sequence.npz")
            pose_manifest_path = Path(tmp_dir) / "pose3d_manifest.jsonl"
            _write_anchor_pose_manifest(
                pose_manifest_path,
                clip_id="robot_emotions_10ms_u02_tag11",
                pose_path=pose_path,
            )
            qwen_catalog_path = Path(tmp_dir) / "kimodo_window_prompt_catalog.jsonl"
            _write_anchor_qwen_catalog(
                qwen_catalog_path,
                prompt_id="robot_emotions_10ms_u02_tag11__w000",
                reference_clip_id="robot_emotions_10ms_u02_tag11",
                window_payload={
                    "window_index": 0,
                    "prompt_id": "robot_emotions_10ms_u02_tag11__w000",
                    "start_sec": 0.0,
                    "end_sec": 0.4,
                    "duration_sec": 0.4,
                    "source_start_index": 0,
                    "source_end_index": 8,
                },
            )

            summary = build_anchor_catalog(
                pose3d_manifest_path=pose_manifest_path,
                qwen_window_catalog_path=qwen_catalog_path,
                output_dir=Path(tmp_dir) / "anchors",
                constraint_keyframes=4,
                runtime=_FakeAnchorRuntime(fps=20.0),
            )

            catalog_entry = json.loads(Path(summary["catalog_path"]).read_text(encoding="utf-8").strip())
            constraints_payload = json.loads(Path(catalog_entry["constraints_path"]).read_text(encoding="utf-8"))
            fullbody_local_rot = np.asarray(constraints_payload[0]["local_joints_rot"], dtype=np.float32)
            arm_indices = [
                int(SMPLX22_JOINT_NAMES.index("left_shoulder")),
                int(SMPLX22_JOINT_NAMES.index("right_shoulder")),
                int(SMPLX22_JOINT_NAMES.index("left_elbow")),
                int(SMPLX22_JOINT_NAMES.index("right_elbow")),
                int(SMPLX22_JOINT_NAMES.index("left_wrist")),
                int(SMPLX22_JOINT_NAMES.index("right_wrist")),
            ]
            arm_rotation_norms = np.linalg.norm(fullbody_local_rot[:, arm_indices, :], axis=-1)
            self.assertGreater(float(np.max(arm_rotation_norms)), 0.05)

    def test_build_anchor_catalog_skips_root2d_for_small_displacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pose_path = Path(tmp_dir) / "pose3d.npz"
            sequence = _make_pose_sequence("robot_emotions_10ms_u02_tag11", x_step=0.004, z_step=0.0)
            write_pose_sequence3d_npz(sequence, pose_path)
            _write_ik_sequence_npz(_make_ik_sequence(sequence), Path(tmp_dir) / "ik_sequence.npz")
            pose_manifest_path = Path(tmp_dir) / "pose3d_manifest.jsonl"
            _write_anchor_pose_manifest(
                pose_manifest_path,
                clip_id="robot_emotions_10ms_u02_tag11",
                pose_path=pose_path,
            )
            qwen_catalog_path = Path(tmp_dir) / "kimodo_window_prompt_catalog.jsonl"
            _write_anchor_qwen_catalog(
                qwen_catalog_path,
                prompt_id="robot_emotions_10ms_u02_tag11__w000",
                reference_clip_id="robot_emotions_10ms_u02_tag11",
                window_payload={
                    "window_index": 0,
                    "prompt_id": "robot_emotions_10ms_u02_tag11__w000",
                    "start_sec": 0.0,
                    "end_sec": 0.4,
                    "duration_sec": 0.4,
                    "source_start_index": 0,
                    "source_end_index": 8,
                },
            )

            summary = build_anchor_catalog(
                pose3d_manifest_path=pose_manifest_path,
                qwen_window_catalog_path=qwen_catalog_path,
                output_dir=Path(tmp_dir) / "anchors",
                constraint_keyframes=4,
                runtime=_FakeAnchorRuntime(fps=20.0),
            )

            catalog_entry = json.loads(Path(summary["catalog_path"]).read_text(encoding="utf-8").strip())
            constraints_payload = json.loads(Path(catalog_entry["constraints_path"]).read_text(encoding="utf-8"))
            self.assertEqual([constraint["type"] for constraint in constraints_payload], ["end-effector"])
            self.assertFalse(catalog_entry["constraint_summary"]["root2d_enabled"])
            self.assertFalse(catalog_entry["constraint_summary"]["heading_enabled"])

    def test_build_anchor_catalog_emits_root2d_without_heading_for_mid_displacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pose_path = Path(tmp_dir) / "pose3d.npz"
            sequence = _make_pose_sequence("robot_emotions_10ms_u02_tag11", x_step=0.01, z_step=0.0)
            write_pose_sequence3d_npz(sequence, pose_path)
            _write_ik_sequence_npz(_make_ik_sequence(sequence), Path(tmp_dir) / "ik_sequence.npz")
            pose_manifest_path = Path(tmp_dir) / "pose3d_manifest.jsonl"
            _write_anchor_pose_manifest(
                pose_manifest_path,
                clip_id="robot_emotions_10ms_u02_tag11",
                pose_path=pose_path,
            )
            qwen_catalog_path = Path(tmp_dir) / "kimodo_window_prompt_catalog.jsonl"
            _write_anchor_qwen_catalog(
                qwen_catalog_path,
                prompt_id="robot_emotions_10ms_u02_tag11__w000",
                reference_clip_id="robot_emotions_10ms_u02_tag11",
                window_payload={
                    "window_index": 0,
                    "prompt_id": "robot_emotions_10ms_u02_tag11__w000",
                    "start_sec": 0.0,
                    "end_sec": 0.4,
                    "duration_sec": 0.4,
                    "source_start_index": 0,
                    "source_end_index": 8,
                },
            )

            summary = build_anchor_catalog(
                pose3d_manifest_path=pose_manifest_path,
                qwen_window_catalog_path=qwen_catalog_path,
                output_dir=Path(tmp_dir) / "anchors",
                constraint_keyframes=4,
                runtime=_FakeAnchorRuntime(fps=20.0),
            )

            catalog_entry = json.loads(Path(summary["catalog_path"]).read_text(encoding="utf-8").strip())
            constraints_payload = json.loads(Path(catalog_entry["constraints_path"]).read_text(encoding="utf-8"))
            self.assertEqual([constraint["type"] for constraint in constraints_payload], ["end-effector", "root2d"])
            self.assertNotIn("global_root_heading", constraints_payload[1])
            self.assertTrue(catalog_entry["constraint_summary"]["root2d_enabled"])
            self.assertFalse(catalog_entry["constraint_summary"]["heading_enabled"])

    def test_build_anchor_catalog_requires_ik_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pose_path = Path(tmp_dir) / "pose3d.npz"
            sequence = _make_pose_sequence("robot_emotions_10ms_u02_tag11")
            write_pose_sequence3d_npz(sequence, pose_path)
            pose_manifest_path = Path(tmp_dir) / "pose3d_manifest.jsonl"
            _write_anchor_pose_manifest(
                pose_manifest_path,
                clip_id="robot_emotions_10ms_u02_tag11",
                pose_path=pose_path,
            )
            qwen_catalog_path = Path(tmp_dir) / "kimodo_window_prompt_catalog.jsonl"
            _write_anchor_qwen_catalog(
                qwen_catalog_path,
                prompt_id="robot_emotions_10ms_u02_tag11__w000",
                reference_clip_id="robot_emotions_10ms_u02_tag11",
                window_payload={
                    "window_index": 0,
                    "prompt_id": "robot_emotions_10ms_u02_tag11__w000",
                    "start_sec": 0.0,
                    "end_sec": 0.4,
                    "duration_sec": 0.4,
                    "source_start_index": 0,
                    "source_end_index": 8,
                },
            )

            with self.assertRaisesRegex(ValueError, "ik_sequence.npz"):
                build_anchor_catalog(
                    pose3d_manifest_path=pose_manifest_path,
                    qwen_window_catalog_path=qwen_catalog_path,
                    output_dir=Path(tmp_dir) / "anchors",
                    runtime=_FakeAnchorRuntime(fps=20.0),
                )

    def test_save_smplx_amass_outputs_writes_next_to_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            _FakeAMASSConverter.saved_paths = []
            npz_path = Path(tmp_dir) / "clip_a" / "motion_00.npz"
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            npz_path.write_bytes(b"fake_npz")

            amass_path = _save_smplx_amass_outputs(
                output={"posed_joints": np.zeros((10, 3, 3), dtype=np.float32)},
                output_path=npz_path,
                skeleton=object(),
                fps=20.0,
                converter_factory=_FakeAMASSConverter,
            )

            self.assertEqual(Path(amass_path), npz_path.with_name("motion_00_amass.npz"))
            self.assertTrue(Path(amass_path).exists())
            self.assertEqual(_FakeAMASSConverter.saved_paths, [str(npz_path.with_name("motion_00_amass.npz"))])

    def test_cli_dispatches_generate_kimodo(self) -> None:
        with patch("robot_emotions_vlm.cli.generate_kimodo_from_catalog", return_value={"status": "ok"}) as mocked_runner:
            with patch("builtins.print") as mocked_print:
                exit_code = robot_emotions_vlm_main(
                    [
                        "generate-kimodo",
                        "--catalog-path",
                        "/tmp/catalog.jsonl",
                        "--output-dir",
                        "/tmp/kimodo_output",
                    ]
                )

        self.assertEqual(exit_code, 0)
        mocked_runner.assert_called_once()
        mocked_print.assert_called_once()

    def test_cli_dispatches_describe_windows(self) -> None:
        with patch("robot_emotions_vlm.cli.describe_windows", return_value={"status": "ok"}) as mocked_runner:
            with patch("builtins.print") as mocked_print:
                exit_code = robot_emotions_vlm_main(
                    [
                        "describe-windows",
                        "--pose3d-manifest-path",
                        "/tmp/pose3d_manifest.jsonl",
                        "--output-dir",
                        "/tmp/qwen_windows",
                    ]
                )

        self.assertEqual(exit_code, 0)
        mocked_runner.assert_called_once()
        self.assertEqual(mocked_runner.call_args.kwargs["num_video_frames"], 48)
        mocked_print.assert_called_once()

    def test_cli_dispatches_build_anchor_catalog(self) -> None:
        with patch("robot_emotions_vlm.cli.build_anchor_catalog", return_value={"status": "ok"}) as mocked_runner:
            with patch("builtins.print") as mocked_print:
                exit_code = robot_emotions_vlm_main(
                    [
                        "build-anchor-catalog",
                        "--pose3d-manifest-path",
                        "/tmp/pose3d_manifest.jsonl",
                        "--qwen-window-catalog-path",
                        "/tmp/kimodo_window_prompt_catalog.jsonl",
                        "--output-dir",
                        "/tmp/anchors",
                        "--constraint-keyframes",
                        "6",
                    ]
                )

        self.assertEqual(exit_code, 0)
        mocked_runner.assert_called_once()
        self.assertEqual(mocked_runner.call_args.kwargs["constraint_keyframes"], 6)
        mocked_print.assert_called_once()

    def test_qwen_backend_passes_fps_none_with_num_frames(self) -> None:
        backend = QwenVideoBackend(QwenGenerationConfig(num_video_frames=8))
        fake_processor = _FakeProcessor()
        backend._processor = fake_processor
        backend._model = _FakeModel()

        raw_response = backend.describe_video(
            "data/RobotEmotions/10ms/User2/Tag11/TAG_11.mp4",
            system_prompt="system",
            user_prompt="user",
        )

        self.assertIn('"prompt_text"', raw_response)
        self.assertEqual(fake_processor.apply_calls[0]["num_frames"], 8)
        self.assertIsNone(fake_processor.apply_calls[0]["fps"])


if __name__ == "__main__":
    unittest.main()
