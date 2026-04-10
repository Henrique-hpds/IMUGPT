import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from robot_emotions_vlm.cli import describe_videos, main as robot_emotions_vlm_main
from robot_emotions_vlm.dataset import RobotEmotionsDataset
from robot_emotions_vlm.kimodo_generation import (
    _save_smplx_amass_outputs,
    generate_kimodo_from_catalog,
    load_catalog_entries,
)
from robot_emotions_vlm.prompts import render_prompts
from robot_emotions_vlm.qwen_backend import QwenGenerationConfig, QwenVideoBackend
from robot_emotions_vlm.schemas import DescriptionValidationError, parse_model_response


def _build_dataset_tree(root: Path) -> None:
    clip_root = root / "30ms" / "User6" / "Tag7"
    clip_root.mkdir(parents=True, exist_ok=True)
    (clip_root / "ESP_6_7_2.csv").write_text("timestamp,ax\n0,0\n", encoding="utf-8")
    (clip_root / "TAG_6_7_2.mp4").write_bytes(b"")


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
        self.skeleton = type("Skeleton", (), {"name": "somaskel77"})()
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
        self.default_model = "Kimodo-SOMA-RP-v1"
        self.model = _FakeKimodoModel()
        self.seed_calls: list[int] = []
        self.save_calls: list[dict[str, object]] = []

    def resolve_device(self) -> str:
        return "cpu"

    def load_model(self, model_name, **kwargs):
        return self.model, "kimodo-soma-rp"

    def get_model_info(self, resolved_model):
        return type("Info", (), {"display_name": "Kimodo-SOMA-RP-v1"})()

    def seed_everything(self, seed: int) -> None:
        self.seed_calls.append(seed)

    def save_outputs(self, **kwargs):
        self.save_calls.append(kwargs)
        output_stem = Path(kwargs["output_stem"])
        npz_path = output_stem.with_suffix(".npz")
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        npz_path.write_bytes(b"fake_npz")
        return {"kimodo_npz_path": str(npz_path.resolve())}


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
            self.assertEqual(runtime.model.calls[0]["num_frames"], 120)
            manifest_entries = [
                json.loads(line)
                for line in Path(summary["manifest_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(manifest_entries[0]["clip_id"], "robot_emotions_10ms_u02_tag11")
            self.assertEqual(manifest_entries[0]["status"], "ok")
            self.assertIn("kimodo_npz_path", manifest_entries[0]["artifacts"])

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
