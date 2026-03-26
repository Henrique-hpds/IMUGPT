import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pose_module.interfaces import COCO_17_JOINT_NAMES, MOTIONBERT_17_JOINT_NAMES, PoseSequence2D
from pose_module.pipeline import run_pose2d_pipeline
from pose_module.processing.cleaner2d import clean_pose_sequence2d
from pose_module.tracking.person_selector import TrackState


_COCO_BASE_POINTS = {
    "nose": (50.0, 20.0),
    "left_eye": (47.0, 18.0),
    "right_eye": (53.0, 18.0),
    "left_ear": (44.0, 20.0),
    "right_ear": (56.0, 20.0),
    "left_shoulder": (40.0, 42.0),
    "right_shoulder": (60.0, 42.0),
    "left_elbow": (34.0, 58.0),
    "right_elbow": (66.0, 58.0),
    "left_wrist": (30.0, 76.0),
    "right_wrist": (70.0, 76.0),
    "left_hip": (45.0, 82.0),
    "right_hip": (55.0, 82.0),
    "left_knee": (45.0, 114.0),
    "right_knee": (55.0, 114.0),
    "left_ankle": (45.0, 144.0),
    "right_ankle": (55.0, 144.0),
}


def _make_pose_sequence(
    num_frames: int,
    *,
    missing: dict[int, tuple[str, ...]] | None = None,
    invalid_bbox_frames: tuple[int, ...] = (),
) -> PoseSequence2D:
    missing = {} if missing is None else dict(missing)
    keypoints = np.full((num_frames, len(COCO_17_JOINT_NAMES), 2), np.nan, dtype=np.float32)
    confidence = np.full((num_frames, len(COCO_17_JOINT_NAMES)), 0.9, dtype=np.float32)
    bbox_xywh = np.full((num_frames, 4), np.nan, dtype=np.float32)

    for frame_index in range(num_frames):
        x_shift = float(frame_index * 2.0)
        for joint_index, joint_name in enumerate(COCO_17_JOINT_NAMES):
            base_x, base_y = _COCO_BASE_POINTS[joint_name]
            keypoints[frame_index, joint_index] = np.asarray([base_x + x_shift, base_y], dtype=np.float32)
        bbox_xywh[frame_index] = np.asarray([20.0 + x_shift, 10.0, 60.0, 150.0], dtype=np.float32)

        for joint_name in missing.get(frame_index, ()):
            joint_index = COCO_17_JOINT_NAMES.index(joint_name)
            keypoints[frame_index, joint_index] = np.asarray([np.nan, np.nan], dtype=np.float32)
            confidence[frame_index, joint_index] = 0.0

        if frame_index in invalid_bbox_frames:
            bbox_xywh[frame_index] = np.asarray([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)

    return PoseSequence2D(
        clip_id="clip_test",
        fps=20.0,
        fps_original=30.0,
        joint_names_2d=list(COCO_17_JOINT_NAMES),
        keypoints_xy=keypoints,
        confidence=confidence,
        bbox_xywh=bbox_xywh,
        frame_indices=np.arange(num_frames, dtype=np.int32),
        timestamps_sec=np.arange(num_frames, dtype=np.float32) / 20.0,
        source="vitpose-b",
    )


class Pose2DCleanerTests(unittest.TestCase):
    def test_clean_pose_sequence2d_interpolates_short_gap_and_maps_to_motionbert17(self) -> None:
        raw_sequence = _make_pose_sequence(
            7,
            missing={
                2: ("left_wrist",),
                3: ("left_wrist",),
            },
        )

        cleaned_sequence, quality_report, _ = clean_pose_sequence2d(
            raw_sequence,
            track_report={"status": "ok", "warnings": []},
        )

        left_wrist_index = MOTIONBERT_17_JOINT_NAMES.index("left_wrist")
        self.assertEqual(cleaned_sequence.joint_names_2d, list(MOTIONBERT_17_JOINT_NAMES))
        self.assertTrue(np.isfinite(cleaned_sequence.keypoints_xy[2:4, left_wrist_index]).all())
        self.assertGreaterEqual(quality_report["frames_interpolated"], 2)
        self.assertEqual(quality_report["status"], "ok")

    def test_clean_pose_sequence2d_keeps_interpolated_lower_leg_joints_low_confidence_and_imputed(self) -> None:
        raw_sequence = _make_pose_sequence(
            7,
            missing={
                2: ("left_ankle",),
                3: ("left_ankle",),
            },
        )

        cleaned_sequence, _, artifacts = clean_pose_sequence2d(
            raw_sequence,
            track_report={"status": "ok", "warnings": []},
        )

        left_ankle_index = MOTIONBERT_17_JOINT_NAMES.index("left_ankle")
        self.assertTrue(np.isfinite(cleaned_sequence.keypoints_xy[2:4, left_ankle_index]).all())
        self.assertTrue(np.all(cleaned_sequence.confidence[2:4, left_ankle_index] <= 0.15))
        self.assertTrue(np.all(cleaned_sequence.confidence[2:4, left_ankle_index] > 0.0))
        self.assertTrue(np.all(cleaned_sequence.imputed_mask[2:4, left_ankle_index]))
        self.assertTrue(np.all(~cleaned_sequence.observed_mask[2:4, left_ankle_index]))
        self.assertTrue(np.all(artifacts["imputed_mask"][2:4, left_ankle_index]))
        self.assertTrue(np.all(~artifacts["observed_mask"][2:4, left_ankle_index]))

    def test_clean_pose_sequence2d_fails_on_excess_missing_joints(self) -> None:
        raw_sequence = _make_pose_sequence(
            10,
            missing={
                0: ("left_hip", "right_hip", "left_knee", "right_knee"),
                1: ("left_hip", "right_hip", "left_knee", "right_knee"),
                2: ("left_hip", "right_hip", "left_knee", "right_knee"),
                3: ("left_hip", "right_hip", "left_knee", "right_knee"),
            },
        )

        _, quality_report, _ = clean_pose_sequence2d(
            raw_sequence,
            track_report={"status": "ok", "warnings": []},
        )

        self.assertEqual(quality_report["status"], "fail")
        self.assertIn("too_many_frames_with_excess_missing_joints", quality_report["notes"])

    def test_run_pose2d_pipeline_exports_cleaner_artifacts(self) -> None:
        raw_sequence = _make_pose_sequence(
            6,
            missing={
                2: ("right_wrist",),
                3: ("right_wrist",),
            },
        )
        pose_quality = {
            "status": "ok",
            "fps": 20.0,
            "fps_original": 30.0,
            "num_selected_frames": 6,
            "frames_with_selected_track": 6,
            "visible_joint_ratio": 0.98,
            "mean_confidence": 0.9,
            "notes": [],
        }
        backend_quality = {
            "fps_original": 30.0,
            "effective_fps": 20.0,
            "frames_total": 6,
            "frames_selected": 6,
            "warnings": [],
        }
        track_report = {
            "status": "ok",
            "selected_track_id": 1,
            "selected_track_stability": 1.0,
            "frames_with_detections": 6,
            "warnings": [],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_prediction_path = Path(tmp_dir) / "raw_predictions.json"
            raw_prediction_path.write_text("[]", encoding="utf-8")
            rendered_outputs = []

            backend_run = {
                "status": "ok",
                "artifacts": {
                    "raw_prediction_json_path": str(raw_prediction_path),
                    "debug_overlay_path": None,
                },
                "selected_frame_indices": list(range(6)),
                "effective_fps": 20.0,
                "quality_report": backend_quality,
            }

            def _fake_render_pose_overlay_video(*, output_path, **kwargs):
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"fake mp4")
                rendered_outputs.append(output_path.name)
                return output_path.resolve()

            with patch("pose_module.pipeline.run_backend_job", return_value=backend_run), patch(
                "pose_module.pipeline.load_raw_prediction_frames",
                return_value=[{"frame_id": idx, "instances": []} for idx in range(6)],
            ), patch(
                "pose_module.pipeline.link_person_tracks",
                return_value=[TrackState(track_id=1)],
            ), patch(
                "pose_module.pipeline.build_person_track_report",
                return_value=track_report,
            ), patch(
                "pose_module.pipeline.canonicalize_pose_sequence2d",
                return_value=(raw_sequence, pose_quality),
            ), patch(
                "pose_module.pipeline.render_pose_overlay_video",
                side_effect=_fake_render_pose_overlay_video,
            ):
                result = run_pose2d_pipeline(
                    clip_id="clip_pipeline",
                    video_path=str(Path(tmp_dir) / "video.mp4"),
                    output_dir=Path(tmp_dir) / "pose",
                    fps_target=20,
                    save_debug=True,
                    env_name="current",
                    video_metadata={"fps": 30.0, "num_frames": 6, "duration_sec": 0.3},
                )

            self.assertTrue(Path(result["artifacts"]["pose2d_raw_keypoints_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["pose2d_clean_keypoints_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["debug_overlay_raw_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["debug_overlay_clean_path"]).exists())
            self.assertEqual(result["pose_sequence"].joint_names_2d, list(MOTIONBERT_17_JOINT_NAMES))
            self.assertIn("temporal_jitter_score", result["quality_report"])
            self.assertEqual(sorted(rendered_outputs), ["debug_overlay_clean.mp4", "debug_overlay_raw.mp4"])


if __name__ == "__main__":
    unittest.main()
