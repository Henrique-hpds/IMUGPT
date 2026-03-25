import unittest
from pathlib import Path

import numpy as np

from pose_module.interfaces import PoseFrameInstance
from pose_module.io.video_loader import select_frame_indices
from pose_module.tracking.person_selector import (
    TrackState,
    build_person_track_report,
    link_person_tracks,
)
from pose_module.vitpose.adapter import canonicalize_pose_sequence2d
from pose_module.robot_emotions.extractor import (
    RobotEmotionsClipRecord,
    resolve_clip_output_dir,
)


def _make_record(clip_id: str) -> RobotEmotionsClipRecord:
    return RobotEmotionsClipRecord(
        clip_id=clip_id,
        domain="30ms",
        user_id=6,
        tag_number=7,
        tag_dir=Path("data/RobotEmotions/30ms/User6/Tag7"),
        imu_csv_path=Path("data/RobotEmotions/30ms/User6/Tag7/ESP_6_7_2.csv"),
        video_path=Path("data/RobotEmotions/30ms/User6/Tag7/TAG_6_7_2.mp4"),
        source_rel_dir="30ms/User6/Tag7",
        take_id="2",
        participant={"name": "Rosaria", "age": 24, "gender": "F"},
        protocol={"emotion": "Sadness"},
    )


def _pose_instance(frame_id: int, x1: float, y1: float, x2: float, y2: float) -> PoseFrameInstance:
    keypoints = np.stack(
        [
            np.linspace(x1, x2, 17, dtype=np.float32),
            np.linspace(y1, y2, 17, dtype=np.float32),
        ],
        axis=1,
    )
    return PoseFrameInstance(
        frame_id=frame_id,
        bbox_xyxy=np.asarray([x1, y1, x2, y2], dtype=np.float32),
        bbox_score=0.95,
        keypoints_xy=keypoints,
        keypoint_scores=np.full((17,), 0.9, dtype=np.float32),
    )


class RobotEmotionsPose2DTests(unittest.TestCase):
    def test_resolve_clip_output_dir_preserves_multi_take_suffix(self) -> None:
        record = _make_record("robot_emotions_30ms_u06_tag07_2")
        clip_dir = resolve_clip_output_dir("/tmp/robot_emotions_extract", record)
        self.assertEqual(
            clip_dir,
            Path("/tmp/robot_emotions_extract/30ms/user_06/robot_emotions_30ms_u06_tag07_2"),
        )

    def test_select_frame_indices_decimates_2997_to_20(self) -> None:
        frame_indices, effective_fps, timestamps_sec = select_frame_indices(
            num_frames=300,
            fps_original=29.97,
            fps_target=20,
        )
        self.assertEqual(float(effective_fps), 20.0)
        self.assertTrue(np.all(np.diff(frame_indices) > 0))
        self.assertEqual(int(frame_indices[0]), 0)
        self.assertLessEqual(int(frame_indices[-1]), 299)
        self.assertGreater(len(frame_indices), 180)
        self.assertLess(len(frame_indices), 250)
        self.assertEqual(len(frame_indices), len(timestamps_sec))

    def test_select_frame_indices_preserves_native_when_below_target(self) -> None:
        frame_indices, effective_fps, timestamps_sec = select_frame_indices(
            num_frames=30,
            fps_original=15.0,
            fps_target=20,
        )
        self.assertEqual(float(effective_fps), 15.0)
        self.assertTrue(np.array_equal(frame_indices, np.arange(30, dtype=np.int32)))
        self.assertEqual(len(frame_indices), len(timestamps_sec))

    def test_link_person_tracks_prefers_long_consistent_track(self) -> None:
        frame_predictions = []
        for frame_id in range(6):
            instances = [_pose_instance(frame_id, 10 + frame_id, 10, 50 + frame_id, 100)]
            if frame_id in {2, 3, 4}:
                instances.append(_pose_instance(frame_id, 150, 10, 200, 100))
            frame_predictions.append({"frame_id": frame_id, "instances": instances})

        tracks = link_person_tracks(frame_predictions)
        self.assertGreaterEqual(len(tracks), 2)
        self.assertEqual(tracks[0].detection_count, 6)
        report = build_person_track_report(tracks, selected_track=tracks[0], total_frames=6)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["selected_track_id"], tracks[0].track_id)

    def test_canonicalize_pose_sequence2d_marks_missing_joints_and_frames(self) -> None:
        track = TrackState(track_id=7)
        partial_keypoints = np.stack(
            [
                np.linspace(0, 10, 15, dtype=np.float32),
                np.linspace(5, 15, 15, dtype=np.float32),
            ],
            axis=1,
        )
        track.add_detection(
            PoseFrameInstance(
                frame_id=0,
                bbox_xyxy=np.asarray([0, 0, 10, 20], dtype=np.float32),
                bbox_score=0.9,
                keypoints_xy=partial_keypoints,
                keypoint_scores=np.full((15,), 0.8, dtype=np.float32),
            )
        )
        track.add_detection(
            PoseFrameInstance(
                frame_id=2,
                bbox_xyxy=np.asarray([1, 1, 11, 21], dtype=np.float32),
                bbox_score=0.85,
                keypoints_xy=np.stack(
                    [
                        np.linspace(1, 11, 17, dtype=np.float32),
                        np.linspace(2, 22, 17, dtype=np.float32),
                    ],
                    axis=1,
                ),
                keypoint_scores=np.full((17,), 0.75, dtype=np.float32),
            )
        )

        sequence, quality = canonicalize_pose_sequence2d(
            clip_id="clip_a",
            selected_track=track,
            selected_frame_indices=np.asarray([0, 1, 2], dtype=np.int32),
            timestamps_sec=np.asarray([0.0, 0.05, 0.1], dtype=np.float32),
            effective_fps=20.0,
            fps_original=29.97,
        )

        self.assertTrue(np.isnan(sequence.keypoints_xy[0, 15:, :]).all())
        self.assertTrue(np.all(sequence.confidence[0, 15:] == 0.0))
        self.assertTrue(np.isnan(sequence.keypoints_xy[1]).all())
        self.assertTrue(np.all(sequence.confidence[1] == 0.0))
        self.assertEqual(quality["status"], "warning")


if __name__ == "__main__":
    unittest.main()
