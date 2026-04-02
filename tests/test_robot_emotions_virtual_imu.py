import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from pose_module.interfaces import (
    IKSequence,
    IMUGPT_22_JOINT_NAMES,
    IMUGPT_22_PARENT_INDICES,
    PoseSequence3D,
    VirtualIMUSequence,
)
from pose_module.robot_emotions.cli import main as robot_emotions_cli_main
from pose_module.robot_emotions.extractor import RobotEmotionsClipRecord
from pose_module.robot_emotions.virtual_imu import run_robot_emotions_virtual_imu


def _make_record(clip_id: str) -> RobotEmotionsClipRecord:
    return RobotEmotionsClipRecord(
        clip_id=clip_id,
        domain="10ms",
        user_id=2,
        tag_number=5,
        tag_dir=Path("data/RobotEmotions/10ms/User2/Tag5"),
        imu_csv_path=Path("data/RobotEmotions/10ms/User2/Tag5/ESP_2_5.csv"),
        video_path=Path("data/RobotEmotions/10ms/User2/Tag5/TAG_2_5.mp4"),
        source_rel_dir="10ms/User2/Tag5",
        take_id=None,
        participant={"name": "Test User", "age": 20, "gender": "F"},
        protocol={"emotion": "Joy"},
    )


def _make_pose3d_sequence() -> PoseSequence3D:
    num_frames = 6
    num_joints = len(IMUGPT_22_JOINT_NAMES)
    root_translation = np.zeros((num_frames, 3), dtype=np.float32)
    joint_positions_xyz = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    for frame_index in range(num_frames):
        root_translation[frame_index] = np.asarray([0.02 * frame_index, 0.0, 0.0], dtype=np.float32)
        joint_positions_xyz[frame_index, :, 0] = root_translation[frame_index, 0] + np.linspace(
            -0.4,
            0.4,
            num_joints,
            dtype=np.float32,
        )
        joint_positions_xyz[frame_index, :, 1] = np.linspace(0.8, -0.8, num_joints, dtype=np.float32)
        joint_positions_xyz[frame_index, :, 2] = np.linspace(0.0, 0.3, num_joints, dtype=np.float32)
        joint_positions_xyz[frame_index, 0] = root_translation[frame_index]
    return PoseSequence3D(
        clip_id="clip_virtual_imu",
        fps=20.0,
        fps_original=30.0,
        joint_names_3d=list(IMUGPT_22_JOINT_NAMES),
        joint_positions_xyz=joint_positions_xyz,
        joint_confidence=np.full((num_frames, num_joints), 0.95, dtype=np.float32),
        skeleton_parents=list(IMUGPT_22_PARENT_INDICES),
        frame_indices=np.arange(num_frames, dtype=np.int32),
        timestamps_sec=np.arange(num_frames, dtype=np.float32) / np.float32(20.0),
        source="unit_test_pose3d_root",
        coordinate_space="pseudo_global_metric",
        root_translation_m=root_translation,
    )


def _make_ik_sequence() -> IKSequence:
    num_frames = 6
    num_joints = len(IMUGPT_22_JOINT_NAMES)
    local_joint_rotations = np.zeros((num_frames, num_joints, 4), dtype=np.float32)
    local_joint_rotations[..., 0] = 1.0
    return IKSequence(
        clip_id="clip_virtual_imu",
        fps=20.0,
        fps_original=30.0,
        joint_names_3d=list(IMUGPT_22_JOINT_NAMES),
        local_joint_rotations=local_joint_rotations,
        root_translation_m=np.zeros((num_frames, 3), dtype=np.float32),
        joint_offsets_m=np.zeros((num_joints, 3), dtype=np.float32),
        skeleton_parents=list(IMUGPT_22_PARENT_INDICES),
        frame_indices=np.arange(num_frames, dtype=np.int32),
        timestamps_sec=np.arange(num_frames, dtype=np.float32) / np.float32(20.0),
        source="unit_test_ik",
    )


def _apply_rotation(values_xyz: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    return np.asarray(values_xyz, dtype=np.float32) @ np.asarray(rotation_matrix, dtype=np.float32).T


def _shift_signal(values_xyz: np.ndarray, lag_samples: int) -> np.ndarray:
    values = np.asarray(values_xyz, dtype=np.float32)
    shifted = np.zeros_like(values)
    if lag_samples > 0:
        shifted[lag_samples:] = values[:-lag_samples]
    elif lag_samples < 0:
        lag = abs(lag_samples)
        shifted[:-lag] = values[lag:]
    else:
        shifted[:] = values
    return shifted


def _sensor_waveforms(timestamps: np.ndarray, *, capture_offset: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(timestamps, dtype=np.float32)
    gyro = np.stack(
        [
            0.65 * np.sin(1.4 * t + 0.4) + 0.18 * np.cos(0.4 * t + capture_offset),
            0.52 * np.cos(1.0 * t + 0.2) + 0.12 * np.sin(1.7 * t + capture_offset),
            0.48 * np.sin(1.8 * t + 0.1 + capture_offset),
        ],
        axis=1,
    ).astype(np.float32)
    acc = np.stack(
        [
            0.34 * np.sin(0.8 * t + 0.4) + 0.07 * np.cos(0.2 * t + capture_offset),
            9.81 + 0.21 * np.cos(0.6 * t + capture_offset) + 0.06 * np.sin(1.2 * t + 0.4),
            0.28 * np.sin(0.55 * t + 0.1) + 0.09 * np.cos(0.9 * t + capture_offset),
        ],
        axis=1,
    ).astype(np.float32)
    return acc, gyro


def _make_virtual_and_real_sequences(
    *,
    clip_id: str,
    rotation_matrix: np.ndarray,
    lag_samples: int,
    capture_offset: float,
) -> tuple[VirtualIMUSequence, np.ndarray, np.ndarray, np.ndarray]:
    virtual_timestamps = np.arange(240, dtype=np.float32) / np.float32(20.0)
    real_timestamps = np.arange(0.0, 12.0, 0.01, dtype=np.float32)
    virtual_acc, virtual_gyro = _sensor_waveforms(virtual_timestamps, capture_offset=capture_offset)
    real_acc_base, real_gyro_base = _sensor_waveforms(real_timestamps, capture_offset=capture_offset)
    real_lag_samples = int(round(float(lag_samples) * (0.05 / 0.01)))
    real_acc = _shift_signal(_apply_rotation(real_acc_base, rotation_matrix), real_lag_samples)[:, None, :]
    real_gyro = _shift_signal(_apply_rotation(real_gyro_base, rotation_matrix), real_lag_samples)[:, None, :]
    virtual_sequence = VirtualIMUSequence(
        clip_id=clip_id,
        fps=20.0,
        sensor_names=["left_forearm"],
        acc=virtual_acc[:, None, :],
        gyro=virtual_gyro[:, None, :],
        timestamps_sec=virtual_timestamps,
        source="unit_test_virtual_imu",
    )
    return virtual_sequence, real_acc, real_gyro, real_timestamps


class _FakeExtractor:
    def __init__(self, dataset_root: str, *, domains: tuple[str, ...]) -> None:
        self.dataset_root = dataset_root
        self.domains = domains

    def select_records(self, *, clip_ids=None):
        record = _make_record("robot_emotions_10ms_u02_tag05")
        if clip_ids is None:
            return [record]
        if record.clip_id in set(clip_ids):
            return [record]
        return []

    def ensure_exported_clip(self, record: RobotEmotionsClipRecord, *, output_root: str | Path):
        return {
            "labels": {"emotion": "Joy"},
            "source": {"video_path": str(record.video_path)},
            "video": {"fps": 30.0, "num_frames": 120, "duration_sec": 4.0},
            "artifacts": {
                "imu_npz_path": str(Path(output_root) / "imu.npz"),
                "metadata_json_path": str(Path(output_root) / "metadata.json"),
            },
        }


class RobotEmotionsVirtualIMUTests(unittest.TestCase):
    def test_run_robot_emotions_virtual_imu_writes_manifest_and_summary(self) -> None:
        record = _make_record("robot_emotions_10ms_u02_tag05")
        fake_pipeline_result = {
            "clip_id": record.clip_id,
            "pose_sequence": _make_pose3d_sequence(),
            "virtual_imu_sequence": VirtualIMUSequence(
                clip_id=record.clip_id,
                fps=20.0,
                sensor_names=["waist", "head", "right_forearm", "left_forearm"],
                acc=np.zeros((6, 4, 3), dtype=np.float32),
                gyro=np.zeros((6, 4, 3), dtype=np.float32),
                timestamps_sec=np.arange(6, dtype=np.float32) / np.float32(20.0),
                source="unit_test_virtual_imu",
            ),
            "ik_sequence": _make_ik_sequence(),
            "quality_report": {"clip_id": record.clip_id, "status": "ok", "notes": []},
            "pose3d_quality_report": {"clip_id": record.clip_id, "status": "ok"},
            "ik_quality_report": {"clip_id": record.clip_id, "status": "ok", "ik_ok": True},
            "virtual_imu_quality_report": {
                "clip_id": record.clip_id,
                "status": "ok",
                "virtual_imu_ok": True,
            },
            "artifacts": {
                "pose3d_npz_path": "/tmp/fake_pose3d.npz",
                "ik_sequence_npz_path": "/tmp/fake_ik_sequence.npz",
                "ik_bvh_path": "/tmp/fake_pose3d_ik.bvh",
                "virtual_imu_npz_path": "/tmp/fake_virtual_imu.npz",
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("pose_module.robot_emotions.virtual_imu.RobotEmotionsExtractor", _FakeExtractor):
                with patch(
                    "pose_module.robot_emotions.virtual_imu.run_virtual_imu_pipeline",
                    return_value=fake_pipeline_result,
                ) as mocked_pipeline:
                    with patch(
                        "pose_module.robot_emotions.virtual_imu.load_alignment_runtime_settings",
                        return_value={"enable": False, "fit_from_current_pair": False},
                    ):
                        summary = run_robot_emotions_virtual_imu(
                            dataset_root="data/RobotEmotions",
                            output_dir=tmp_dir,
                            clip_ids=[record.clip_id],
                            env_name="openmmlab",
                            estimate_sensor_frame=True,
                            estimate_sensor_names=["left_forearm", "right_forearm"],
                        )

            self.assertEqual(summary["num_ok"], 1)
            manifest_path = Path(summary["virtual_imu_manifest_path"])
            self.assertTrue(manifest_path.exists())
            manifest_entries = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip() != ""
            ]
            self.assertEqual(manifest_entries[0]["artifacts"]["virtual_imu_npz_path"], "/tmp/fake_virtual_imu.npz")
            self.assertTrue(manifest_entries[0]["ik_quality_report"]["ik_ok"])
            self.assertTrue(manifest_entries[0]["virtual_imu_quality_report"]["virtual_imu_ok"])
            mocked_pipeline.assert_called_once()
            self.assertTrue(mocked_pipeline.call_args.kwargs["estimate_sensor_frame"])
            self.assertEqual(
                mocked_pipeline.call_args.kwargs["estimate_sensor_names"],
                ("left_forearm", "right_forearm"),
            )

    def test_cli_export_virtual_imu_dispatches_wrapper(self) -> None:
        with patch(
            "pose_module.robot_emotions.cli.run_robot_emotions_virtual_imu",
            return_value={"status": "ok"},
        ) as mocked_wrapper:
            with patch("builtins.print") as mocked_print:
                exit_code = robot_emotions_cli_main(
                    [
                        "export-virtual-imu",
                        "--output-dir",
                        "/tmp/robot_emotions_virtual_imu",
                        "--clip-id",
                        "robot_emotions_10ms_u02_tag05",
                        "--estimate-sensor-frame",
                        "--estimate-sensor-names",
                        "left_forearm",
                        "right_forearm",
                    ]
                )

        self.assertEqual(exit_code, 0)
        mocked_wrapper.assert_called_once()
        self.assertEqual(mocked_wrapper.call_args.kwargs["output_dir"], "/tmp/robot_emotions_virtual_imu")
        self.assertTrue(mocked_wrapper.call_args.kwargs["estimate_sensor_frame"])
        self.assertEqual(
            mocked_wrapper.call_args.kwargs["estimate_sensor_names"],
            ["left_forearm", "right_forearm"],
        )
        mocked_print.assert_called_once()

    def test_run_robot_emotions_virtual_imu_defers_rank_calibration_for_subject_level_alignment(self) -> None:
        record = _make_record("robot_emotions_10ms_u02_tag05")
        fake_pipeline_result = {
            "clip_id": record.clip_id,
            "pose_sequence": _make_pose3d_sequence(),
            "virtual_imu_sequence": VirtualIMUSequence(
                clip_id=record.clip_id,
                fps=20.0,
                sensor_names=["waist", "head", "right_forearm", "left_forearm"],
                acc=np.zeros((6, 4, 3), dtype=np.float32),
                gyro=np.zeros((6, 4, 3), dtype=np.float32),
                timestamps_sec=np.arange(6, dtype=np.float32) / np.float32(20.0),
                source="unit_test_virtual_imu",
            ),
            "ik_sequence": _make_ik_sequence(),
            "quality_report": {"clip_id": record.clip_id, "status": "ok", "notes": []},
            "pose3d_quality_report": {"clip_id": record.clip_id, "status": "ok"},
            "ik_quality_report": {"clip_id": record.clip_id, "status": "ok", "ik_ok": True},
            "virtual_imu_quality_report": {
                "clip_id": record.clip_id,
                "status": "ok",
                "virtual_imu_ok": True,
            },
            "artifacts": {"virtual_imu_npz_path": "/tmp/fake_virtual_imu.npz"},
            "imusim_result": {
                "raw_virtual_imu_sequence": VirtualIMUSequence(
                    clip_id=record.clip_id,
                    fps=20.0,
                    sensor_names=["waist", "head", "right_forearm", "left_forearm"],
                    acc=np.zeros((6, 4, 3), dtype=np.float32),
                    gyro=np.zeros((6, 4, 3), dtype=np.float32),
                    timestamps_sec=np.arange(6, dtype=np.float32) / np.float32(20.0),
                    source="unit_test_virtual_imu_raw",
                ),
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("pose_module.robot_emotions.virtual_imu.RobotEmotionsExtractor", _FakeExtractor):
                with patch(
                    "pose_module.robot_emotions.virtual_imu.run_virtual_imu_pipeline",
                    return_value=fake_pipeline_result,
                ) as mocked_pipeline:
                    with patch(
                        "pose_module.robot_emotions.virtual_imu.load_alignment_runtime_settings",
                        return_value={"enable": True, "fit_from_current_pair": False},
                    ):
                        with patch(
                            "pose_module.robot_emotions.virtual_imu._apply_subject_level_geometric_alignment",
                            return_value=None,
                        ):
                            run_robot_emotions_virtual_imu(
                                dataset_root="data/RobotEmotions",
                                output_dir=tmp_dir,
                                clip_ids=[record.clip_id],
                                env_name="openmmlab",
                                real_imu_reference_path="/tmp/reference.npz",
                            )

            self.assertTrue(mocked_pipeline.call_args.kwargs["defer_real_imu_calibration"])

    def test_run_robot_emotions_virtual_imu_fits_subject_level_geometric_alignment(self) -> None:
        record_a = _make_record("robot_emotions_10ms_u02_tag05")
        record_b = _make_record("robot_emotions_10ms_u02_tag06")
        rotation_matrix = Rotation.from_euler("xyz", [18.0, -12.0, 26.0], degrees=True).as_matrix().astype(
            np.float32
        )
        virtual_a, real_acc_a, real_gyro_a, timestamps_a = _make_virtual_and_real_sequences(
            clip_id=record_a.clip_id,
            rotation_matrix=rotation_matrix,
            lag_samples=3,
            capture_offset=0.2,
        )
        virtual_b, real_acc_b, real_gyro_b, timestamps_b = _make_virtual_and_real_sequences(
            clip_id=record_b.clip_id,
            rotation_matrix=rotation_matrix,
            lag_samples=-2,
            capture_offset=0.8,
        )
        virtual_by_clip = {
            record_a.clip_id: virtual_a,
            record_b.clip_id: virtual_b,
        }
        real_by_clip = {
            record_a.clip_id: (real_acc_a, real_gyro_a, timestamps_a),
            record_b.clip_id: (real_acc_b, real_gyro_b, timestamps_b),
        }

        class _TwoClipExtractor:
            def __init__(self, dataset_root: str, *, domains: tuple[str, ...]) -> None:
                self.dataset_root = dataset_root
                self.domains = domains

            def select_records(self, *, clip_ids=None):
                records = [record_a, record_b]
                if clip_ids is None:
                    return records
                requested = set(clip_ids)
                return [record for record in records if record.clip_id in requested]

            def ensure_exported_clip(self, record: RobotEmotionsClipRecord, *, output_root: str | Path):
                clip_dir = Path(output_root) / record.domain / f"user_{record.user_id:02d}" / record.clip_id
                clip_dir.mkdir(parents=True, exist_ok=True)
                real_acc, real_gyro, timestamps = real_by_clip[record.clip_id]
                imu_npz_path = clip_dir / "imu.npz"
                np.savez_compressed(
                    imu_npz_path,
                    acc=real_acc,
                    gyro=real_gyro,
                    timestamps_sec=timestamps,
                    sensor_names=np.asarray(["left_forearm"]),
                )
                metadata_json_path = clip_dir / "metadata.json"
                metadata_json_path.write_text(
                    json.dumps(
                        {
                            "domain": record.domain,
                            "user_id": record.user_id,
                            "clip_id": record.clip_id,
                            "imu": {"sensor_names": ["left_forearm"]},
                        },
                        ensure_ascii=True,
                    ),
                    encoding="utf-8",
                )
                return {
                    "labels": {"emotion": "Joy"},
                    "source": {"video_path": str(record.video_path)},
                    "video": {"fps": 30.0, "num_frames": int(timestamps.shape[0]), "duration_sec": 12.0},
                    "artifacts": {
                        "imu_npz_path": str(imu_npz_path.resolve()),
                        "metadata_json_path": str(metadata_json_path.resolve()),
                    },
                }

        def _fake_pipeline(*, clip_id: str, output_dir: str | Path, **kwargs):
            pose_dir = Path(output_dir)
            pose_dir.mkdir(parents=True, exist_ok=True)
            virtual_sequence = virtual_by_clip[clip_id]
            np.savez_compressed(pose_dir / "virtual_imu.npz", **virtual_sequence.to_npz_payload())
            return {
                "clip_id": clip_id,
                "pose_sequence": _make_pose3d_sequence(),
                "virtual_imu_sequence": virtual_sequence,
                "ik_sequence": _make_ik_sequence(),
                "quality_report": {"clip_id": clip_id, "status": "ok", "notes": []},
                "pose3d_quality_report": {"clip_id": clip_id, "status": "ok", "notes": []},
                "ik_quality_report": {"clip_id": clip_id, "status": "ok", "ik_ok": True, "notes": []},
                "virtual_imu_quality_report": {
                    "clip_id": clip_id,
                    "status": "ok",
                    "virtual_imu_ok": True,
                    "acc_noise_std_m_s2": 0.0,
                    "gyro_noise_std_rad_s": 0.0,
                    "notes": [],
                },
                "frame_alignment_quality_report": {"enabled": False, "status": None, "notes": []},
                "virtual_imu_calibration_report": None,
                "geometric_alignment_quality_report": {"enabled": False, "status": None, "notes": []},
                "artifacts": {
                    "virtual_imu_npz_path": str((pose_dir / "virtual_imu.npz").resolve()),
                    "quality_report_json_path": str((pose_dir / "quality_report.json").resolve()),
                },
                "imusim_result": {
                    "virtual_imu_sequence": virtual_sequence,
                    "raw_virtual_imu_sequence": virtual_sequence,
                    "quality_report": {
                        "clip_id": clip_id,
                        "status": "ok",
                        "virtual_imu_ok": True,
                        "acc_noise_std_m_s2": 0.0,
                        "gyro_noise_std_rad_s": 0.0,
                        "notes": [],
                    },
                    "calibration_report": None,
                    "artifacts": {
                        "virtual_imu_npz_path": str((pose_dir / "virtual_imu.npz").resolve()),
                        "virtual_imu_report_json_path": str((pose_dir / "virtual_imu_report.json").resolve()),
                        "virtual_imu_calibration_report_json_path": None,
                    },
                },
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("pose_module.robot_emotions.virtual_imu.RobotEmotionsExtractor", _TwoClipExtractor):
                with patch(
                    "pose_module.robot_emotions.virtual_imu.run_virtual_imu_pipeline",
                    side_effect=_fake_pipeline,
                ):
                    with patch(
                        "pose_module.robot_emotions.virtual_imu.load_alignment_runtime_settings",
                        return_value={"enable": True, "fit_from_current_pair": False},
                    ):
                        summary = run_robot_emotions_virtual_imu(
                            dataset_root="data/RobotEmotions",
                            output_dir=tmp_dir,
                            clip_ids=[record_a.clip_id, record_b.clip_id],
                            env_name="openmmlab",
                        )

            self.assertEqual(summary["num_ok"], 2)
            transform_path = Path(tmp_dir) / "10ms" / "user_02" / "imu_alignment_transforms.json"
            self.assertTrue(transform_path.exists())
            manifest_entries = [
                json.loads(line)
                for line in Path(summary["virtual_imu_manifest_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip() != ""
            ]
            self.assertEqual(len(manifest_entries), 2)
            for entry in manifest_entries:
                quality = entry["quality_report"]
                self.assertTrue(quality["geometric_alignment_enabled"])
                self.assertEqual(quality["geometric_alignment_status"], "ok")
                self.assertGreater(
                    float(quality["geometric_alignment_mean_gyro_corr_after"]),
                    float(quality["geometric_alignment_mean_gyro_corr_before"]),
                )
