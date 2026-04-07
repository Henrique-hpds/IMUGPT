import unittest

import numpy as np
import pandas as pd

from evaluation.classifiers.data import ALL_CAPTURE_BLACKLIST, _merge_capture_table_with_manifest, apply_capture_blacklist
from evaluation.classifiers.alignment import align_target_to_reference, estimate_lag_cross_correlation
from evaluation.classifiers.features import (
    build_imu_feature_tensor,
    build_pose_feature_tensor,
    build_pose_sensor_proxy,
    extract_quality_vector,
)
from pose_module.interfaces import IMUGPT_22_JOINT_NAMES, IMUGPT_22_PARENT_INDICES, PoseSequence3D


def _shift_signal(values: np.ndarray, lag_samples: int) -> np.ndarray:
    block = np.asarray(values, dtype=np.float32)
    shifted = np.zeros_like(block)
    if lag_samples > 0:
        shifted[lag_samples:] = block[:-lag_samples]
    elif lag_samples < 0:
        lag_abs = abs(lag_samples)
        shifted[:-lag_abs] = block[lag_abs:]
    else:
        shifted[:] = block
    return shifted


class EvaluationClassifierAlignmentTests(unittest.TestCase):
    def test_merge_capture_table_with_manifest_keeps_rows_when_manifest_take_id_is_missing(self) -> None:
        base_table = pd.DataFrame(
            [
                {"clip_id": "clip_a", "domain": "30ms", "user_id": 6, "tag_number": 7, "take_id": "1"},
                {"clip_id": "clip_b", "domain": "30ms", "user_id": 6, "tag_number": 7, "take_id": "2"},
            ]
        )
        manifest_frame = pd.DataFrame(
            [
                {"clip_id": "clip_a", "domain": "30ms", "user_id": 6, "tag_number": 7, "take_id": None},
                {"clip_id": "clip_b", "domain": "30ms", "user_id": 6, "tag_number": 7, "take_id": None},
            ]
        )

        merged = _merge_capture_table_with_manifest(base_table, manifest_frame)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged["take_id"].tolist(), ["1", "2"])

    def test_apply_capture_blacklist_supports_capture_and_take_specific_entries(self) -> None:
        captures_df = pd.DataFrame(
            [
                {"domain": "30ms", "user_id": 2, "tag_number": 3, "take_id": "1", "clip_id": "blocked_all_takes_1"},
                {"domain": "30ms", "user_id": 2, "tag_number": 3, "take_id": "2", "clip_id": "blocked_all_takes_2"},
                {"domain": "30ms", "user_id": 6, "tag_number": 7, "take_id": "1", "clip_id": "blocked_take_1_only"},
                {"domain": "30ms", "user_id": 6, "tag_number": 7, "take_id": "2", "clip_id": "allowed_take_2"},
                {"domain": "10ms", "user_id": 2, "tag_number": 3, "take_id": "1", "clip_id": "allowed_other_domain"},
            ]
        )

        filtered = apply_capture_blacklist(captures_df, capture_blacklist=ALL_CAPTURE_BLACKLIST)
        annotated = apply_capture_blacklist(
            captures_df,
            capture_blacklist=ALL_CAPTURE_BLACKLIST,
            drop_blacklisted=False,
        )

        self.assertEqual(
            filtered["clip_id"].tolist(),
            ["allowed_take_2", "allowed_other_domain"],
        )
        self.assertEqual(
            annotated["is_blacklisted"].tolist(),
            [True, True, True, False, False],
        )

    def test_estimate_lag_cross_correlation_recovers_shift_magnitude(self) -> None:
        timestamps_sec = np.arange(0.0, 8.0, 0.05, dtype=np.float32)
        reference_signal = np.sin(2.3 * timestamps_sec) + 0.15 * np.cos(0.7 * timestamps_sec)
        target_signal = _shift_signal(reference_signal, 4)

        lag_report = estimate_lag_cross_correlation(
            reference_signal,
            target_signal,
            max_lag_samples=8,
        )

        self.assertEqual(abs(int(lag_report["lag_samples"])), 4)
        self.assertGreater(float(lag_report["correlation"]), 0.9)

    def test_align_target_to_reference_improves_summary_correlation(self) -> None:
        timestamps_sec = np.arange(0.0, 10.0, 0.05, dtype=np.float32)
        reference_summary = np.sin(1.7 * timestamps_sec) + 0.25 * np.cos(0.4 * timestamps_sec)
        target_summary = _shift_signal(reference_summary, 5)

        reference_values = np.stack(
            [
                reference_summary,
                np.gradient(reference_summary, timestamps_sec).astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        target_values = np.stack(
            [
                target_summary,
                0.5 * target_summary,
                np.gradient(target_summary, timestamps_sec).astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)

        aligned = align_target_to_reference(
            timestamps_sec,
            reference_values,
            timestamps_sec,
            target_values,
            reference_summary=reference_summary,
            target_summary=target_summary,
            max_lag_samples=8,
            dtw_radius=8,
        )

        before_corr = aligned["correlation_before_dtw"]
        after_corr = aligned["correlation_after_dtw"]
        self.assertIsNotNone(before_corr)
        self.assertIsNotNone(after_corr)
        self.assertGreater(float(after_corr), float(before_corr))
        self.assertEqual(aligned["reference_values"].shape[0], aligned["aligned_target_values"].shape[0])
        self.assertGreater(aligned["reference_values"].shape[0], 50)

    def test_pose_feature_tensor_and_sensor_proxy_follow_pipeline_contract(self) -> None:
        num_frames = 40
        timestamps_sec = np.arange(num_frames, dtype=np.float32) / np.float32(20.0)
        joint_positions_xyz = np.zeros((num_frames, len(IMUGPT_22_JOINT_NAMES), 3), dtype=np.float32)
        for joint_index in range(len(IMUGPT_22_JOINT_NAMES)):
            joint_positions_xyz[:, joint_index, 0] = 0.05 * joint_index
            joint_positions_xyz[:, joint_index, 1] = np.sin(timestamps_sec + 0.1 * joint_index)
            joint_positions_xyz[:, joint_index, 2] = np.cos(0.5 * timestamps_sec + 0.07 * joint_index)

        sequence = PoseSequence3D(
            clip_id="unit_test_clip",
            fps=20.0,
            fps_original=20.0,
            joint_names_3d=IMUGPT_22_JOINT_NAMES,
            joint_positions_xyz=joint_positions_xyz,
            joint_confidence=np.ones((num_frames, len(IMUGPT_22_JOINT_NAMES)), dtype=np.float32),
            skeleton_parents=IMUGPT_22_PARENT_INDICES,
            frame_indices=np.arange(num_frames, dtype=np.int32),
            timestamps_sec=timestamps_sec,
            source="unit_test",
            coordinate_space="pseudo_global_metric",
        )

        feature_tensor = build_pose_feature_tensor(sequence)
        proxy = build_pose_sensor_proxy(sequence)

        self.assertEqual(feature_tensor["values"].shape, (num_frames, len(IMUGPT_22_JOINT_NAMES), 16))
        self.assertEqual(len(feature_tensor["channel_names"]), 16)
        self.assertEqual(proxy["acc"].shape, (num_frames, 4, 3))
        self.assertEqual(proxy["gyro"].shape, (num_frames, 4, 3))
        self.assertEqual(proxy["sensor_names"], ["waist", "head", "right_forearm", "left_forearm"])

    def test_build_imu_feature_tensor_supports_acc_euler_mode_without_using_gyro(self) -> None:
        timestamps_sec = np.asarray([0.0, 0.1, 0.2], dtype=np.float32)
        acc = np.asarray(
            [
                [[1.0, 2.0, 2.0]],
                [[2.0, 1.0, 2.0]],
                [[2.0, 2.0, 1.0]],
            ],
            dtype=np.float32,
        )
        zero_gyro = np.zeros_like(acc)
        noisy_gyro = np.full_like(acc, 123.0)

        zero_gyro_features = build_imu_feature_tensor(
            acc,
            zero_gyro,
            timestamps_sec,
            feature_mode="acc_euler",
        )
        noisy_gyro_features = build_imu_feature_tensor(
            acc,
            noisy_gyro,
            timestamps_sec,
            feature_mode="acc_euler",
        )

        np.testing.assert_allclose(
            zero_gyro_features["values"],
            noisy_gyro_features["values"],
            atol=1e-6,
        )
        self.assertEqual(
            zero_gyro_features["channel_names"][:6],
            [
                "acc_x",
                "acc_y",
                "acc_z",
                "euler_theta_deg",
                "euler_psi_deg",
                "euler_phi_deg",
            ],
        )
        self.assertEqual(zero_gyro_features["feature_mode"], "acc_euler")

        expected_euler = np.asarray(
            [
                [
                    np.degrees(np.arctan(1.0 / np.sqrt((2.0 ** 2) + (2.0 ** 2)))),
                    np.degrees(np.arctan(2.0 / np.sqrt((1.0 ** 2) + (2.0 ** 2)))),
                    np.degrees(np.arctan(2.0 / np.sqrt((1.0 ** 2) + (2.0 ** 2)))),
                ],
                [
                    np.degrees(np.arctan(2.0 / np.sqrt((1.0 ** 2) + (2.0 ** 2)))),
                    np.degrees(np.arctan(1.0 / np.sqrt((2.0 ** 2) + (2.0 ** 2)))),
                    np.degrees(np.arctan(2.0 / np.sqrt((2.0 ** 2) + (1.0 ** 2)))),
                ],
                [
                    np.degrees(np.arctan(2.0 / np.sqrt((2.0 ** 2) + (1.0 ** 2)))),
                    np.degrees(np.arctan(2.0 / np.sqrt((2.0 ** 2) + (1.0 ** 2)))),
                    np.degrees(np.arctan(1.0 / np.sqrt((2.0 ** 2) + (2.0 ** 2)))),
                ],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            zero_gyro_features["values"][:, 0, 3:6],
            expected_euler,
            atol=1e-5,
        )

    def test_extract_quality_vector_omits_gyro_metric_in_acc_euler_mode(self) -> None:
        quality_vector = extract_quality_vector(
            {
                "visible_joint_ratio": 0.8,
                "mean_confidence": 0.7,
                "temporal_jitter_score": 0.1,
                "root_drift_score": 0.2,
                "geometric_alignment_mean_acc_corr_after": 0.9,
                "geometric_alignment_mean_gyro_corr_after": 0.3,
            },
            pose_imu_alignment={
                "correlation_after_dtw": 0.95,
                "dtw_normalized_distance": 0.05,
            },
            imu_feature_mode="acc_euler",
        )

        self.assertNotIn("geometric_alignment_mean_gyro_corr_after", quality_vector["feature_names"])
        self.assertEqual(len(quality_vector["feature_names"]), 7)


if __name__ == "__main__":
    unittest.main()
