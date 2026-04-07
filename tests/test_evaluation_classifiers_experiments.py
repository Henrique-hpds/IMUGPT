import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from evaluation.classifiers.experiments import SplitConfig, _build_oof_summary, run_experiment_suite
from evaluation.classifiers.metrics import build_support_report, compute_multitask_metrics
from evaluation.classifiers.training import (
    ModelConfig,
    TrainingConfig,
    _build_classification_arrays,
    _gather_predictions,
    train_multitask_model,
)

try:
    import torch
except ImportError:  # pragma: no cover - depends on optional torch dependency
    torch = None


def _make_label_encoders() -> dict[str, dict[str, object]]:
    return {
        "emotion": {"class_names": ["A", "B"]},
        "modality": {"class_names": ["Sit", "Stand"]},
        "stimulus": {"class_names": ["Low", "High"]},
        "flat_tag": {"class_names": ["TagA", "TagB"]},
    }


def _make_small_arrays(classification_mask: np.ndarray | None = None) -> dict[str, object]:
    num_samples = 3
    if classification_mask is None:
        classification_mask = np.ones(num_samples, dtype=np.float32)
    return {
        "pose": np.arange(num_samples * 3, dtype=np.float32).reshape(num_samples, 3, 1, 1),
        "imu": np.arange(num_samples * 3, dtype=np.float32).reshape(num_samples, 3, 1, 1),
        "quality": np.ones((num_samples, 1), dtype=np.float32),
        "domain": np.asarray([1, 0, 1], dtype=np.int64),
        "classification_mask": np.asarray(classification_mask, dtype=np.float32),
        "targets": {
            "emotion": np.asarray([0, 1, 0], dtype=np.int64),
            "modality": np.asarray([0, 1, 0], dtype=np.int64),
            "stimulus": np.asarray([0, 1, 0], dtype=np.int64),
            "flat_tag": np.asarray([0, 1, 0], dtype=np.int64),
        },
        "metadata": pd.DataFrame(
            {
                "sample_id": ["sample_0", "sample_1", "sample_2"],
                "capture_id": ["cap_0", "cap_1", "cap_2"],
            }
        ),
    }


def _make_suite_dataset_bundle() -> dict[str, object]:
    num_samples = 6
    time_steps = 9
    num_joints = 22
    pose_channels = 16
    num_sensors = 4
    imu_channels = 12
    rng = np.random.default_rng(7)

    metadata = pd.DataFrame(
        {
            "sample_id": [f"sample_{index}" for index in range(num_samples)],
            "capture_id": [f"cap_{index}" for index in range(num_samples)],
            "clip_id": [f"clip_{index}" for index in range(num_samples)],
            "subject_group": ["g0", "g0", "g1", "g1", "g2", "g2"],
            "emotion": ["A", "B", "A", "B", "A", "B"],
            "modality": ["Sit", "Stand", "Sit", "Stand", "Sit", "Stand"],
            "stimulus": ["Low", "High", "Low", "High", "Low", "High"],
            "flat_tag": ["TagA", "TagB", "TagA", "TagB", "TagA", "TagB"],
        }
    )
    label_encoders = {
        "emotion": {"class_names": ["A", "B"], "mapping": {"A": 0, "B": 1}},
        "modality": {"class_names": ["Sit", "Stand"], "mapping": {"Sit": 0, "Stand": 1}},
        "stimulus": {"class_names": ["Low", "High"], "mapping": {"Low": 0, "High": 1}},
        "flat_tag": {"class_names": ["TagA", "TagB"], "mapping": {"TagA": 0, "TagB": 1}},
    }
    metadata["emotion_id"] = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    metadata["modality_id"] = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    metadata["stimulus_id"] = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    metadata["flat_tag_id"] = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)

    return {
        "metadata": metadata,
        "pose_windows": rng.normal(size=(num_samples, time_steps, num_joints, pose_channels)).astype(np.float32),
        "imu_real_windows": rng.normal(size=(num_samples, time_steps, num_sensors, imu_channels)).astype(np.float32),
        "imu_synthetic_windows": rng.normal(size=(num_samples, time_steps, num_sensors, imu_channels)).astype(np.float32),
        "quality_windows": rng.normal(size=(num_samples, 2)).astype(np.float32),
        "label_encoders": label_encoders,
    }


@unittest.skipIf(torch is None, "PyTorch is required for classifier experiment tests.")
class EvaluationClassifierExperimentTests(unittest.TestCase):
    def test_build_classification_arrays_excludes_unsupervised_rows(self) -> None:
        arrays = _make_small_arrays(classification_mask=np.asarray([1.0, 0.0, 1.0], dtype=np.float32))

        classification_arrays = _build_classification_arrays(arrays)

        self.assertEqual(classification_arrays["metadata"]["sample_id"].tolist(), ["sample_0", "sample_2"])
        self.assertTrue(np.allclose(classification_arrays["classification_mask"], np.ones(2, dtype=np.float32)))

    def test_gather_predictions_ignores_unsupervised_rows_for_metrics(self) -> None:
        class FixedLogitModel(torch.nn.Module):
            def forward(self, *, pose_inputs, imu_inputs, quality_inputs, domain_lambda=1.0):  # type: ignore[override]
                batch_size = pose_inputs.shape[0]
                logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]], dtype=torch.float32, device=pose_inputs.device)
                logits = logits[:batch_size]
                return {
                    "emotion_logits": logits,
                    "modality_logits": logits,
                    "stimulus_logits": logits,
                    "flat_tag_logits": logits,
                }

        arrays = _make_small_arrays(classification_mask=np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
        report = _gather_predictions(
            FixedLogitModel(),
            arrays,
            _make_label_encoders(),
            batch_size=3,
            device=torch.device("cpu"),
            filter_unsupervised=True,
        )

        self.assertEqual(report["metadata"]["sample_id"].tolist(), ["sample_0"])
        self.assertEqual(report["targets"]["emotion"].tolist(), [0])
        self.assertEqual(report["metrics"]["per_head"]["emotion"]["confusion_matrix"], [[1, 0], [0, 0]])

    def test_train_multitask_model_uses_grl_lambda_without_domain_loss_weight_scaling(self) -> None:
        class RecordingModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scalar = torch.nn.Parameter(torch.tensor(0.1))
                self.domain_lambdas: list[float] = []

            def forward(self, *, pose_inputs, imu_inputs, quality_inputs, domain_lambda=1.0):  # type: ignore[override]
                self.domain_lambdas.append(float(domain_lambda))
                batch_size = pose_inputs.shape[0]
                logits = self.scalar * torch.ones((batch_size, 2), dtype=torch.float32, device=pose_inputs.device)
                return {
                    "emotion_logits": logits,
                    "modality_logits": logits,
                    "stimulus_logits": logits,
                    "flat_tag_logits": logits,
                    "domain_logits": logits,
                }

        model = RecordingModel()
        train_arrays = _make_small_arrays(classification_mask=np.asarray([1.0, 1.0, 0.0], dtype=np.float32))
        val_arrays = _make_small_arrays(classification_mask=np.asarray([1.0, 0.0, 0.0], dtype=np.float32))

        with patch("evaluation.classifiers.training.build_model", return_value=model):
            train_multitask_model(
                train_arrays=train_arrays,
                val_arrays=val_arrays,
                label_encoders=_make_label_encoders(),
                model_config=ModelConfig(hidden_dim=8, quality_dim=1),
                training_config=TrainingConfig(
                    max_epochs=1,
                    batch_size=2,
                    classification_batch_size=2,
                    domain_batch_size=2,
                    domain_loss_weight=0.1,
                    grl_lambda=0.7,
                    device="cpu",
                ),
                use_pose_branch=True,
                use_imu_branch=True,
                use_domain_head=True,
            )

        self.assertIn(0.0, model.domain_lambdas)
        self.assertIn(0.7, model.domain_lambdas)
        self.assertNotIn(0.07, model.domain_lambdas)

    def test_build_support_report_marks_single_subject_class_as_unsupported(self) -> None:
        metadata = pd.DataFrame(
            {
                "subject_group": ["g0", "g1", "g2", "g2"],
                "emotion_id": [0, 1, 1, 2],
            }
        )
        label_encoders = {"emotion": {"class_names": ["Rare", "Shared", "SingleGroup"]}}

        support_report = build_support_report(
            metadata,
            label_encoders,
            group_column="subject_group",
            min_subject_groups=2,
        )

        status_by_class = {
            row["class_name"]: row["support_status"]
            for _, row in support_report.iterrows()
        }
        self.assertEqual(status_by_class["Rare"], "unsupported")
        self.assertEqual(status_by_class["Shared"], "supported")
        self.assertEqual(status_by_class["SingleGroup"], "unsupported")

    def test_oof_summary_uses_concatenated_predictions_instead_of_fold_mean(self) -> None:
        label_encoders = _make_label_encoders()
        support_report = pd.DataFrame(
            [
                {"head_name": "emotion", "class_id": 0, "class_name": "A", "is_supported": True},
                {"head_name": "emotion", "class_id": 1, "class_name": "B", "is_supported": True},
                {"head_name": "modality", "class_id": 0, "class_name": "Sit", "is_supported": True},
                {"head_name": "modality", "class_id": 1, "class_name": "Stand", "is_supported": True},
                {"head_name": "stimulus", "class_id": 0, "class_name": "Low", "is_supported": True},
                {"head_name": "stimulus", "class_id": 1, "class_name": "High", "is_supported": True},
            ]
        )
        scored_class_ids = {"emotion": [0, 1], "modality": [0, 1], "stimulus": [0, 1]}
        oof_reports = {
            "vision_only": {
                "metrics": compute_multitask_metrics(
                    y_true={
                        "emotion": np.asarray([0, 0, 1, 1], dtype=np.int64),
                        "modality": np.asarray([0, 0, 1, 1], dtype=np.int64),
                        "stimulus": np.asarray([0, 0, 1, 1], dtype=np.int64),
                    },
                    y_pred={
                        "emotion": np.asarray([0, 1, 1, 1], dtype=np.int64),
                        "modality": np.asarray([0, 1, 1, 1], dtype=np.int64),
                        "stimulus": np.asarray([0, 1, 1, 1], dtype=np.int64),
                    },
                    probabilities={
                        "emotion": np.asarray([[0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]], dtype=np.float32),
                        "modality": np.asarray([[0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]], dtype=np.float32),
                        "stimulus": np.asarray([[0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]], dtype=np.float32),
                    },
                    label_encoders=label_encoders,
                    scored_class_ids=scored_class_ids,
                ),
                "metadata": pd.DataFrame({"sample_id": ["s0", "s1", "s2", "s3"]}),
            }
        }
        summary = _build_oof_summary(
            oof_reports,
            support_report=support_report,
            primary_head="emotion",
        )

        expected_metrics = compute_multitask_metrics(
            y_true={
                "emotion": np.asarray([0, 0, 1, 1], dtype=np.int64),
                "modality": np.asarray([0, 0, 1, 1], dtype=np.int64),
                "stimulus": np.asarray([0, 0, 1, 1], dtype=np.int64),
            },
            y_pred={
                "emotion": np.asarray([0, 1, 1, 1], dtype=np.int64),
                "modality": np.asarray([0, 1, 1, 1], dtype=np.int64),
                "stimulus": np.asarray([0, 1, 1, 1], dtype=np.int64),
            },
            probabilities={
                "emotion": np.asarray([[0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]], dtype=np.float32),
                "modality": np.asarray([[0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]], dtype=np.float32),
                "stimulus": np.asarray([[0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]], dtype=np.float32),
            },
            label_encoders=label_encoders,
            scored_class_ids=scored_class_ids,
        )
        fold_mean = np.mean([1.0, 0.4])

        self.assertAlmostEqual(
            float(summary.iloc[0]["emotion_macro_f1_supported_oof"]),
            float(expected_metrics["per_head"]["emotion"]["supported_macro_f1"]),
        )
        self.assertNotAlmostEqual(float(summary.iloc[0]["emotion_macro_f1_supported_oof"]), float(fold_mean))

    def test_run_experiment_suite_returns_oof_summary_and_support_report(self) -> None:
        dataset_bundle = _make_suite_dataset_bundle()

        result = run_experiment_suite(
            dataset_bundle,
            experiment_names=["vision_only"],
            split_config=SplitConfig(n_splits=2, strategy="group_kfold", primary_head="emotion"),
            model_config=ModelConfig(hidden_dim=8, trunk_blocks=1, quality_dim=2, dropout=0.0),
            training_config=TrainingConfig(
                max_epochs=1,
                batch_size=2,
                classification_batch_size=2,
                device="cpu",
            ),
        )

        self.assertIn("oof_summary", result)
        self.assertIn("oof_reports", result)
        self.assertIn("support_report", result)
        self.assertIn("fold_diagnostics", result)
        self.assertFalse(result["oof_summary"].empty)
        self.assertFalse(result["support_report"].empty)
        self.assertIn("vision_only", result["oof_reports"])
        self.assertEqual(result["summary"].iloc[0]["experiment_name"], "vision_only")


if __name__ == "__main__":
    unittest.main()
