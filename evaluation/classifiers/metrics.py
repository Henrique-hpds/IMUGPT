from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PRIMARY_HEADS = ("emotion", "modality", "stimulus")


def _resolve_scored_labels(
    num_classes: int,
    scored_class_ids: Sequence[int] | None,
) -> np.ndarray:
    if num_classes <= 0:
        return np.zeros(0, dtype=np.int64)

    if scored_class_ids is None:
        return np.arange(num_classes, dtype=np.int64)

    labels = np.asarray(list(scored_class_ids), dtype=np.int64)
    labels = labels[(labels >= 0) & (labels < num_classes)]
    labels = np.unique(labels)
    if labels.size == 0:
        return np.arange(num_classes, dtype=np.int64)
    return labels


def _balanced_accuracy_from_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: np.ndarray,
) -> float:
    confusion = confusion_matrix(y_true, y_pred, labels=labels).astype(np.float64)
    supports = confusion.sum(axis=1)
    recalls = np.divide(
        np.diag(confusion),
        supports,
        out=np.zeros_like(supports, dtype=np.float64),
        where=supports > 0,
    )
    return float(recalls.mean()) if recalls.size > 0 else 0.0


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    num_bins: int = 15,
) -> float | None:
    truth = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] != truth.shape[0] or probs.shape[1] == 0:
        return None

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correctness = (predictions == truth).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, int(num_bins) + 1)

    ece = 0.0
    for bin_index in range(int(num_bins)):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]
        if bin_index == int(num_bins) - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if int(np.count_nonzero(mask)) == 0:
            continue
        mean_confidence = float(np.mean(confidences[mask]))
        mean_accuracy = float(np.mean(correctness[mask]))
        ece += abs(mean_confidence - mean_accuracy) * (int(np.count_nonzero(mask)) / max(1, truth.shape[0]))
    return float(ece)


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    truth = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] != truth.shape[0] or probs.shape[1] == 0:
        return None
    if probs.shape[1] == 2:
        return float(brier_score_loss(truth, probs[:, 1]))

    one_hot = np.zeros_like(probs, dtype=np.float64)
    one_hot[np.arange(truth.shape[0]), truth] = 1.0
    return float(np.mean(np.sum(np.square(probs - one_hot), axis=1)))


def compute_head_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    probabilities: np.ndarray | None,
    class_names: Sequence[str],
    scored_class_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    labels = np.arange(len(class_names), dtype=np.int64)
    supported_labels = _resolve_scored_labels(len(class_names), scored_class_ids)

    if truth.size == 0:
        zero_confusion = np.zeros((len(class_names), len(class_names)), dtype=int).tolist()
        return {
            "macro_f1": 0.0,
            "balanced_accuracy": 0.0,
            "weighted_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "per_class_f1": {str(class_name): 0.0 for class_name in class_names},
            "confusion_matrix": zero_confusion,
            "ece": None,
            "brier_score": None,
            "auroc_ovr": None,
            "auprc_ovr": None,
            "supported_class_ids": supported_labels.astype(int).tolist(),
            "supported_class_names": [str(class_names[class_id]) for class_id in supported_labels.tolist()],
            "supported_macro_f1": 0.0,
            "supported_balanced_accuracy": 0.0,
            "supported_weighted_f1": 0.0,
            "supported_macro_precision": 0.0,
            "supported_macro_recall": 0.0,
        }

    per_class_f1_values = f1_score(truth, pred, labels=labels, average=None, zero_division=0)
    output = {
        "macro_f1": float(f1_score(truth, pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": _balanced_accuracy_from_labels(truth, pred, labels=labels),
        "weighted_f1": float(f1_score(truth, pred, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(truth, pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, pred, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": {
            str(class_name): float(per_class_f1_values[class_index])
            for class_index, class_name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(truth, pred, labels=labels).astype(int).tolist(),
        "ece": None,
        "brier_score": None,
        "auroc_ovr": None,
        "auprc_ovr": None,
        "supported_class_ids": supported_labels.astype(int).tolist(),
        "supported_class_names": [str(class_names[class_id]) for class_id in supported_labels.tolist()],
        "supported_macro_f1": float(
            f1_score(truth, pred, labels=supported_labels, average="macro", zero_division=0)
        ),
        "supported_balanced_accuracy": _balanced_accuracy_from_labels(truth, pred, labels=supported_labels),
        "supported_weighted_f1": float(
            f1_score(truth, pred, labels=supported_labels, average="weighted", zero_division=0)
        ),
        "supported_macro_precision": float(
            precision_score(truth, pred, labels=supported_labels, average="macro", zero_division=0)
        ),
        "supported_macro_recall": float(
            recall_score(truth, pred, labels=supported_labels, average="macro", zero_division=0)
        ),
    }
    if probabilities is None:
        return output

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] != truth.shape[0] or probs.shape[1] != len(class_names):
        return output

    output["ece"] = expected_calibration_error(truth, probs)
    output["brier_score"] = multiclass_brier_score(truth, probs)

    if np.unique(truth).size < 2:
        return output

    try:
        if len(class_names) == 2:
            output["auroc_ovr"] = float(roc_auc_score(truth, probs[:, 1]))
            output["auprc_ovr"] = float(average_precision_score(truth, probs[:, 1]))
        else:
            output["auroc_ovr"] = float(roc_auc_score(truth, probs, multi_class="ovr", average="macro"))
            output["auprc_ovr"] = float(average_precision_score(np.eye(len(class_names))[truth], probs, average="macro"))
    except ValueError:
        output["auroc_ovr"] = None
        output["auprc_ovr"] = None
    return output


def compute_multitask_metrics(
    *,
    y_true: Mapping[str, np.ndarray],
    y_pred: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray] | None,
    label_encoders: Mapping[str, Mapping[str, Any]],
    scored_class_ids: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    head_metrics = {}
    for head_name in label_encoders.keys():
        if head_name not in y_true or head_name not in y_pred:
            continue
        head_probabilities = None if probabilities is None else probabilities.get(head_name)
        head_metrics[head_name] = compute_head_metrics(
            y_true=np.asarray(y_true[head_name]),
            y_pred=np.asarray(y_pred[head_name]),
            probabilities=head_probabilities,
            class_names=label_encoders[head_name]["class_names"],
            scored_class_ids=(None if scored_class_ids is None else scored_class_ids.get(head_name)),
        )

    primary_scores_supported = [
        float(head_metrics[head_name]["supported_macro_f1"])
        for head_name in PRIMARY_HEADS
        if head_name in head_metrics
    ]
    primary_scores_all = [
        float(head_metrics[head_name]["macro_f1"])
        for head_name in PRIMARY_HEADS
        if head_name in head_metrics
    ]
    weighted_primary_score = None
    weighted_primary_score_all = None
    if all(head_name in head_metrics for head_name in PRIMARY_HEADS):
        weighted_primary_score = (
            0.5 * float(head_metrics["emotion"]["supported_macro_f1"])
            + 0.25 * float(head_metrics["modality"]["supported_macro_f1"])
            + 0.25 * float(head_metrics["stimulus"]["supported_macro_f1"])
        )
        weighted_primary_score_all = (
            0.5 * float(head_metrics["emotion"]["macro_f1"])
            + 0.25 * float(head_metrics["modality"]["macro_f1"])
            + 0.25 * float(head_metrics["stimulus"]["macro_f1"])
        )

    return {
        "per_head": head_metrics,
        "global_score_macro_f1_mean": None if len(primary_scores_supported) == 0 else float(np.mean(primary_scores_supported)),
        "global_score_weighted_macro_f1": weighted_primary_score,
        "global_score_macro_f1_mean_all": None if len(primary_scores_all) == 0 else float(np.mean(primary_scores_all)),
        "global_score_weighted_macro_f1_all": weighted_primary_score_all,
    }


def build_support_report(
    metadata: pd.DataFrame,
    label_encoders: Mapping[str, Mapping[str, Any]],
    *,
    group_column: str = "subject_group",
    min_subject_groups: int = 2,
) -> pd.DataFrame:
    rows = []
    for head_name, encoder in label_encoders.items():
        label_column = f"{head_name}_id"
        if label_column not in metadata.columns:
            continue
        frame = metadata[[label_column, group_column]].copy()
        sample_counts = frame[label_column].value_counts().to_dict()
        group_counts = frame.groupby(label_column)[group_column].nunique().to_dict()
        for class_id, class_name in enumerate(encoder["class_names"]):
            num_samples = int(sample_counts.get(class_id, 0))
            num_subject_groups = int(group_counts.get(class_id, 0))
            is_supported = bool(num_subject_groups >= int(min_subject_groups))
            rows.append(
                {
                    "head_name": str(head_name),
                    "class_id": int(class_id),
                    "class_name": str(class_name),
                    "num_samples": num_samples,
                    "num_subject_groups": num_subject_groups,
                    "min_subject_groups_required": int(min_subject_groups),
                    "is_supported": is_supported,
                    "support_status": "supported" if is_supported else "unsupported",
                }
            )
    return pd.DataFrame(rows)


def build_scored_class_ids(
    support_report: pd.DataFrame,
    *,
    head_names: Sequence[str] | None = None,
) -> dict[str, list[int]]:
    if support_report.empty:
        return {}

    selected_heads = set(PRIMARY_HEADS if head_names is None else [str(head_name) for head_name in head_names])
    scored_class_ids: dict[str, list[int]] = {}
    for head_name in sorted(selected_heads):
        head_rows = support_report.loc[support_report["head_name"] == head_name]
        supported_ids = head_rows.loc[head_rows["is_supported"], "class_id"].astype(int).tolist()
        if supported_ids:
            scored_class_ids[head_name] = supported_ids
    return scored_class_ids


def summarize_unsupported_classes(
    support_report: pd.DataFrame,
    *,
    head_names: Sequence[str] | None = None,
) -> str:
    if support_report.empty:
        return ""

    selected_heads = PRIMARY_HEADS if head_names is None else tuple(str(head_name) for head_name in head_names)
    chunks = []
    for head_name in selected_heads:
        unsupported = (
            support_report.loc[
                (support_report["head_name"] == head_name) & (~support_report["is_supported"]),
                "class_name",
            ]
            .astype(str)
            .tolist()
        )
        if unsupported:
            chunks.append(f"{head_name}: {', '.join(unsupported)}")
    return "; ".join(chunks)


def suite_results_frame(experiment_results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in experiment_results:
        metrics = dict(result.get("metrics", {}))
        per_head = dict(metrics.get("per_head", {}))
        rows.append(
            {
                "experiment_name": result.get("experiment_name"),
                "split_id": result.get("split_id"),
                "global_score_macro_f1_mean": metrics.get("global_score_macro_f1_mean"),
                "global_score_weighted_macro_f1": metrics.get("global_score_weighted_macro_f1"),
                "global_score_macro_f1_mean_all": metrics.get("global_score_macro_f1_mean_all"),
                "global_score_weighted_macro_f1_all": metrics.get("global_score_weighted_macro_f1_all"),
                "emotion_macro_f1": None if "emotion" not in per_head else per_head["emotion"]["supported_macro_f1"],
                "emotion_macro_f1_all": None if "emotion" not in per_head else per_head["emotion"]["macro_f1"],
                "modality_macro_f1": None if "modality" not in per_head else per_head["modality"]["supported_macro_f1"],
                "modality_macro_f1_all": None if "modality" not in per_head else per_head["modality"]["macro_f1"],
                "stimulus_macro_f1": None if "stimulus" not in per_head else per_head["stimulus"]["supported_macro_f1"],
                "stimulus_macro_f1_all": None if "stimulus" not in per_head else per_head["stimulus"]["macro_f1"],
            }
        )
    return pd.DataFrame(rows)


def compute_domain_gap_summary(experiment_results: Sequence[Mapping[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    frame = experiment_results.copy() if isinstance(experiment_results, pd.DataFrame) else suite_results_frame(experiment_results)
    if frame.empty:
        return frame

    if "experiment_name" not in frame.columns:
        raise ValueError("compute_domain_gap_summary expects an experiment summary with an 'experiment_name' column.")

    if "global_score_macro_f1_mean" not in frame.columns:
        raise ValueError("compute_domain_gap_summary expects a 'global_score_macro_f1_mean' column.")

    score_by_experiment = {
        str(row["experiment_name"]): float(row["global_score_macro_f1_mean"])
        for _, row in frame.iterrows()
        if pd.notna(row["global_score_macro_f1_mean"])
    }
    rows = []
    for family in ("imu_only", "vision_imu"):
        r2r_key = f"{family}_r2r"
        s2r_key = f"{family}_s2r"
        mixed_key = f"{family}_mixed2r"
        if r2r_key in score_by_experiment and s2r_key in score_by_experiment:
            rows.append(
                {
                    "metric": f"{family}_gap_s2r",
                    "value": score_by_experiment[r2r_key] - score_by_experiment[s2r_key],
                }
            )
        if r2r_key in score_by_experiment and mixed_key in score_by_experiment:
            rows.append(
                {
                    "metric": f"{family}_gap_mixed2r",
                    "value": score_by_experiment[r2r_key] - score_by_experiment[mixed_key],
                }
            )
            rows.append(
                {
                    "metric": f"{family}_gain_mixed_over_r2r",
                    "value": score_by_experiment[mixed_key] - score_by_experiment[r2r_key],
                }
            )
        if s2r_key in score_by_experiment and mixed_key in score_by_experiment:
            rows.append(
                {
                    "metric": f"{family}_gain_mixed_over_s2r",
                    "value": score_by_experiment[mixed_key] - score_by_experiment[s2r_key],
                }
            )
    return pd.DataFrame(rows)


def plot_confusion_matrices(
    metrics: Mapping[str, Any],
    label_encoders: Mapping[str, Mapping[str, Any]],
) -> tuple[plt.Figure, np.ndarray]:
    per_head = dict(metrics.get("per_head", {}))
    available_heads = [head_name for head_name in PRIMARY_HEADS if head_name in per_head]
    if len(available_heads) == 0:
        raise ValueError("There are no primary heads available to plot.")

    figure, axes = plt.subplots(1, len(available_heads), figsize=(5.2 * len(available_heads), 4.6), squeeze=False)
    for axis, head_name in zip(axes[0], available_heads):
        head_metrics = per_head[head_name]
        confusion = np.asarray(head_metrics["confusion_matrix"], dtype=np.float32)
        axis.imshow(confusion, cmap="Blues")
        axis.set_title(f"{head_name} confusion")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        class_names = label_encoders[head_name]["class_names"]
        axis.set_xticks(np.arange(len(class_names)))
        axis.set_yticks(np.arange(len(class_names)))
        axis.set_xticklabels(class_names, rotation=45, ha="right")
        axis.set_yticklabels(class_names)
    figure.tight_layout()
    return figure, axes
