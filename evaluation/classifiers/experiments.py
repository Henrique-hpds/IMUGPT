from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency guard
    tqdm = None

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - depends on sklearn version
    StratifiedGroupKFold = None

from .metrics import (
    PRIMARY_HEADS,
    build_scored_class_ids,
    build_support_report,
    compute_domain_gap_summary,
    compute_multitask_metrics,
    suite_results_frame,
    summarize_unsupported_classes,
)
from .training import ModelConfig, TrainingConfig, evaluate_multitask_model, train_multitask_model

EXPERIMENT_SPECS: dict[str, dict[str, Any]] = {
    "vision_only": {
        "use_pose": True,
        "use_imu": False,
        "use_domain_head": False,
        "train_blocks": [("real", True)],
        "eval_domain": "real",
    },
    "imu_only_r2r": {
        "use_pose": False,
        "use_imu": True,
        "use_domain_head": False,
        "train_blocks": [("real", True)],
        "eval_domain": "real",
    },
    "imu_only_s2r": {
        "use_pose": False,
        "use_imu": True,
        "use_domain_head": True,
        "train_blocks": [("synthetic", True), ("real", False)],
        "eval_domain": "real",
    },
    "imu_only_mixed2r": {
        "use_pose": False,
        "use_imu": True,
        "use_domain_head": True,
        "train_blocks": [("real", True), ("synthetic", True)],
        "eval_domain": "real",
    },
    "vision_imu_r2r": {
        "use_pose": True,
        "use_imu": True,
        "use_domain_head": False,
        "train_blocks": [("real", True)],
        "eval_domain": "real",
    },
    "vision_imu_s2r": {
        "use_pose": True,
        "use_imu": True,
        "use_domain_head": True,
        "train_blocks": [("synthetic", True), ("real", False)],
        "eval_domain": "real",
    },
    "vision_imu_mixed2r": {
        "use_pose": True,
        "use_imu": True,
        "use_domain_head": True,
        "train_blocks": [("real", True), ("synthetic", True)],
        "eval_domain": "real",
    },
}


@dataclass(frozen=True)
class SplitConfig:
    n_splits: int = 5
    random_state: int = 42
    strategy: str = "group_kfold"
    group_column: str = "subject_group"
    stratify_column: str = "flat_tag_id"
    primary_head: str = "emotion"
    min_subject_groups_per_class: int = 2
    aggregate_mode: str = "oof"


def build_subject_group_splits(
    metadata: pd.DataFrame,
    *,
    config: SplitConfig | None = None,
) -> list[dict[str, Any]]:
    resolved_config = SplitConfig() if config is None else config
    metadata_frame = metadata.reset_index(drop=True).copy()
    groups = metadata_frame[resolved_config.group_column].astype(str).to_numpy()
    stratify_values = metadata_frame[resolved_config.stratify_column].astype(int).to_numpy()
    sample_indices = np.arange(metadata_frame.shape[0], dtype=np.int64)

    splits: list[dict[str, Any]] = []
    strategy = str(resolved_config.strategy).strip().lower()
    if strategy == "group_kfold":
        splitter = GroupKFold(n_splits=int(resolved_config.n_splits))
        split_iterator = splitter.split(sample_indices, groups=groups)
    elif strategy == "stratified_group_kfold" and StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(
            n_splits=int(resolved_config.n_splits),
            shuffle=True,
            random_state=int(resolved_config.random_state),
        )
        split_iterator = splitter.split(sample_indices, stratify_values, groups)
    else:
        raise ValueError(
            "Unsupported split strategy. Expected 'group_kfold' or 'stratified_group_kfold' with sklearn support."
        )

    for split_index, (train_indices, test_indices) in enumerate(split_iterator):
        splits.append(
            {
                "split_id": int(split_index),
                "train_indices": np.asarray(train_indices, dtype=np.int64),
                "test_indices": np.asarray(test_indices, dtype=np.int64),
            }
        )
    return splits


def _stack_blocks(blocks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(blocks) == 0:
        raise ValueError("At least one block is required.")

    return {
        "pose": np.concatenate([np.asarray(block["pose"], dtype=np.float32) for block in blocks], axis=0),
        "imu": np.concatenate([np.asarray(block["imu"], dtype=np.float32) for block in blocks], axis=0),
        "quality": np.concatenate([np.asarray(block["quality"], dtype=np.float32) for block in blocks], axis=0),
        "domain": np.concatenate([np.asarray(block["domain"], dtype=np.int64) for block in blocks], axis=0),
        "classification_mask": np.concatenate(
            [np.asarray(block["classification_mask"], dtype=np.float32) for block in blocks],
            axis=0,
        ),
        "targets": {
            head_name: np.concatenate(
                [np.asarray(block["targets"][head_name], dtype=np.int64) for block in blocks],
                axis=0,
            )
            for head_name in blocks[0]["targets"].keys()
        },
        "metadata": pd.concat([block["metadata"] for block in blocks], axis=0, ignore_index=True),
    }


def _base_block(
    dataset_bundle: Mapping[str, Any],
    *,
    indices: np.ndarray,
    domain_name: str,
    include_pose: bool,
    include_imu: bool,
    supervised: bool,
) -> dict[str, Any]:
    metadata = dataset_bundle["metadata"].iloc[np.asarray(indices, dtype=np.int64)].reset_index(drop=True).copy()
    pose_windows = np.asarray(dataset_bundle["pose_windows"], dtype=np.float32)[indices]
    imu_windows = (
        np.asarray(dataset_bundle["imu_real_windows"], dtype=np.float32)[indices]
        if str(domain_name) == "real"
        else np.asarray(dataset_bundle["imu_synthetic_windows"], dtype=np.float32)[indices]
    )
    zero_pose = np.zeros_like(pose_windows, dtype=np.float32)
    zero_imu = np.zeros_like(imu_windows, dtype=np.float32)
    domain_label = 0 if str(domain_name) == "real" else 1
    metadata["source_domain"] = str(domain_name)
    metadata["is_supervised"] = bool(supervised)

    return {
        "pose": pose_windows if include_pose else zero_pose,
        "imu": imu_windows if include_imu else zero_imu,
        "quality": np.asarray(dataset_bundle["quality_windows"], dtype=np.float32)[indices],
        "domain": np.full(len(indices), domain_label, dtype=np.int64),
        "classification_mask": np.full(len(indices), 1.0 if supervised else 0.0, dtype=np.float32),
        "targets": {
            head_name: metadata[f"{head_name}_id"].to_numpy(dtype=np.int64)
            for head_name in ("emotion", "modality", "stimulus", "flat_tag")
        },
        "metadata": metadata,
    }


def build_experiment_arrays(
    dataset_bundle: Mapping[str, Any],
    *,
    experiment_name: str,
    train_indices: np.ndarray,
    val_indices: np.ndarray | None = None,
    eval_indices: np.ndarray,
) -> dict[str, Any]:
    if experiment_name not in EXPERIMENT_SPECS:
        raise ValueError(f"Unsupported experiment_name: {experiment_name}")

    spec = EXPERIMENT_SPECS[experiment_name]

    def _stack_train_blocks(indices: np.ndarray) -> dict[str, Any]:
        blocks = [
            _base_block(
                dataset_bundle,
                indices=np.asarray(indices, dtype=np.int64),
                domain_name=str(domain_name),
                include_pose=bool(spec["use_pose"]),
                include_imu=bool(spec["use_imu"]),
                supervised=bool(supervised),
            )
            for domain_name, supervised in spec["train_blocks"]
        ]
        return _stack_blocks(blocks)

    train_block = _stack_train_blocks(np.asarray(train_indices, dtype=np.int64))
    resolved_val_indices = np.asarray(eval_indices if val_indices is None else val_indices, dtype=np.int64)
    val_block = _stack_train_blocks(resolved_val_indices)
    eval_block = _base_block(
        dataset_bundle,
        indices=np.asarray(eval_indices, dtype=np.int64),
        domain_name=str(spec["eval_domain"]),
        include_pose=bool(spec["use_pose"]),
        include_imu=bool(spec["use_imu"]),
        supervised=True,
    )
    return {
        "train": train_block,
        "val": val_block,
        "eval": eval_block,
        "spec": spec,
    }


def _train_val_split(
    metadata: pd.DataFrame,
    *,
    random_state: int,
    group_column: str = "capture_id",
) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = metadata[group_column].astype(str).nunique()
    if unique_groups <= 1:
        indices = np.arange(metadata.shape[0], dtype=np.int64)
        return indices, indices

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=int(random_state))
    sample_indices = np.arange(metadata.shape[0], dtype=np.int64)
    groups = metadata[group_column].astype(str).to_numpy()
    train_indices, val_indices = next(splitter.split(sample_indices, groups=groups))
    return np.asarray(train_indices, dtype=np.int64), np.asarray(val_indices, dtype=np.int64)


def _slice_arrays(arrays: Mapping[str, Any], indices: np.ndarray) -> dict[str, Any]:
    index_array = np.asarray(indices, dtype=np.int64)
    return {
        "pose": np.asarray(arrays["pose"], dtype=np.float32)[index_array],
        "imu": np.asarray(arrays["imu"], dtype=np.float32)[index_array],
        "quality": np.asarray(arrays["quality"], dtype=np.float32)[index_array],
        "domain": np.asarray(arrays["domain"], dtype=np.int64)[index_array],
        "classification_mask": np.asarray(arrays["classification_mask"], dtype=np.float32)[index_array],
        "targets": {
            head_name: np.asarray(values, dtype=np.int64)[index_array]
            for head_name, values in dict(arrays["targets"]).items()
        },
        "metadata": arrays["metadata"].iloc[index_array].reset_index(drop=True),
    }


def run_single_experiment(
    dataset_bundle: Mapping[str, Any],
    *,
    experiment_name: str,
    split: Mapping[str, Any],
    model_config: ModelConfig | None = None,
    training_config: TrainingConfig | None = None,
    scored_class_ids: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    resolved_model_config = ModelConfig() if model_config is None else model_config
    resolved_training_config = TrainingConfig() if training_config is None else training_config
    outer_train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    inner_train_relative_indices, inner_val_relative_indices = _train_val_split(
        dataset_bundle["metadata"].iloc[outer_train_indices].reset_index(drop=True),
        random_state=int(split.get("split_id", 0)) + 17,
    )
    inner_train_indices = outer_train_indices[np.asarray(inner_train_relative_indices, dtype=np.int64)]
    inner_val_indices = outer_train_indices[np.asarray(inner_val_relative_indices, dtype=np.int64)]
    prepared = build_experiment_arrays(
        dataset_bundle,
        experiment_name=experiment_name,
        train_indices=inner_train_indices,
        val_indices=inner_val_indices,
        eval_indices=np.asarray(split["test_indices"], dtype=np.int64),
    )

    training_result = train_multitask_model(
        train_arrays=prepared["train"],
        val_arrays=prepared["val"],
        label_encoders=dataset_bundle["label_encoders"],
        model_config=resolved_model_config,
        training_config=resolved_training_config,
        use_pose_branch=bool(prepared["spec"]["use_pose"]),
        use_imu_branch=bool(prepared["spec"]["use_imu"]),
        use_domain_head=bool(prepared["spec"]["use_domain_head"]),
        scored_class_ids=scored_class_ids,
    )
    test_report = evaluate_multitask_model(
        training_result["model"],
        prepared["eval"],
        dataset_bundle["label_encoders"],
        batch_size=int(resolved_training_config.batch_size),
        device=training_result["model"].emotion_head[0].weight.device,
        scored_class_ids=scored_class_ids,
        filter_unsupervised=True,
    )
    return {
        "experiment_name": str(experiment_name),
        "split_id": int(split.get("split_id", 0)),
        "spec": dict(prepared["spec"]),
        "metrics": dict(test_report["metrics"]),
        "history": list(training_result["history"]),
        "train_metrics": dict(training_result["train_report"]["metrics"]),
        "val_metrics": dict(training_result["val_report"]["metrics"]),
        "test_predictions": test_report,
    }


def _aggregate_oof_report(
    experiment_results: Sequence[Mapping[str, Any]],
    *,
    label_encoders: Mapping[str, Mapping[str, Any]],
    scored_class_ids: Mapping[str, Sequence[int]] | None,
) -> dict[str, Any]:
    if len(experiment_results) == 0:
        return {
            "targets": {},
            "predictions": {},
            "probabilities": {},
            "metadata": pd.DataFrame(),
            "metrics": compute_multitask_metrics(
                y_true={},
                y_pred={},
                probabilities=None,
                label_encoders=label_encoders,
                scored_class_ids=scored_class_ids,
            ),
        }

    targets: dict[str, list[np.ndarray]] = {}
    predictions: dict[str, list[np.ndarray]] = {}
    probabilities: dict[str, list[np.ndarray]] = {}
    metadata_blocks: list[pd.DataFrame] = []

    for result in experiment_results:
        test_predictions = dict(result["test_predictions"])
        for head_name, values in dict(test_predictions.get("targets", {})).items():
            targets.setdefault(head_name, []).append(np.asarray(values, dtype=np.int64))
        for head_name, values in dict(test_predictions.get("predictions", {})).items():
            predictions.setdefault(head_name, []).append(np.asarray(values, dtype=np.int64))
        for head_name, values in dict(test_predictions.get("probabilities", {})).items():
            probabilities.setdefault(head_name, []).append(np.asarray(values, dtype=np.float32))
        metadata = test_predictions.get("metadata")
        if metadata is not None:
            annotated_metadata = metadata.copy()
            annotated_metadata["split_id"] = int(result["split_id"])
            metadata_blocks.append(annotated_metadata)

    merged_targets = {
        head_name: np.concatenate(blocks, axis=0).astype(np.int64)
        for head_name, blocks in targets.items()
        if len(blocks) > 0
    }
    merged_predictions = {
        head_name: np.concatenate(blocks, axis=0).astype(np.int64)
        for head_name, blocks in predictions.items()
        if len(blocks) > 0
    }
    merged_probabilities = {
        head_name: np.concatenate(blocks, axis=0).astype(np.float32)
        for head_name, blocks in probabilities.items()
        if len(blocks) > 0
    }
    metrics = compute_multitask_metrics(
        y_true=merged_targets,
        y_pred=merged_predictions,
        probabilities=merged_probabilities,
        label_encoders=label_encoders,
        scored_class_ids=scored_class_ids,
    )
    return {
        "targets": merged_targets,
        "predictions": merged_predictions,
        "probabilities": merged_probabilities,
        "metadata": pd.concat(metadata_blocks, axis=0, ignore_index=True) if metadata_blocks else pd.DataFrame(),
        "metrics": metrics,
    }


def _build_oof_summary(
    oof_reports: Mapping[str, Mapping[str, Any]],
    *,
    support_report: pd.DataFrame,
    primary_head: str,
) -> pd.DataFrame:
    rows = []
    unsupported_classes = summarize_unsupported_classes(support_report, head_names=PRIMARY_HEADS)

    for experiment_name, oof_report in oof_reports.items():
        per_head = dict(oof_report["metrics"].get("per_head", {}))
        primary_head_metrics = per_head.get(primary_head, {})
        rows.append(
            {
                "experiment_name": str(experiment_name),
                "global_score_macro_f1_mean": oof_report["metrics"].get("global_score_macro_f1_mean"),
                "global_score_weighted_macro_f1": oof_report["metrics"].get("global_score_weighted_macro_f1"),
                "global_score_macro_f1_mean_all": oof_report["metrics"].get("global_score_macro_f1_mean_all"),
                "global_score_weighted_macro_f1_all": oof_report["metrics"].get("global_score_weighted_macro_f1_all"),
                "emotion_macro_f1_supported_oof": (
                    None if "emotion" not in per_head else per_head["emotion"]["supported_macro_f1"]
                ),
                "emotion_macro_f1_all_oof": (
                    None if "emotion" not in per_head else per_head["emotion"]["macro_f1"]
                ),
                "modality_macro_f1_oof": (
                    None if "modality" not in per_head else per_head["modality"]["supported_macro_f1"]
                ),
                "stimulus_macro_f1_oof": (
                    None if "stimulus" not in per_head else per_head["stimulus"]["supported_macro_f1"]
                ),
                "emotion_macro_f1": primary_head_metrics.get("supported_macro_f1") if primary_head == "emotion" else (
                    None if "emotion" not in per_head else per_head["emotion"]["supported_macro_f1"]
                ),
                "modality_macro_f1": None if "modality" not in per_head else per_head["modality"]["supported_macro_f1"],
                "stimulus_macro_f1": None if "stimulus" not in per_head else per_head["stimulus"]["supported_macro_f1"],
                "unsupported_classes": unsupported_classes,
                "num_oof_samples": 0 if oof_report["metadata"].empty else int(len(oof_report["metadata"])),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    ranking_column = f"{primary_head}_macro_f1_supported_oof" if f"{primary_head}_macro_f1_supported_oof" in summary.columns else "global_score_macro_f1_mean"
    return summary.sort_values(ranking_column, ascending=False, kind="stable").reset_index(drop=True)


def run_experiment_suite(
    dataset_bundle: Mapping[str, Any],
    *,
    experiment_names: Sequence[str] | None = None,
    split_config: SplitConfig | None = None,
    model_config: ModelConfig | None = None,
    training_config: TrainingConfig | None = None,
) -> dict[str, Any]:
    selected_experiments = list(EXPERIMENT_SPECS.keys()) if experiment_names is None else [str(name) for name in experiment_names]
    resolved_split_config = SplitConfig() if split_config is None else split_config
    resolved_training_config = TrainingConfig() if training_config is None else training_config
    splits = build_subject_group_splits(dataset_bundle["metadata"], config=resolved_split_config)
    support_report = build_support_report(
        dataset_bundle["metadata"],
        dataset_bundle["label_encoders"],
        group_column=str(resolved_split_config.group_column),
        min_subject_groups=int(resolved_split_config.min_subject_groups_per_class),
    )
    scored_class_ids = build_scored_class_ids(support_report, head_names=PRIMARY_HEADS)
    results = []
    total_runs = int(len(selected_experiments) * len(splits))
    suite_progress = None
    if bool(resolved_training_config.show_progress) and tqdm is not None:
        suite_progress = tqdm(total=total_runs, desc="Experiment suite", unit="fold")

    try:
        for experiment_name in selected_experiments:
            for split in splits:
                if suite_progress is not None:
                    current_fold = int(split.get("split_id", 0)) + 1
                    suite_progress.set_postfix(experiment=experiment_name, fold=f"{current_fold}/{len(splits)}")
                results.append(
                    run_single_experiment(
                        dataset_bundle,
                        experiment_name=experiment_name,
                        split=split,
                        model_config=model_config,
                        training_config=resolved_training_config,
                        scored_class_ids=scored_class_ids,
                    )
                )
                if suite_progress is not None:
                    suite_progress.update(1)
    finally:
        if suite_progress is not None:
            suite_progress.close()

    results_frame = suite_results_frame(results)
    oof_reports = {
        experiment_name: _aggregate_oof_report(
            [result for result in results if str(result.get("experiment_name")) == str(experiment_name)],
            label_encoders=dataset_bundle["label_encoders"],
            scored_class_ids=scored_class_ids,
        )
        for experiment_name in selected_experiments
    }
    oof_summary = _build_oof_summary(
        oof_reports,
        support_report=support_report,
        primary_head=str(resolved_split_config.primary_head),
    )
    aggregate_mode = str(resolved_split_config.aggregate_mode).strip().lower()
    if aggregate_mode == "oof":
        summary = oof_summary.copy()
    elif aggregate_mode == "mean_fold":
        summary = (
            results_frame.groupby("experiment_name", as_index=False)
            .agg(
                global_score_macro_f1_mean=("global_score_macro_f1_mean", "mean"),
                global_score_weighted_macro_f1=("global_score_weighted_macro_f1", "mean"),
                global_score_macro_f1_mean_all=("global_score_macro_f1_mean_all", "mean"),
                global_score_weighted_macro_f1_all=("global_score_weighted_macro_f1_all", "mean"),
                emotion_macro_f1=("emotion_macro_f1", "mean"),
                emotion_macro_f1_all=("emotion_macro_f1_all", "mean"),
                modality_macro_f1=("modality_macro_f1", "mean"),
                modality_macro_f1_all=("modality_macro_f1_all", "mean"),
                stimulus_macro_f1=("stimulus_macro_f1", "mean"),
                stimulus_macro_f1_all=("stimulus_macro_f1_all", "mean"),
            )
            .sort_values("emotion_macro_f1", ascending=False, kind="stable")
            .reset_index(drop=True)
        )
    else:
        raise ValueError("Unsupported aggregate_mode. Expected 'oof' or 'mean_fold'.")
    return {
        "results": results,
        "results_frame": results_frame,
        "fold_diagnostics": results_frame,
        "summary": summary,
        "oof_summary": oof_summary,
        "oof_reports": oof_reports,
        "support_report": support_report,
        "domain_gap_summary": compute_domain_gap_summary(oof_summary),
        "splits": splits,
    }
