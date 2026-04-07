from __future__ import annotations

from importlib import import_module

_EXPORT_MAP = {
    "ALL_CAPTURE_BLACKLIST": (".data", "ALL_CAPTURE_BLACKLIST"),
    "EXPERIMENT_SPECS": (".experiments", "EXPERIMENT_SPECS"),
    "ModelConfig": (".training", "ModelConfig"),
    "SplitConfig": (".experiments", "SplitConfig"),
    "TrainingConfig": (".training", "TrainingConfig"),
    "WindowedDatasetConfig": (".data", "WindowedDatasetConfig"),
    "align_target_to_reference": (".alignment", "align_target_to_reference"),
    "apply_capture_blacklist": (".data", "apply_capture_blacklist"),
    "build_classifier_capture_table": (".data", "build_classifier_capture_table"),
    "build_subject_group_splits": (".experiments", "build_subject_group_splits"),
    "build_windowed_multimodal_dataset": (".data", "build_windowed_multimodal_dataset"),
    "compute_domain_gap_summary": (".metrics", "compute_domain_gap_summary"),
    "compute_multitask_metrics": (".metrics", "compute_multitask_metrics"),
    "estimate_lag_cross_correlation": (".alignment", "estimate_lag_cross_correlation"),
    "load_capture_modalities": (".data", "load_capture_modalities"),
    "normalize_capture_blacklist": (".data", "normalize_capture_blacklist"),
    "plot_confusion_matrices": (".metrics", "plot_confusion_matrices"),
    "prepare_capture_windows": (".data", "prepare_capture_windows"),
    "resample_values_to_reference": (".alignment", "resample_values_to_reference"),
    "run_experiment_suite": (".experiments", "run_experiment_suite"),
    "run_single_experiment": (".experiments", "run_single_experiment"),
}

__all__ = sorted(_EXPORT_MAP)


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORT_MAP[name]
    module = import_module(module_name, __name__)
    return getattr(module, attribute_name)
