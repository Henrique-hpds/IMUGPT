from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import pearsonr
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - depends on sklearn version
    StratifiedGroupKFold = None

from evaluation.tsne import (
    extract_selected_modalities,
    load_capture_pair_from_row,
    resample_real_to_synthetic_rate,
    segment_signal_windows,
)
from evaluation.utils import build_exported_capture_table, configure_capture_table_display, find_project_root


CLASS_COLUMNS = ("emotion", "modality", "stimulus")
DISTANCE_METRIC_COLUMNS = ("cfid_mean", "js_mean", "mmd_mean", "dtw_rtr", "dtw_rts", "dtw_sts")


def _normalize_label_value(value: Any) -> str:
    if pd.isna(value):
        return "None"
    return str(value)


def _capture_id_from_row(row: pd.Series | dict[str, Any]) -> str:
    series = pd.Series(row)
    take_id = series.get("take_id")
    if pd.isna(take_id) or take_id in (None, ""):
        return str(series["clip_id"])
    return f"{series['clip_id']}::{take_id}"


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return None
    if isinstance(value, (int, np.integer, float, np.floating)):
        return float(value)
    return None


def _string_array(values: Sequence[str]) -> np.ndarray:
    normalized = [str(value) for value in values]
    max_length = max((len(value) for value in normalized), default=1)
    return np.asarray(normalized, dtype=f"<U{max_length}")


def _sample_indices(indices: np.ndarray, max_samples: int | None, rng: np.random.Generator) -> np.ndarray:
    if max_samples is None or len(indices) <= int(max_samples):
        return np.asarray(indices, dtype=np.int64)
    chosen = np.sort(rng.choice(indices, size=int(max_samples), replace=False))
    return np.asarray(chosen, dtype=np.int64)


def _balanced_sample_indices_by_label(
    labels: Sequence[str],
    indices: np.ndarray,
    rng: np.random.Generator,
    max_total_samples: int | None = None,
) -> np.ndarray:
    label_frame = pd.DataFrame({"label": [str(label) for label in labels], "index": np.asarray(indices, dtype=np.int64)})
    counts = label_frame["label"].value_counts()
    if counts.empty:
        return np.asarray([], dtype=np.int64)

    target_per_label = int(counts.min())
    if max_total_samples is not None:
        target_per_label = min(target_per_label, max(1, int(max_total_samples) // max(1, counts.size)))

    sampled_indices: list[np.ndarray] = []
    for label in sorted(counts.index):
        label_indices = label_frame.loc[label_frame["label"] == label, "index"].to_numpy(dtype=np.int64)
        if label_indices.size <= target_per_label:
            sampled_indices.append(np.sort(label_indices))
            continue
        chosen = np.sort(rng.choice(label_indices, size=target_per_label, replace=False))
        sampled_indices.append(chosen)

    if not sampled_indices:
        return np.asarray([], dtype=np.int64)
    return np.sort(np.concatenate(sampled_indices).astype(np.int64))


def _flatten_windows(windows_4d: np.ndarray) -> np.ndarray:
    block = np.asarray(windows_4d, dtype=np.float32)
    if block.ndim != 4:
        raise ValueError("windows_4d must have shape [n_windows, time, sensors, channels].")
    return block.reshape(block.shape[0], block.shape[1], -1).astype(np.float32)


def load_protocol_capture_table(output_root: Path | str | None = None) -> pd.DataFrame:
    configure_capture_table_display()
    if output_root is None:
        project_root = find_project_root()
        output_root = project_root / "output" / "robot_emotions_virtual_imu_v2_all_dataset"
    return build_exported_capture_table(output_root)


def get_shared_sensor_names(
    captures_df: pd.DataFrame,
    *,
    synthetic_filename: str = "virtual_imu.npz",
) -> list[str]:
    if captures_df.empty:
        raise ValueError("captures_df is empty.")

    shared_sensors: list[str] | None = None
    for _, row in captures_df.reset_index(drop=True).iterrows():
        capture_pair = load_capture_pair_from_row(row, synthetic_filename=synthetic_filename)
        capture_shared = [
            sensor_name
            for sensor_name in capture_pair["real"]["sensor_names"]
            if sensor_name in set(capture_pair["synthetic"]["sensor_names"])
        ]
        if shared_sensors is None:
            shared_sensors = capture_shared
            continue
        shared_sensors = [sensor_name for sensor_name in shared_sensors if sensor_name in set(capture_shared)]

    if not shared_sensors:
        raise ValueError("There are no sensors shared by every real/synthetic capture pair.")
    return shared_sensors


def build_paired_window_dataset(
    captures_df: pd.DataFrame,
    *,
    signal_groups: Sequence[str] = ("acc",),
    selected_sensors: Sequence[str] | None = None,
    selected_axes: Sequence[str] = ("x", "y", "z"),
    resample_method: str = "resample_poly",
    window_type: str = "seconds",
    window_size: int | None = None,
    window_duration_sec: float | None = 3.0,
    stride_or_overlap_mode: str = "overlap",
    stride: int | None = None,
    stride_sec: float | None = None,
    overlap: float = 0.5,
    max_windows_per_capture: int | None = None,
    random_state: int = 42,
    synthetic_filename: str = "virtual_imu.npz",
) -> dict[str, Any]:
    if captures_df.empty:
        raise ValueError("captures_df is empty.")

    rng = np.random.default_rng(int(random_state))
    captures = captures_df.reset_index(drop=True).copy()
    selected_sensors_resolved = (
        [str(sensor_name) for sensor_name in selected_sensors]
        if selected_sensors is not None
        else get_shared_sensor_names(captures, synthetic_filename=synthetic_filename)
    )

    real_windows_blocks: list[np.ndarray] = []
    synthetic_windows_blocks: list[np.ndarray] = []
    pair_rows: list[dict[str, Any]] = []
    capture_summary_rows: list[dict[str, Any]] = []
    channel_labels: list[str] | None = None

    for capture_index, (_, row) in enumerate(captures.iterrows()):
        capture_pair = load_capture_pair_from_row(row, synthetic_filename=synthetic_filename)
        signal_bundle = extract_selected_modalities(
            capture_pair,
            signal_groups=signal_groups,
            selected_sensors=selected_sensors_resolved,
            selected_axes=selected_axes,
        )
        aligned_bundle = resample_real_to_synthetic_rate(
            real_timestamps_sec=signal_bundle["real_timestamps_sec"],
            real_values=signal_bundle["real_values"],
            synthetic_timestamps_sec=signal_bundle["synthetic_timestamps_sec"],
            synthetic_values=signal_bundle["synthetic_values"],
            method=resample_method,
        )

        real_segment = segment_signal_windows(
            aligned_bundle["real_resampled_values"],
            window_type=window_type,
            window_size=window_size,
            window_duration_sec=window_duration_sec,
            sampling_frequency_hz=aligned_bundle["synthetic_frequency_hz"],
            stride_or_overlap_mode=stride_or_overlap_mode,
            stride=stride,
            stride_sec=stride_sec,
            overlap=overlap,
        )
        synthetic_segment = segment_signal_windows(
            aligned_bundle["synthetic_values"],
            window_type=window_type,
            window_size=window_size,
            window_duration_sec=window_duration_sec,
            sampling_frequency_hz=aligned_bundle["synthetic_frequency_hz"],
            stride_or_overlap_mode=stride_or_overlap_mode,
            stride=stride,
            stride_sec=stride_sec,
            overlap=overlap,
        )

        if real_segment["windows"].shape != synthetic_segment["windows"].shape:
            raise ValueError(
                "Real and synthetic windows must match after alignment and segmentation. "
                f"Got {real_segment['windows'].shape} vs {synthetic_segment['windows'].shape}"
            )

        capture_real_windows = np.asarray(real_segment["windows"], dtype=np.float32)
        capture_synthetic_windows = np.asarray(synthetic_segment["windows"], dtype=np.float32)
        capture_start_indices = np.asarray(real_segment["start_indices"], dtype=np.int64)

        if max_windows_per_capture is not None and capture_real_windows.shape[0] > int(max_windows_per_capture):
            local_indices = np.sort(rng.choice(capture_real_windows.shape[0], size=int(max_windows_per_capture), replace=False))
            capture_real_windows = capture_real_windows[local_indices]
            capture_synthetic_windows = capture_synthetic_windows[local_indices]
            capture_start_indices = capture_start_indices[local_indices]

        if channel_labels is None:
            channel_labels = list(signal_bundle["channel_labels"])

        capture_id = _capture_id_from_row(row)
        real_windows_blocks.append(capture_real_windows)
        synthetic_windows_blocks.append(capture_synthetic_windows)

        capture_summary_rows.append(
            {
                "capture_index": int(capture_index),
                "capture_id": capture_id,
                "clip_id": str(row["clip_id"]),
                "domain": str(row["domain"]),
                "user_id": int(row["user_id"]),
                "tag_number": int(row["tag_number"]),
                "take_id": None if pd.isna(row.get("take_id")) else row.get("take_id"),
                "emotion": row.get("emotion"),
                "modality": row.get("modality"),
                "stimulus": row.get("stimulus"),
                "aligned_sampling_frequency_hz": float(aligned_bundle["synthetic_frequency_hz"]),
                "aligned_frames": int(aligned_bundle["real_resampled_values"].shape[0]),
                "window_size_samples": int(real_segment["window_size"]),
                "window_duration_sec": None if real_segment["window_duration_sec"] is None else float(real_segment["window_duration_sec"]),
                "step_size_samples": int(real_segment["step_size"]),
                "step_duration_sec": None if real_segment["step_duration_sec"] is None else float(real_segment["step_duration_sec"]),
                "num_windows": int(capture_real_windows.shape[0]),
                "selected_sensors": ", ".join(selected_sensors_resolved),
                "selected_axes": ", ".join([str(axis_name) for axis_name in selected_axes]),
                "signal_groups": ", ".join([str(signal_group) for signal_group in signal_groups]),
            }
        )

        for window_local_index, start_index in enumerate(capture_start_indices):
            pair_rows.append(
                {
                    "pair_index": len(pair_rows),
                    "capture_index": int(capture_index),
                    "capture_id": capture_id,
                    "clip_id": str(row["clip_id"]),
                    "domain": str(row["domain"]),
                    "user_id": int(row["user_id"]),
                    "tag_number": int(row["tag_number"]),
                    "take_id": None if pd.isna(row.get("take_id")) else row.get("take_id"),
                    "emotion": row.get("emotion"),
                    "modality": row.get("modality"),
                    "stimulus": row.get("stimulus"),
                    "window_index_within_capture": int(window_local_index),
                    "window_start_index": int(start_index),
                    "window_start_time_sec": float(aligned_bundle["timestamps_sec"][start_index]),
                }
            )

    if not real_windows_blocks:
        raise ValueError("No windows were produced from the selected captures.")

    real_windows_4d = np.concatenate(real_windows_blocks, axis=0).astype(np.float32)
    synthetic_windows_4d = np.concatenate(synthetic_windows_blocks, axis=0).astype(np.float32)

    if real_windows_4d.shape != synthetic_windows_4d.shape:
        raise ValueError(
            "Real and synthetic window tensors must have the same shape. "
            f"Got {real_windows_4d.shape} vs {synthetic_windows_4d.shape}"
        )

    feature_names = [
        f"{sensor_name}/{channel_label}"
        for sensor_name in selected_sensors_resolved
        for channel_label in (channel_labels or [])
    ]

    return {
        "real_windows_4d": real_windows_4d,
        "synthetic_windows_4d": synthetic_windows_4d,
        "real_windows": _flatten_windows(real_windows_4d),
        "synthetic_windows": _flatten_windows(synthetic_windows_4d),
        "pair_metadata_df": pd.DataFrame(pair_rows),
        "capture_window_summary_df": pd.DataFrame(capture_summary_rows),
        "selected_sensors": list(selected_sensors_resolved),
        "channel_labels": list(channel_labels or []),
        "feature_names": feature_names,
    }


def split_embedder_training_windows(
    metadata_df: pd.DataFrame,
    *,
    train_fraction: float | None = 0.3,
    rare_class_threshold: int = 1,
    random_state: int = 42,
) -> dict[str, Any]:
    if metadata_df.empty:
        raise ValueError("metadata_df is empty.")

    capture_df = metadata_df[
        ["capture_id", "clip_id", "emotion", "modality", "stimulus"]
    ].drop_duplicates("capture_id", keep="first").reset_index(drop=True)

    if train_fraction is None or float(train_fraction) <= 0.0:
        mask = np.ones(len(metadata_df), dtype=bool)
        summary_df = pd.DataFrame(
            [
                {
                    "embedder_split_strategy": "all_real_windows",
                    "train_fraction": 0.0,
                    "num_train_captures": int(capture_df["capture_id"].nunique()),
                    "num_eval_captures": int(capture_df["capture_id"].nunique()),
                    "num_train_windows": int(mask.sum()),
                    "num_eval_windows": int(mask.sum()),
                    "note": "TS2Vec trained and evaluated on the same real windows. This is more practical, but leakage-prone.",
                }
            ]
        )
        return {
            "train_mask": mask,
            "eval_mask": mask.copy(),
            "train_capture_ids": capture_df["capture_id"].astype(str).tolist(),
            "eval_capture_ids": capture_df["capture_id"].astype(str).tolist(),
            "summary_df": summary_df,
        }

    rng = np.random.default_rng(int(random_state))
    protected_eval_captures: set[str] = set()
    for class_column in CLASS_COLUMNS:
        counts = capture_df[class_column].map(_normalize_label_value).value_counts()
        rare_values = set(counts[counts <= int(rare_class_threshold)].index)
        protected_eval_captures.update(
            capture_df.loc[
                capture_df[class_column].map(_normalize_label_value).isin(rare_values), "capture_id"
            ].astype(str).tolist()
        )

    all_capture_ids = capture_df["capture_id"].astype(str).tolist()
    eligible_capture_ids = [capture_id for capture_id in all_capture_ids if capture_id not in protected_eval_captures]
    desired_train_captures = int(round(float(train_fraction) * len(all_capture_ids)))
    desired_train_captures = min(max(1, desired_train_captures), max(1, len(eligible_capture_ids)))

    if not eligible_capture_ids:
        mask = np.ones(len(metadata_df), dtype=bool)
        summary_df = pd.DataFrame(
            [
                {
                    "embedder_split_strategy": "all_real_windows_fallback",
                    "train_fraction": float(train_fraction),
                    "num_train_captures": int(capture_df["capture_id"].nunique()),
                    "num_eval_captures": int(capture_df["capture_id"].nunique()),
                    "num_train_windows": int(mask.sum()),
                    "num_eval_windows": int(mask.sum()),
                    "note": "All captures were protected for evaluation; the embedder fell back to using all real windows.",
                }
            ]
        )
        return {
            "train_mask": mask,
            "eval_mask": mask.copy(),
            "train_capture_ids": all_capture_ids,
            "eval_capture_ids": all_capture_ids,
            "summary_df": summary_df,
        }

    train_capture_ids = sorted(rng.choice(np.asarray(eligible_capture_ids), size=desired_train_captures, replace=False).tolist())
    if len(train_capture_ids) == len(all_capture_ids):
        train_capture_ids = train_capture_ids[:-1]
    eval_capture_ids = [capture_id for capture_id in all_capture_ids if capture_id not in set(train_capture_ids)]

    if not eval_capture_ids:
        eval_capture_ids = [train_capture_ids.pop()]

    train_mask = metadata_df["capture_id"].astype(str).isin(train_capture_ids).to_numpy()
    eval_mask = metadata_df["capture_id"].astype(str).isin(eval_capture_ids).to_numpy()

    summary_df = pd.DataFrame(
        [
            {
                "embedder_split_strategy": "capture_holdout",
                "train_fraction": float(train_fraction),
                "num_train_captures": int(len(train_capture_ids)),
                "num_eval_captures": int(len(eval_capture_ids)),
                "num_train_windows": int(train_mask.sum()),
                "num_eval_windows": int(eval_mask.sum()),
                "protected_eval_captures": int(len(protected_eval_captures)),
                "note": "TS2Vec is trained only on real windows from the training captures; all protocol metrics are computed on evaluation captures only.",
            }
        ]
    )

    return {
        "train_mask": train_mask,
        "eval_mask": eval_mask,
        "train_capture_ids": train_capture_ids,
        "eval_capture_ids": eval_capture_ids,
        "summary_df": summary_df,
    }


def build_metric_groups(
    metadata_df: pd.DataFrame,
    *,
    max_samples_per_group: int | None = None,
    include_category_global: bool = True,
    include_overall: bool = True,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    if metadata_df.empty:
        raise ValueError("metadata_df is empty.")

    rng = np.random.default_rng(int(random_state))
    groups: list[dict[str, Any]] = []
    all_indices = metadata_df.index.to_numpy(dtype=np.int64)

    if include_overall:
        overall_indices = _sample_indices(all_indices, max_samples_per_group, rng)
        groups.append(
            {
                "group_key": "overall::all",
                "class_category": "overall",
                "class_value": "all",
                "indices": overall_indices.tolist(),
            }
        )

    for class_column in CLASS_COLUMNS:
        normalized_labels = metadata_df[class_column].map(_normalize_label_value)
        for class_value in sorted(normalized_labels.unique()):
            class_indices = metadata_df.index[normalized_labels == class_value].to_numpy(dtype=np.int64)
            sampled_indices = _sample_indices(class_indices, max_samples_per_group, rng)
            groups.append(
                {
                    "group_key": f"{class_column}::{class_value}",
                    "class_category": class_column,
                    "class_value": class_value,
                    "indices": sampled_indices.tolist(),
                }
            )

        if include_category_global:
            balanced_indices = _balanced_sample_indices_by_label(
                labels=normalized_labels.tolist(),
                indices=metadata_df.index.to_numpy(dtype=np.int64),
                rng=rng,
                max_total_samples=max_samples_per_group,
            )
            groups.append(
                {
                    "group_key": f"{class_column}::__global__",
                    "class_category": class_column,
                    "class_value": "__global__",
                    "indices": balanced_indices.tolist(),
                }
            )

    return groups


def summarize_metric_groups(groups: Sequence[dict[str, Any]], metadata_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in groups:
        indices = np.asarray(group["indices"], dtype=np.int64)
        subset = metadata_df.iloc[indices]
        rows.append(
            {
                "group_key": str(group["group_key"]),
                "class_category": str(group["class_category"]),
                "class_value": str(group["class_value"]),
                "num_pairs": int(len(indices)),
                "num_captures": int(subset["capture_id"].nunique()) if not subset.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def run_ts2vec_protocol_worker(
    *,
    embedder_train_windows: np.ndarray,
    eval_real_windows: np.ndarray,
    eval_synthetic_windows: np.ndarray,
    eval_capture_ids: Sequence[str],
    groups: Sequence[dict[str, Any]],
    conda_env_name: str = "ts2vec",
    conda_executable: str | Path | None = None,
    repr_dims: int = 64,
    hidden_dims: int = 64,
    depth: int = 8,
    batch_size: int = 16,
    n_iters: int = 200,
    temporal_unit: int = 0,
    ts2vec_device: str = "cpu",
    gru_hidden_size: int = 64,
    gru_num_layers: int = 1,
    gru_dropout: float = 0.0,
    gru_batch_size: int = 64,
    gru_epochs: int = 20,
    gru_learning_rate: float = 1e-3,
    gru_weight_decay: float = 1e-4,
    gru_patience: int = 4,
    cv_folds: int = 10,
    random_state: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    resolved_conda = Path(conda_executable) if conda_executable is not None else None
    if resolved_conda is None:
        conda_path = shutil.which("conda")
        if conda_path is None:
            raise RuntimeError("`conda` was not found in PATH, but it is required to run TS2Vec.")
        resolved_conda = Path(conda_path)

    worker_path = Path(__file__).resolve().with_name("ts2vec_worker.py")
    capture_ids_array = _string_array(eval_capture_ids)
    env_python = resolved_conda.resolve().parents[1] / "envs" / str(conda_env_name) / "bin" / "python"

    with tempfile.TemporaryDirectory(prefix="ts2vec_protocol_") as temporary_dir:
        temporary_root = Path(temporary_dir)
        input_npz = temporary_root / "worker_input.npz"
        groups_json = temporary_root / "groups.json"
        output_npz = temporary_root / "worker_output.npz"

        np.savez_compressed(
            input_npz,
            embedder_train_windows=np.asarray(embedder_train_windows, dtype=np.float32),
            eval_real_windows=np.asarray(eval_real_windows, dtype=np.float32),
            eval_synthetic_windows=np.asarray(eval_synthetic_windows, dtype=np.float32),
            capture_ids=capture_ids_array,
        )
        groups_json.write_text(json.dumps(list(groups)), encoding="utf-8")

        command = [
            str(env_python if env_python.exists() else resolved_conda),
        ]
        if not env_python.exists():
            command.extend(
                [
                    "run",
                    "-n",
                    str(conda_env_name),
                    "python",
                ]
            )
        command.extend(
            [
            str(worker_path),
            "--input-npz",
            str(input_npz),
            "--groups-json",
            str(groups_json),
            "--output-npz",
            str(output_npz),
            "--repr-dims",
            str(int(repr_dims)),
            "--hidden-dims",
            str(int(hidden_dims)),
            "--depth",
            str(int(depth)),
            "--batch-size",
            str(int(batch_size)),
            "--n-iters",
            str(int(n_iters)),
            "--temporal-unit",
            str(int(temporal_unit)),
            "--ts2vec-device",
            str(ts2vec_device),
            "--gru-hidden-size",
            str(int(gru_hidden_size)),
            "--gru-num-layers",
            str(int(gru_num_layers)),
            "--gru-dropout",
            str(float(gru_dropout)),
            "--gru-batch-size",
            str(int(gru_batch_size)),
            "--gru-epochs",
            str(int(gru_epochs)),
            "--gru-learning-rate",
            str(float(gru_learning_rate)),
            "--gru-weight-decay",
            str(float(gru_weight_decay)),
            "--gru-patience",
            str(int(gru_patience)),
            "--cv-folds",
            str(int(cv_folds)),
            "--random-state",
            str(int(random_state)),
            ]
        )

        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=not verbose,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            stderr = "" if error.stderr is None else str(error.stderr).strip()
            stdout = "" if error.stdout is None else str(error.stdout).strip()
            message_parts = ["TS2Vec worker execution failed."]
            if stderr:
                message_parts.append(f"stderr: {stderr}")
            if stdout:
                message_parts.append(f"stdout: {stdout}")
            raise RuntimeError(" ".join(message_parts)) from error

        if not verbose and completed.stdout:
            print(completed.stdout)
        if not verbose and completed.stderr:
            print(completed.stderr)

        with np.load(output_npz, allow_pickle=False) as payload:
            predictive_results = json.loads(str(payload["predictive_results_json"].item()))
            worker_summary = json.loads(str(payload["worker_summary_json"].item()))
            return {
                "real_embeddings": np.asarray(payload["real_embeddings"], dtype=np.float32),
                "synthetic_embeddings": np.asarray(payload["synthetic_embeddings"], dtype=np.float32),
                "loss_log": np.asarray(payload["loss_log"], dtype=np.float32),
                "predictive_results": predictive_results,
                "worker_summary": worker_summary,
            }


def compute_frechet_distance(
    real_embeddings: np.ndarray,
    synthetic_embeddings: np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    real = np.asarray(real_embeddings, dtype=np.float64)
    synthetic = np.asarray(synthetic_embeddings, dtype=np.float64)
    if real.shape[0] < 2 or synthetic.shape[0] < 2:
        return float("nan")

    real_mean = real.mean(axis=0)
    synthetic_mean = synthetic.mean(axis=0)
    real_cov = np.cov(real, rowvar=False) + (float(eps) * np.eye(real.shape[1], dtype=np.float64))
    synthetic_cov = np.cov(synthetic, rowvar=False) + (float(eps) * np.eye(synthetic.shape[1], dtype=np.float64))

    cov_sqrt = sqrtm(real_cov @ synthetic_cov)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    mean_diff = real_mean - synthetic_mean
    fid = mean_diff @ mean_diff
    fid += np.trace(real_cov + synthetic_cov - (2.0 * cov_sqrt))
    return float(np.real_if_close(fid))


def compute_js_distance(
    real_embeddings: np.ndarray,
    synthetic_embeddings: np.ndarray,
    *,
    num_bins: int = 32,
) -> float:
    real = np.asarray(real_embeddings, dtype=np.float64)
    synthetic = np.asarray(synthetic_embeddings, dtype=np.float64)
    if real.shape[0] < 2 or synthetic.shape[0] < 2:
        return float("nan")

    distances: list[float] = []
    for feature_index in range(real.shape[1]):
        pooled = np.concatenate([real[:, feature_index], synthetic[:, feature_index]])
        if pooled.size == 0 or not np.isfinite(pooled).all():
            continue
        value_min = float(np.min(pooled))
        value_max = float(np.max(pooled))
        if math.isclose(value_min, value_max):
            distances.append(0.0)
            continue

        bins = np.linspace(value_min, value_max, int(num_bins) + 1)
        real_hist, _ = np.histogram(real[:, feature_index], bins=bins)
        synthetic_hist, _ = np.histogram(synthetic[:, feature_index], bins=bins)
        if real_hist.sum() == 0 or synthetic_hist.sum() == 0:
            continue

        real_prob = (real_hist.astype(np.float64) + 1e-12) / float(real_hist.sum() + (1e-12 * real_hist.size))
        synthetic_prob = (synthetic_hist.astype(np.float64) + 1e-12) / float(
            synthetic_hist.sum() + (1e-12 * synthetic_hist.size)
        )
        distances.append(float(jensenshannon(real_prob, synthetic_prob)))

    if not distances:
        return float("nan")
    return float(np.mean(distances))


def _resolve_mmd_gamma(
    real_embeddings: np.ndarray,
    synthetic_embeddings: np.ndarray,
    *,
    max_samples_for_heuristic: int = 256,
    gamma: float | None = None,
) -> float:
    if gamma is not None:
        return float(gamma)

    combined = np.concatenate([real_embeddings, synthetic_embeddings], axis=0)
    if combined.shape[0] > int(max_samples_for_heuristic):
        rng = np.random.default_rng(0)
        sample_indices = rng.choice(combined.shape[0], size=int(max_samples_for_heuristic), replace=False)
        combined = combined[sample_indices]

    squared_distances = pairwise_distances(combined, metric="euclidean", squared=True)
    upper_triangle = squared_distances[np.triu_indices_from(squared_distances, k=1)]
    positive_distances = upper_triangle[upper_triangle > 0.0]
    median_distance = float(np.median(positive_distances)) if positive_distances.size > 0 else 1.0
    median_distance = max(median_distance, 1e-6)
    return float(1.0 / median_distance)


def compute_mmd_rbf(
    real_embeddings: np.ndarray,
    synthetic_embeddings: np.ndarray,
    *,
    gamma: float | None = None,
) -> float:
    real = np.asarray(real_embeddings, dtype=np.float64)
    synthetic = np.asarray(synthetic_embeddings, dtype=np.float64)
    if real.shape[0] < 2 or synthetic.shape[0] < 2:
        return float("nan")

    resolved_gamma = _resolve_mmd_gamma(real, synthetic, gamma=gamma)
    kernel_xx = rbf_kernel(real, real, gamma=resolved_gamma)
    kernel_yy = rbf_kernel(synthetic, synthetic, gamma=resolved_gamma)
    kernel_xy = rbf_kernel(real, synthetic, gamma=resolved_gamma)

    m = real.shape[0]
    n = synthetic.shape[0]
    mmd2 = ((kernel_xx.sum() - np.trace(kernel_xx)) / (m * (m - 1))) + ((kernel_yy.sum() - np.trace(kernel_yy)) / (n * (n - 1)))
    mmd2 -= 2.0 * kernel_xy.mean()
    return float(math.sqrt(max(0.0, mmd2)))


def bootstrap_distribution_metric(
    real_values: np.ndarray,
    synthetic_values: np.ndarray,
    metric_fn,
    *,
    n_bootstrap: int = 500,
    sample_size: int | None = 256,
    random_state: int = 42,
) -> dict[str, Any]:
    real = np.asarray(real_values)
    synthetic = np.asarray(synthetic_values)
    if real.shape[0] < 2 or synthetic.shape[0] < 2:
        return {"mean": None, "std": None, "num_bootstrap": 0, "warning": "At least 2 samples are required."}

    effective_sample_size = min(real.shape[0], synthetic.shape[0])
    if sample_size is not None:
        effective_sample_size = min(effective_sample_size, int(sample_size))
    if effective_sample_size < 2:
        return {"mean": None, "std": None, "num_bootstrap": 0, "warning": "Bootstrap sample size is smaller than 2."}

    rng = np.random.default_rng(int(random_state))
    scores: list[float] = []
    for _ in range(int(n_bootstrap)):
        real_indices = rng.integers(0, real.shape[0], size=effective_sample_size)
        synthetic_indices = rng.integers(0, synthetic.shape[0], size=effective_sample_size)
        score = float(metric_fn(real[real_indices], synthetic[synthetic_indices]))
        if np.isfinite(score):
            scores.append(score)

    if not scores:
        return {"mean": None, "std": None, "num_bootstrap": 0, "warning": "All bootstrap samples were invalid."}

    ddof = 1 if len(scores) > 1 else 0
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=ddof)),
        "num_bootstrap": int(len(scores)),
        "warning": None,
    }


def dtw_distance_multivariate(
    real_sequence: np.ndarray,
    synthetic_sequence: np.ndarray,
    *,
    band_radius: int | None = None,
) -> float:
    real = np.asarray(real_sequence, dtype=np.float64)
    synthetic = np.asarray(synthetic_sequence, dtype=np.float64)
    if real.ndim != 2 or synthetic.ndim != 2:
        raise ValueError("DTW sequences must have shape [time, features].")

    local_cost = cdist(real, synthetic, metric="euclidean")
    n_real, n_synthetic = local_cost.shape
    accumulated = np.full((n_real + 1, n_synthetic + 1), np.inf, dtype=np.float64)
    accumulated[0, 0] = 0.0

    for real_index in range(1, n_real + 1):
        if band_radius is None:
            col_start = 1
            col_end = n_synthetic + 1
        else:
            col_start = max(1, real_index - int(band_radius))
            col_end = min(n_synthetic + 1, real_index + int(band_radius) + 1)

        for synthetic_index in range(col_start, col_end):
            best_previous = min(
                accumulated[real_index - 1, synthetic_index],
                accumulated[real_index, synthetic_index - 1],
                accumulated[real_index - 1, synthetic_index - 1],
            )
            accumulated[real_index, synthetic_index] = local_cost[real_index - 1, synthetic_index - 1] + best_previous

    return float(accumulated[n_real, n_synthetic] / float(n_real + n_synthetic))


def _sample_distinct_pairs(num_samples: int, target_pairs: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    if num_samples < 2:
        return []

    max_unique_pairs = (num_samples * (num_samples - 1)) // 2
    requested_pairs = min(int(target_pairs), int(max_unique_pairs))
    if requested_pairs <= 0:
        return []

    sampled_pairs: set[tuple[int, int]] = set()
    while len(sampled_pairs) < requested_pairs:
        first, second = rng.choice(num_samples, size=2, replace=False).tolist()
        sampled_pairs.add((min(first, second), max(first, second)))

    return sorted(sampled_pairs)


def compute_dtw_summary(
    real_windows: np.ndarray,
    synthetic_windows: np.ndarray,
    metadata_df: pd.DataFrame,
    *,
    num_pairs: int = 64,
    band_radius: int | None = None,
    random_state: int = 42,
    group_key: str,
    class_category: str,
    class_value: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    real = np.asarray(real_windows, dtype=np.float32)
    synthetic = np.asarray(synthetic_windows, dtype=np.float32)
    if real.ndim != 3 or synthetic.ndim != 3:
        raise ValueError("DTW windows must have shape [n_windows, time, features].")

    rng = np.random.default_rng(int(random_state))
    detail_rows: list[dict[str, Any]] = []
    scenario_to_scores: dict[str, list[float]] = {"rtr": [], "rts": [], "sts": []}

    paired_indices = np.arange(min(real.shape[0], synthetic.shape[0]), dtype=np.int64)
    if paired_indices.size > 0:
        if paired_indices.size > int(num_pairs):
            paired_indices = np.sort(rng.choice(paired_indices, size=int(num_pairs), replace=False))
        for pair_order, window_index in enumerate(paired_indices):
            distance = dtw_distance_multivariate(real[window_index], synthetic[window_index], band_radius=band_radius)
            scenario_to_scores["rts"].append(distance)
            detail_rows.append(
                {
                    "group_key": group_key,
                    "class_category": class_category,
                    "class_value": class_value,
                    "scenario": "rts",
                    "pair_order": int(pair_order),
                    "distance": float(distance),
                    "capture_id": str(metadata_df.iloc[window_index]["capture_id"]),
                }
            )

    for scenario_name, window_block in (("rtr", real), ("sts", synthetic)):
        pair_indices = _sample_distinct_pairs(window_block.shape[0], int(num_pairs), rng)
        for pair_order, (first_index, second_index) in enumerate(pair_indices):
            distance = dtw_distance_multivariate(window_block[first_index], window_block[second_index], band_radius=band_radius)
            scenario_to_scores[scenario_name].append(distance)
            detail_rows.append(
                {
                    "group_key": group_key,
                    "class_category": class_category,
                    "class_value": class_value,
                    "scenario": scenario_name,
                    "pair_order": int(pair_order),
                    "distance": float(distance),
                    "capture_id": str(metadata_df.iloc[first_index]["capture_id"]),
                }
            )

    summary = {
        "rtr": _as_optional_float(np.mean(scenario_to_scores["rtr"])) if scenario_to_scores["rtr"] else None,
        "rts": _as_optional_float(np.mean(scenario_to_scores["rts"])) if scenario_to_scores["rts"] else None,
        "sts": _as_optional_float(np.mean(scenario_to_scores["sts"])) if scenario_to_scores["sts"] else None,
        "warning": None if any(scenario_to_scores.values()) else "No DTW pairs were available for this group.",
    }
    return summary, detail_rows


def _resolve_classification_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    requested_splits: int,
    random_state: int,
) -> tuple[str, list[tuple[np.ndarray, np.ndarray]]]:
    if labels.size < 2:
        return "unavailable", []

    _, class_counts = np.unique(labels, return_counts=True)
    max_class_splits = int(class_counts.min()) if class_counts.size > 0 else 0
    if max_class_splits < 2:
        return "unavailable", []

    unique_groups = np.unique(groups.astype(str))
    if unique_groups.size >= 2:
        n_splits = min(int(requested_splits), int(unique_groups.size))
        if n_splits >= 2:
            if StratifiedGroupKFold is not None:
                splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
                splits = list(splitter.split(np.zeros(labels.size), labels, groups=groups))
                return "group", [(np.asarray(train_idx), np.asarray(test_idx)) for train_idx, test_idx in splits]

            splitter = GroupKFold(n_splits=n_splits)
            splits = list(splitter.split(np.zeros(labels.size), labels, groups=groups))
            return "group", [(np.asarray(train_idx), np.asarray(test_idx)) for train_idx, test_idx in splits]

    n_splits = min(int(requested_splits), int(max_class_splits))
    if n_splits < 2:
        return "unavailable", []

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
    splits = list(splitter.split(np.zeros(labels.size), labels))
    return "window", [(np.asarray(train_idx), np.asarray(test_idx)) for train_idx, test_idx in splits]


def compute_discriminative_score(
    real_windows: np.ndarray,
    synthetic_windows: np.ndarray,
    capture_ids: Sequence[str],
    *,
    cv_folds: int = 10,
    random_state: int = 42,
) -> dict[str, Any]:
    real = np.asarray(real_windows, dtype=np.float32)
    synthetic = np.asarray(synthetic_windows, dtype=np.float32)
    if real.shape != synthetic.shape:
        raise ValueError("Real and synthetic windows must have the same shape for discriminative score.")

    num_samples = int(real.shape[0])
    if num_samples < 2:
        return {
            "mean": None,
            "std": None,
            "num_splits": 0,
            "cv_strategy": "unavailable",
            "fold_scores": [],
            "warning": "At least 2 windows are required for discriminative score.",
        }

    features = np.concatenate([real.reshape(num_samples, -1), synthetic.reshape(num_samples, -1)], axis=0)
    labels = np.concatenate(
        [
            np.zeros(num_samples, dtype=np.int64),
            np.ones(num_samples, dtype=np.int64),
        ]
    )
    repeated_groups = np.concatenate([_string_array(capture_ids), _string_array(capture_ids)], axis=0)

    strategy, splits = _resolve_classification_splits(
        labels=labels,
        groups=repeated_groups,
        requested_splits=int(cv_folds),
        random_state=int(random_state),
    )
    if not splits:
        return {
            "mean": None,
            "std": None,
            "num_splits": 0,
            "cv_strategy": strategy,
            "fold_scores": [],
            "warning": "Could not build cross-validation folds for discriminative score.",
        }

    scores: list[float] = []
    for fold_index, (train_idx, test_idx) in enumerate(splits):
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        max_iter=300,
                        early_stopping=False,
                        random_state=int(random_state) + fold_index,
                    ),
                ),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            pipeline.fit(features[train_idx], labels[train_idx])
        predictions = pipeline.predict(features[test_idx])
        scores.append(float(accuracy_score(labels[test_idx], predictions)))

    ddof = 1 if len(scores) > 1 else 0
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=ddof)),
        "num_splits": int(len(scores)),
        "cv_strategy": strategy,
        "fold_scores": [float(score) for score in scores],
        "warning": None,
    }


def compute_pearson_summary(
    real_windows_4d: np.ndarray,
    synthetic_windows_4d: np.ndarray,
    metadata_df: pd.DataFrame,
    *,
    selected_sensors: Sequence[str],
    channel_labels: Sequence[str],
    group_key: str,
    class_category: str,
    class_value: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    real = np.asarray(real_windows_4d, dtype=np.float64)
    synthetic = np.asarray(synthetic_windows_4d, dtype=np.float64)
    if real.shape != synthetic.shape:
        raise ValueError("Pearson inputs must have the same shape.")

    detail_rows: list[dict[str, Any]] = []
    window_means: list[float] = []

    for window_index in range(real.shape[0]):
        channel_scores: list[float] = []
        for sensor_index, sensor_name in enumerate(selected_sensors):
            for channel_index, channel_label in enumerate(channel_labels):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    score = pearsonr(real[window_index, :, sensor_index, channel_index], synthetic[window_index, :, sensor_index, channel_index]).statistic
                score_value = float(score) if np.isfinite(score) else np.nan
                if np.isfinite(score_value):
                    channel_scores.append(score_value)
                detail_rows.append(
                    {
                        "group_key": group_key,
                        "class_category": class_category,
                        "class_value": class_value,
                        "window_index": int(window_index),
                        "capture_id": str(metadata_df.iloc[window_index]["capture_id"]),
                        "sensor_name": str(sensor_name),
                        "channel_label": str(channel_label),
                        "pearson": None if not np.isfinite(score_value) else float(score_value),
                    }
                )
        if channel_scores:
            window_means.append(float(np.mean(channel_scores)))

    if not window_means:
        return {"mean": None, "std": None, "warning": "No valid Pearson correlations were found."}, detail_rows

    ddof = 1 if len(window_means) > 1 else 0
    return {
        "mean": float(np.mean(window_means)),
        "std": float(np.std(window_means, ddof=ddof)),
        "warning": None,
    }, detail_rows


def build_embedding_dataframe(
    real_embeddings: np.ndarray,
    synthetic_embeddings: np.ndarray,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    embedding_dim = int(real_embeddings.shape[1])
    embedding_columns = [f"embedding_{dimension:03d}" for dimension in range(embedding_dim)]

    real_df = metadata_df.copy().reset_index(drop=True)
    real_df["domain"] = "real"
    real_df[embedding_columns] = pd.DataFrame(real_embeddings, columns=embedding_columns)

    synthetic_df = metadata_df.copy().reset_index(drop=True)
    synthetic_df["domain"] = "synthetic"
    synthetic_df[embedding_columns] = pd.DataFrame(synthetic_embeddings, columns=embedding_columns)

    return pd.concat([real_df, synthetic_df], axis=0, ignore_index=True)


def evaluate_protocol_groups(
    *,
    groups: Sequence[dict[str, Any]],
    metadata_df: pd.DataFrame,
    real_windows: np.ndarray,
    synthetic_windows: np.ndarray,
    real_windows_4d: np.ndarray,
    synthetic_windows_4d: np.ndarray,
    real_embeddings: np.ndarray,
    synthetic_embeddings: np.ndarray,
    predictive_results: dict[str, Any],
    selected_sensors: Sequence[str],
    channel_labels: Sequence[str],
    bootstrap_iterations: int = 500,
    bootstrap_sample_size: int | None = 256,
    js_bins: int = 32,
    mmd_gamma: float | None = None,
    dtw_pairs: int = 64,
    dtw_band_radius: int | None = None,
    cv_folds: int = 10,
    random_state: int = 42,
) -> dict[str, Any]:
    result_rows: list[dict[str, Any]] = []
    pearson_rows: list[dict[str, Any]] = []
    dtw_rows: list[dict[str, Any]] = []

    for group_index, group in enumerate(groups):
        group_key = str(group["group_key"])
        class_category = str(group["class_category"])
        class_value = str(group["class_value"])
        indices = np.asarray(group["indices"], dtype=np.int64)

        group_metadata = metadata_df.iloc[indices].reset_index(drop=True)
        group_real_windows = np.asarray(real_windows[indices], dtype=np.float32)
        group_synthetic_windows = np.asarray(synthetic_windows[indices], dtype=np.float32)
        group_real_windows_4d = np.asarray(real_windows_4d[indices], dtype=np.float32)
        group_synthetic_windows_4d = np.asarray(synthetic_windows_4d[indices], dtype=np.float32)
        group_real_embeddings = np.asarray(real_embeddings[indices], dtype=np.float32)
        group_synthetic_embeddings = np.asarray(synthetic_embeddings[indices], dtype=np.float32)

        bootstrap_seed = int(random_state) + (group_index * 1000)
        cfid = bootstrap_distribution_metric(
            group_real_embeddings,
            group_synthetic_embeddings,
            lambda real_value, synthetic_value: compute_frechet_distance(real_value, synthetic_value),
            n_bootstrap=int(bootstrap_iterations),
            sample_size=bootstrap_sample_size,
            random_state=bootstrap_seed,
        )
        js = bootstrap_distribution_metric(
            group_real_embeddings,
            group_synthetic_embeddings,
            lambda real_value, synthetic_value: compute_js_distance(real_value, synthetic_value, num_bins=int(js_bins)),
            n_bootstrap=int(bootstrap_iterations),
            sample_size=bootstrap_sample_size,
            random_state=bootstrap_seed + 1,
        )
        mmd = bootstrap_distribution_metric(
            group_real_embeddings,
            group_synthetic_embeddings,
            lambda real_value, synthetic_value: compute_mmd_rbf(real_value, synthetic_value, gamma=mmd_gamma),
            n_bootstrap=int(bootstrap_iterations),
            sample_size=bootstrap_sample_size,
            random_state=bootstrap_seed + 2,
        )
        dtw_summary, dtw_detail_rows = compute_dtw_summary(
            group_real_windows,
            group_synthetic_windows,
            group_metadata,
            num_pairs=int(dtw_pairs),
            band_radius=dtw_band_radius,
            random_state=bootstrap_seed + 3,
            group_key=group_key,
            class_category=class_category,
            class_value=class_value,
        )
        ds = compute_discriminative_score(
            group_real_windows,
            group_synthetic_windows,
            group_metadata["capture_id"].astype(str).to_numpy(),
            cv_folds=int(cv_folds),
            random_state=bootstrap_seed + 4,
        )
        pearson_summary, pearson_detail_rows = compute_pearson_summary(
            group_real_windows_4d,
            group_synthetic_windows_4d,
            group_metadata,
            selected_sensors=selected_sensors,
            channel_labels=channel_labels,
            group_key=group_key,
            class_category=class_category,
            class_value=class_value,
        )
        predictive_summary = dict(predictive_results.get(group_key, {}))

        warning_messages = [
            warning
            for warning in (
                cfid.get("warning"),
                js.get("warning"),
                mmd.get("warning"),
                dtw_summary.get("warning"),
                ds.get("warning"),
                pearson_summary.get("warning"),
                predictive_summary.get("warning"),
            )
            if warning
        ]

        dtw_rows.extend(dtw_detail_rows)
        pearson_rows.extend(pearson_detail_rows)
        result_rows.append(
            {
                "group_key": group_key,
                "class_category": class_category,
                "class_value": class_value,
                "num_pairs": int(len(indices)),
                "num_captures": int(group_metadata["capture_id"].nunique()),
                "cfid_mean": cfid.get("mean"),
                "cfid_std": cfid.get("std"),
                "cfid_num_bootstrap": int(cfid.get("num_bootstrap", 0)),
                "js_mean": js.get("mean"),
                "js_std": js.get("std"),
                "js_num_bootstrap": int(js.get("num_bootstrap", 0)),
                "mmd_mean": mmd.get("mean"),
                "mmd_std": mmd.get("std"),
                "mmd_num_bootstrap": int(mmd.get("num_bootstrap", 0)),
                "dtw_rtr": dtw_summary.get("rtr"),
                "dtw_rts": dtw_summary.get("rts"),
                "dtw_sts": dtw_summary.get("sts"),
                "ds_mean": ds.get("mean"),
                "ds_std": ds.get("std"),
                "ds_num_splits": int(ds.get("num_splits", 0)),
                "ds_cv_strategy": ds.get("cv_strategy"),
                "ps_mean": predictive_summary.get("mean"),
                "ps_std": predictive_summary.get("std"),
                "ps_mae_mean": predictive_summary.get("mae_mean"),
                "ps_mae_std": predictive_summary.get("mae_std"),
                "ps_metric_name": predictive_summary.get("metric_name", "r2"),
                "ps_num_splits": int(predictive_summary.get("num_splits", 0)),
                "ps_cv_strategy": predictive_summary.get("cv_strategy"),
                "pearson_mean": pearson_summary.get("mean"),
                "pearson_std": pearson_summary.get("std"),
                "warnings": " | ".join(warning_messages) if warning_messages else None,
            }
        )

    return {
        "results_df": pd.DataFrame(result_rows),
        "pearson_detail_df": pd.DataFrame(pearson_rows),
        "dtw_detail_df": pd.DataFrame(dtw_rows),
    }


def results_to_protocol_records(results_df: pd.DataFrame) -> list[dict[str, Any]]:
    protocol_records: list[dict[str, Any]] = []
    for _, row in results_df.iterrows():
        protocol_records.append(
            {
                "class_category": str(row["class_category"]),
                "class": str(row["class_value"]),
                "metrics": {
                    "cfid": {"mean": _as_optional_float(row.get("cfid_mean")), "std": _as_optional_float(row.get("cfid_std"))},
                    "js": {"mean": _as_optional_float(row.get("js_mean")), "std": _as_optional_float(row.get("js_std"))},
                    "mmd": {"mean": _as_optional_float(row.get("mmd_mean")), "std": _as_optional_float(row.get("mmd_std"))},
                    "dtw": {
                        "rtr": _as_optional_float(row.get("dtw_rtr")),
                        "rts": _as_optional_float(row.get("dtw_rts")),
                        "sts": _as_optional_float(row.get("dtw_sts")),
                    },
                    "ds": {"mean": _as_optional_float(row.get("ds_mean")), "std": _as_optional_float(row.get("ds_std"))},
                    "ps": {"mean": _as_optional_float(row.get("ps_mean")), "std": _as_optional_float(row.get("ps_std"))},
                    "pearson": {
                        "mean": _as_optional_float(row.get("pearson_mean")),
                        "std": _as_optional_float(row.get("pearson_std")),
                    },
                },
            }
        )
    return protocol_records


def _normalize_metric_for_spider(values: np.ndarray, *, invert: bool) -> np.ndarray:
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return np.zeros_like(values, dtype=np.float64)

    normalized = np.zeros_like(values, dtype=np.float64)
    finite_values = values[finite_mask]
    minimum = float(finite_values.min())
    maximum = float(finite_values.max())
    if math.isclose(minimum, maximum):
        normalized[finite_mask] = 1.0
        return normalized

    normalized[finite_mask] = (values[finite_mask] - minimum) / (maximum - minimum)
    if invert:
        normalized[finite_mask] = 1.0 - normalized[finite_mask]
    return normalized


def plot_metric_spider(results_df: pd.DataFrame, metric_column: str) -> plt.Figure | None:
    subset = results_df[
        results_df["class_category"].astype(str).isin(CLASS_COLUMNS)
        & (results_df["class_value"].astype(str) != "__global__")
    ].copy()
    if subset.empty:
        return None
    if metric_column not in subset.columns:
        raise ValueError(f"Metric column not found: {metric_column}")

    title_suffix = "raw scale"

    fig, axes = plt.subplots(1, len(CLASS_COLUMNS), figsize=(18, 6), subplot_kw={"projection": "polar"})
    plotted_any = False

    for axis, class_category in zip(axes, CLASS_COLUMNS):
        category_subset = subset[subset["class_category"].astype(str) == str(class_category)].copy()
        category_subset = category_subset.sort_values("class_value", kind="stable").reset_index(drop=True)
        raw_values = pd.to_numeric(category_subset[metric_column], errors="coerce").to_numpy(dtype=np.float64)

        if category_subset.empty or raw_values.size < 2:
            axis.set_axis_off()
            continue

        finite_values = raw_values[np.isfinite(raw_values)]
        if finite_values.size == 0:
            axis.set_axis_off()
            continue

        spider_values = raw_values.astype(np.float64)
        angles = np.linspace(0.0, 2.0 * np.pi, spider_values.size, endpoint=False)
        closed_angles = np.concatenate([angles, angles[:1]])
        closed_values = np.concatenate([spider_values, spider_values[:1]])

        radial_min = float(finite_values.min())
        radial_max = float(finite_values.max())
        if math.isclose(radial_min, radial_max):
            padding = max(1e-6, abs(radial_max) * 0.05, 0.05)
            radial_min -= padding
            radial_max += padding
        elif radial_min > 0.0:
            radial_min = 0.0

        axis.plot(closed_angles, closed_values, linewidth=2.2, color="tab:blue")
        axis.fill(closed_angles, closed_values, alpha=0.15, color="tab:blue")
        axis.set_xticks(angles)
        axis.set_xticklabels(category_subset["class_value"].astype(str).tolist())
        axis.set_ylim(radial_min, radial_max)
        axis.set_title(str(class_category).title())
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return None

    fig.suptitle(f"{metric_column} | class spider charts ({title_suffix})", y=1.03)
    fig.tight_layout()
    return fig


def plot_dtw_curves(results_df: pd.DataFrame, class_category: str) -> plt.Figure | None:
    subset = results_df[(results_df["class_category"] == str(class_category)) & (results_df["class_value"] != "__global__")].copy()
    if subset.empty:
        return None

    subset = subset.sort_values("class_value", kind="stable").reset_index(drop=True)
    x_positions = np.arange(len(subset), dtype=np.int64)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x_positions, subset["dtw_rtr"], marker="o", linewidth=2.0, label="RTR")
    ax.plot(x_positions, subset["dtw_rts"], marker="o", linewidth=2.0, label="RTS")
    ax.plot(x_positions, subset["dtw_sts"], marker="o", linewidth=2.0, label="STS")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(subset["class_value"], rotation=25, ha="right")
    ax.set_ylabel("Mean DTW distance")
    ax.set_title(f"{class_category.title()} | DTW scenarios")
    ax.legend()
    ax.grid(alpha=0.25)
    return fig


def plot_pearson_histograms(
    pearson_detail_df: pd.DataFrame,
    class_category: str,
    *,
    bins: int = 20,
) -> plt.Figure | None:
    subset = pearson_detail_df[pearson_detail_df["class_category"] == str(class_category)].copy()
    if subset.empty:
        return None

    class_values = sorted(subset["class_value"].dropna().astype(str).unique().tolist())
    if not class_values:
        return None

    n_cols = 2
    n_rows = int(math.ceil(len(class_values) / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, max(3.5, 3.4 * n_rows)), squeeze=False)
    axes_flat = axes.flatten()

    for axis, class_value in zip(axes_flat, class_values):
        class_subset = subset[subset["class_value"].astype(str) == class_value]
        values = pd.to_numeric(class_subset["pearson"], errors="coerce").dropna().to_numpy(dtype=np.float64)
        axis.hist(values, bins=int(bins), color="tab:blue", alpha=0.8)
        axis.set_title(f"{class_category.title()} | {class_value}")
        axis.set_xlabel("Pearson r")
        axis.set_ylabel("Count")
        axis.grid(alpha=0.25)

    for axis in axes_flat[len(class_values) :]:
        axis.axis("off")

    fig.tight_layout()
    return fig


def run_similarity_protocol(
    *,
    captures_df: pd.DataFrame | None = None,
    output_root: Path | str | None = None,
    capture_domains: Sequence[str] | None = None,
    capture_users: Sequence[int] | None = None,
    capture_tags: Sequence[int] | None = None,
    signal_groups: Sequence[str] = ("acc",),
    selected_sensors: Sequence[str] | None = None,
    selected_axes: Sequence[str] = ("x", "y", "z"),
    resample_method: str = "resample_poly",
    window_type: str = "seconds",
    window_size: int | None = None,
    window_duration_sec: float | None = 3.0,
    stride_or_overlap_mode: str = "overlap",
    stride: int | None = None,
    stride_sec: float | None = None,
    overlap: float = 0.5,
    max_windows_per_capture: int | None = None,
    max_samples_per_group: int | None = None,
    bootstrap_iterations: int = 500,
    bootstrap_sample_size: int | None = 256,
    js_bins: int = 32,
    mmd_gamma: float | None = None,
    dtw_pairs: int = 64,
    dtw_band_radius: int | None = None,
    cv_folds: int = 10,
    embedder_train_fraction: float | None = 0.3,
    embedder_rare_class_threshold: int = 1,
    ts2vec_env_name: str = "ts2vec",
    ts2vec_conda_executable: str | Path | None = None,
    ts2vec_repr_dims: int = 64,
    ts2vec_hidden_dims: int = 64,
    ts2vec_depth: int = 8,
    ts2vec_batch_size: int = 16,
    ts2vec_n_iters: int = 200,
    ts2vec_temporal_unit: int = 0,
    ts2vec_device: str = "cpu",
    gru_hidden_size: int = 64,
    gru_num_layers: int = 1,
    gru_dropout: float = 0.0,
    gru_batch_size: int = 64,
    gru_epochs: int = 20,
    gru_learning_rate: float = 1e-3,
    gru_weight_decay: float = 1e-4,
    gru_patience: int = 4,
    random_state: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    configure_capture_table_display()

    if captures_df is None:
        captures = load_protocol_capture_table(output_root)
    else:
        captures = captures_df.copy()

    if capture_domains is not None:
        captures = captures[captures["domain"].astype(str).isin([str(value) for value in capture_domains])].copy()
    if capture_users is not None:
        captures = captures[captures["user_id"].astype(int).isin([int(value) for value in capture_users])].copy()
    if capture_tags is not None:
        captures = captures[captures["tag_number"].astype(int).isin([int(value) for value in capture_tags])].copy()
    if captures.empty:
        raise ValueError("No captures remain after the requested filters.")

    paired_dataset = build_paired_window_dataset(
        captures,
        signal_groups=signal_groups,
        selected_sensors=selected_sensors,
        selected_axes=selected_axes,
        resample_method=resample_method,
        window_type=window_type,
        window_size=window_size,
        window_duration_sec=window_duration_sec,
        stride_or_overlap_mode=stride_or_overlap_mode,
        stride=stride,
        stride_sec=stride_sec,
        overlap=overlap,
        max_windows_per_capture=max_windows_per_capture,
        random_state=random_state,
    )

    split_result = split_embedder_training_windows(
        paired_dataset["pair_metadata_df"],
        train_fraction=embedder_train_fraction,
        rare_class_threshold=embedder_rare_class_threshold,
        random_state=random_state,
    )

    eval_mask = np.asarray(split_result["eval_mask"], dtype=bool)
    train_mask = np.asarray(split_result["train_mask"], dtype=bool)

    evaluation_metadata_df = paired_dataset["pair_metadata_df"].loc[eval_mask].reset_index(drop=True)
    evaluation_real_windows = paired_dataset["real_windows"][eval_mask]
    evaluation_synthetic_windows = paired_dataset["synthetic_windows"][eval_mask]
    evaluation_real_windows_4d = paired_dataset["real_windows_4d"][eval_mask]
    evaluation_synthetic_windows_4d = paired_dataset["synthetic_windows_4d"][eval_mask]
    train_real_windows = paired_dataset["real_windows"][train_mask]

    groups = build_metric_groups(
        evaluation_metadata_df,
        max_samples_per_group=max_samples_per_group,
        include_category_global=True,
        include_overall=True,
        random_state=random_state,
    )
    group_summary_df = summarize_metric_groups(groups, evaluation_metadata_df)

    worker_result = run_ts2vec_protocol_worker(
        embedder_train_windows=train_real_windows,
        eval_real_windows=evaluation_real_windows,
        eval_synthetic_windows=evaluation_synthetic_windows,
        eval_capture_ids=evaluation_metadata_df["capture_id"].astype(str).tolist(),
        groups=groups,
        conda_env_name=ts2vec_env_name,
        conda_executable=ts2vec_conda_executable,
        repr_dims=ts2vec_repr_dims,
        hidden_dims=ts2vec_hidden_dims,
        depth=ts2vec_depth,
        batch_size=ts2vec_batch_size,
        n_iters=ts2vec_n_iters,
        temporal_unit=ts2vec_temporal_unit,
        ts2vec_device=ts2vec_device,
        gru_hidden_size=gru_hidden_size,
        gru_num_layers=gru_num_layers,
        gru_dropout=gru_dropout,
        gru_batch_size=gru_batch_size,
        gru_epochs=gru_epochs,
        gru_learning_rate=gru_learning_rate,
        gru_weight_decay=gru_weight_decay,
        gru_patience=gru_patience,
        cv_folds=cv_folds,
        random_state=random_state,
        verbose=verbose,
    )

    evaluation_result = evaluate_protocol_groups(
        groups=groups,
        metadata_df=evaluation_metadata_df,
        real_windows=evaluation_real_windows,
        synthetic_windows=evaluation_synthetic_windows,
        real_windows_4d=evaluation_real_windows_4d,
        synthetic_windows_4d=evaluation_synthetic_windows_4d,
        real_embeddings=worker_result["real_embeddings"],
        synthetic_embeddings=worker_result["synthetic_embeddings"],
        predictive_results=worker_result["predictive_results"],
        selected_sensors=paired_dataset["selected_sensors"],
        channel_labels=paired_dataset["channel_labels"],
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_sample_size=bootstrap_sample_size,
        js_bins=js_bins,
        mmd_gamma=mmd_gamma,
        dtw_pairs=dtw_pairs,
        dtw_band_radius=dtw_band_radius,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    results_df = evaluation_result["results_df"]
    pearson_detail_df = evaluation_result["pearson_detail_df"]
    dtw_detail_df = evaluation_result["dtw_detail_df"]
    protocol_records = results_to_protocol_records(results_df)
    embedding_df = build_embedding_dataframe(
        worker_result["real_embeddings"],
        worker_result["synthetic_embeddings"],
        evaluation_metadata_df,
    )

    window_summary_df = pd.DataFrame(
        [
            {
                "num_captures_total": int(paired_dataset["capture_window_summary_df"]["capture_id"].nunique()),
                "num_captures_eval": int(evaluation_metadata_df["capture_id"].nunique()),
                "num_windows_total": int(len(paired_dataset["pair_metadata_df"])),
                "num_windows_eval": int(len(evaluation_metadata_df)),
                "selected_sensors": ", ".join(paired_dataset["selected_sensors"]),
                "channel_labels": ", ".join(paired_dataset["channel_labels"]),
                "ts2vec_repr_dims": int(ts2vec_repr_dims),
                "bootstrap_iterations": int(bootstrap_iterations),
                "cv_folds": int(cv_folds),
            }
        ]
    )

    capture_window_summary_df = paired_dataset["capture_window_summary_df"].copy()
    evaluation_capture_window_summary_df = capture_window_summary_df[
        capture_window_summary_df["capture_id"].isin(split_result["eval_capture_ids"])
    ].copy()

    metric_spider_columns = [
        "cfid_mean",
        "js_mean",
        "mmd_mean",
        "dtw_rtr",
        "dtw_rts",
        "dtw_sts",
        "ds_mean",
        "ps_mean",
        "pearson_mean",
    ]
    radar_figures = {
        metric_column: plot_metric_spider(results_df, metric_column)
        for metric_column in metric_spider_columns
    }
    dtw_figures = {
        class_column: plot_dtw_curves(results_df, class_column)
        for class_column in CLASS_COLUMNS
    }
    pearson_histogram_figures = {
        class_column: plot_pearson_histograms(pearson_detail_df, class_column)
        for class_column in CLASS_COLUMNS
    }

    return {
        "captures_df": captures.reset_index(drop=True),
        "pair_metadata_df": paired_dataset["pair_metadata_df"],
        "evaluation_metadata_df": evaluation_metadata_df,
        "capture_window_summary_df": capture_window_summary_df,
        "evaluation_capture_window_summary_df": evaluation_capture_window_summary_df.reset_index(drop=True),
        "window_summary_df": window_summary_df,
        "split_summary_df": split_result["summary_df"],
        "group_summary_df": group_summary_df,
        "results_df": results_df,
        "pearson_detail_df": pearson_detail_df,
        "dtw_detail_df": dtw_detail_df,
        "embedding_df": embedding_df,
        "protocol_records": protocol_records,
        "worker_summary": worker_result["worker_summary"],
        "embedder_loss_log": worker_result["loss_log"],
        "selected_sensors": paired_dataset["selected_sensors"],
        "channel_labels": paired_dataset["channel_labels"],
        "feature_names": paired_dataset["feature_names"],
        "metric_groups": groups,
        "radar_figures": radar_figures,
        "metric_spider_figures": radar_figures,
        "dtw_figures": dtw_figures,
        "pearson_histogram_figures": pearson_histogram_figures,
    }
