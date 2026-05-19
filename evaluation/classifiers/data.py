from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd

from evaluation.tsne import segment_signal_windows
from evaluation.utils import build_exported_capture_table, load_real_capture

from .alignment import align_target_to_reference, resample_values_to_reference
from .features import (
    build_imu_feature_tensor,
    build_motion_summary_signal,
    build_pose_feature_tensor,
    build_pose_sensor_proxy,
    extract_quality_vector,
    load_pose_sequence3d,
    resolve_imu_orientation_features
)

CaptureBlacklistEntry = tuple[str, int, int] | tuple[str, int, int, str | int]
NormalizedCaptureBlacklistEntry = tuple[str, int, int, str | None]

# Captures with known quality issues (bad tracking, occlusion, sync failure)
ALL_CAPTURE_BLACKLIST: tuple[CaptureBlacklistEntry, ...] = (
    ("30ms", 2, 3),
    ("30ms", 2, 5),
    ("30ms", 2, 6),
    ("30ms", 2, 8),
    ("30ms", 4, 1),
    ("30ms", 5, 8),
    ("30ms", 6, 2),
    ("30ms", 6, 3),
    ("30ms", 6, 7, "1"),
)  # Domain, user_id, tag_number[, take_id]

@dataclass(frozen=True)
class WindowedDatasetConfig:
    window_size: int = 81
    overlap: float = 0.5
    alignment_resample_method: str = "linear"
    alignment_max_lag_samples: int = 20
    alignment_dtw_radius: int = 20
    selected_sensors: Sequence[str] | None = None
    synthetic_variant: str = "raw"
    imu_feature_mode: str = "acc_gyro"
    max_windows_per_capture: int | None = None
    random_state: int = 42
    capture_blacklist: Sequence[CaptureBlacklistEntry] = ALL_CAPTURE_BLACKLIST
    drop_blacklisted_captures: bool = True

# Normalizes blacklist entries to (domain, user_id, tag_number, take_id|None) for uniform lookup
def normalize_capture_blacklist(capture_blacklist: Sequence[CaptureBlacklistEntry] | None) -> set[NormalizedCaptureBlacklistEntry]:
    if capture_blacklist is None:
        return set()

    normalized: set[NormalizedCaptureBlacklistEntry] = set()

    for entry in capture_blacklist:
        if len(entry) == 3:
            domain, user_id, tag_number = entry
            normalized.add((str(domain), int(user_id), int(tag_number), None))
            continue

        if len(entry) == 4:
            domain, user_id, tag_number, take_id = entry
            take_id_normalized = _normalize_take_id_value(take_id)
            normalized.add((str(domain), int(user_id), int(tag_number), take_id_normalized))
            continue

        raise ValueError(
            "Each capture_blacklist entry must have 3 fields "
            "(domain, user_id, tag_number) or 4 fields "
            "(domain, user_id, tag_number, take_id)."
        )
        
    return normalized

def _normalize_take_id_value(take_id: Any) -> str | None:
    if take_id is None or pd.isna(take_id):
        return None
    normalized = str(take_id).strip()
    return normalized or None

def apply_capture_blacklist(captures_df: pd.DataFrame,*,capture_blacklist: Sequence[CaptureBlacklistEntry] | None = ALL_CAPTURE_BLACKLIST,drop_blacklisted: bool = True) -> pd.DataFrame:
    frame = captures_df.copy()
    if frame.empty:
        frame["capture_blacklist_key"] = pd.Series(dtype=object)
        frame["is_blacklisted"] = pd.Series(dtype=bool)
        return frame

    blacklist = normalize_capture_blacklist(capture_blacklist)
    take_ids = (
        frame["take_id"].map(_normalize_take_id_value) if "take_id" in frame.columns
        else pd.Series([None] * len(frame), index=frame.index, dtype=object)
    )
    
    exact_keys = list(zip(frame["domain"].astype(str), frame["user_id"].astype(int), frame["tag_number"].astype(int), take_ids))
    
    broad_keys = [(domain, user_id, tag_number, None) for domain, user_id, tag_number, _ in exact_keys]
    frame["capture_blacklist_key"] = exact_keys
    frame["is_blacklisted"] = [
        exact_key in blacklist or broad_key in blacklist
        for exact_key, broad_key in zip(exact_keys, broad_keys)
    ]
    
    if bool(drop_blacklisted):
        frame = frame.loc[~frame["is_blacklisted"]].copy()
        
    return frame.reset_index(drop=True)

def _load_manifest_frame(output_root: Path | str) -> pd.DataFrame:
    output_root = Path(output_root)
    manifest_path = output_root / "virtual_imu_manifest.jsonl"
    if not manifest_path.exists():
        raise RuntimeError(f"Manifest file not found at expected location: {manifest_path}")

    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            entry = json.loads(line)
            artifacts = dict(entry.get("artifacts", {}))
            quality_report = dict(entry.get("quality_report", {}))
            rows.append(
                {
                    "clip_id": str(entry.get("clip_id")),
                    "domain": str(entry.get("domain")),
                    "user_id": int(entry.get("user_id")),
                    "tag_number": int(entry.get("tag_number")),
                    "take_id": entry.get("take_id"),
                    "status": str(entry.get("status", "fail")),
                    "pose3d_npz_path": artifacts.get("pose3d_npz_path"),
                    "virtual_imu_npz_path": artifacts.get("virtual_imu_npz_path"),
                    "virtual_imu_frame_aligned_npz_path": artifacts.get("virtual_imu_frame_aligned_npz_path"),
                    "virtual_imu_geometric_aligned_npz_path": artifacts.get("virtual_imu_geometric_aligned_npz_path"),
                    "quality_report": quality_report,
                }
            )
    return pd.DataFrame(rows)

def _merge_capture_table_with_manifest(base_table: pd.DataFrame, manifest_frame: pd.DataFrame) -> pd.DataFrame:
    
    merged = base_table.merge(
        manifest_frame,
        how="inner",
        on=["clip_id", "domain", "user_id", "tag_number"],
        suffixes=("", "_manifest"),
        validate="one_to_one"
    )
    
    if merged.empty:
        raise RuntimeError("Could not match capture table against virtual_imu_manifest.jsonl.")

    base_take_ids = (
        merged["take_id"].map(_normalize_take_id_value) if "take_id" in merged.columns
        else pd.Series([None] * len(merged), index=merged.index, dtype=object)
    )
    
    manifest_take_ids = (
        merged["take_id_manifest"].map(_normalize_take_id_value) if "take_id_manifest" in merged.columns
        else pd.Series([None] * len(merged), index=merged.index, dtype=object)
    )
    
    mismatched_take_ids = manifest_take_ids.notna() & base_take_ids.notna() & (manifest_take_ids != base_take_ids)
    
    if bool(mismatched_take_ids.any()):
        mismatch_examples = merged.loc[mismatched_take_ids, ["clip_id", "take_id", "take_id_manifest"]].head(5)
        raise RuntimeError(
            "Found take_id mismatches between exported capture table and virtual_imu_manifest.jsonl: "
            f"{mismatch_examples.to_dict(orient='records')}"
        )

    if "take_id_manifest" in merged.columns:
        merged["take_id"] = merged["take_id"].where(merged["take_id"].notna(), merged["take_id_manifest"])
    return merged


# Builds the full capture index: merges export table with manifest, adds subject_group/flat_tag columns, drops invalid/blacklisted clips
def build_classifier_capture_table(output_root: Path | str, *, capture_blacklist: Sequence[CaptureBlacklistEntry] | None = ALL_CAPTURE_BLACKLIST, drop_blacklisted: bool = True) -> pd.DataFrame:
    
    base_table = build_exported_capture_table(output_root).copy()
    manifest_frame = _load_manifest_frame(output_root).copy()
    
    if manifest_frame.empty:
        raise RuntimeError("virtual_imu_manifest.jsonl is empty.")

    merged = _merge_capture_table_with_manifest(base_table, manifest_frame)
    
    merged["subject_group"] = merged.apply(
        lambda row: f"{row['domain']}_user_{int(row['user_id']):02d}",
        axis=1
    )
    
    merged["flat_tag"] = merged.apply(
        lambda row: f"{row['emotion']}|{row['modality']}|{row['stimulus']}",
        axis=1
    )
    
    merged["frame_aligned_available"] = merged["virtual_imu_frame_aligned_npz_path"].notna()
    merged["virtual_imu_uncalibrated_npz_path"] = merged["virtual_imu_geometric_aligned_npz_path"]
    merged["real_imu_reference_npz_path"] = merged["clip_dir"].apply(
        lambda d: str(Path(d) / "imu.npz") if pd.notna(d) else None
    )
    merged = merged[merged["pose3d_npz_path"].notna() & merged["virtual_imu_npz_path"].notna()].copy()
    
    merged = apply_capture_blacklist(
        merged,
        capture_blacklist=capture_blacklist,
        drop_blacklisted=drop_blacklisted
    )

    return merged.sort_values(
        ["domain", "user_id", "tag_number", "take_id", "clip_id"],
        kind="stable",
    ).reset_index(drop=True)

def _load_virtual_capture_from_path(virtual_imu_npz_path: str | Path) -> dict[str, Any]:
    payload_path = Path(virtual_imu_npz_path)
    with np.load(payload_path, allow_pickle=True) as payload:
        return {
            "timestamps_sec": np.asarray(payload["timestamps_sec"], dtype=np.float32),
            "acc": np.asarray(payload["acc"], dtype=np.float32),
            "gyro": np.asarray(payload["gyro"], dtype=np.float32),
            "sensor_names": [str(value) for value in np.asarray(payload["sensor_names"]).tolist()],
            "fps": float(np.asarray(payload["fps"]).item()),
            "clip_id": str(np.asarray(payload["clip_id"]).item()),
            "source": str(np.asarray(payload["source"]).item()),
            "path": str(payload_path.resolve()),
        }

def _resolve_synthetic_npz_path(capture_row: pd.Series | dict[str, Any], *, synthetic_variant: str) -> Path:
    row = pd.Series(capture_row)
    normalized_variant = str(synthetic_variant).strip().lower()
    if normalized_variant not in {"raw", "frame_aligned", "auto"}:
        raise ValueError("synthetic_variant must be 'raw', 'frame_aligned', or 'auto'.")

    raw_path = Path(str(row["virtual_imu_npz_path"]))
    frame_aligned_value = row.get("virtual_imu_frame_aligned_npz_path")
    frame_aligned_path = None if pd.isna(frame_aligned_value) or frame_aligned_value in (None, "") else Path(str(frame_aligned_value))

    if normalized_variant == "frame_aligned":
        if frame_aligned_path is None or not frame_aligned_path.exists():
            raise ValueError("Requested synthetic_variant='frame_aligned', but no aligned file is available.")
        return frame_aligned_path
    if normalized_variant == "auto" and frame_aligned_path is not None and frame_aligned_path.exists():
        return frame_aligned_path
    return raw_path

def _split_real_imu_channels(real_capture: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    channel_axis_order = [str(value).lower() for value in real_capture["channel_axis_order"]]
    acc_indices = [channel_axis_order.index(axis_name) for axis_name in ("ax", "ay", "az")]
    gyro_indices = [channel_axis_order.index(axis_name) for axis_name in ("gx", "gy", "gz")]
    imu_block = np.asarray(real_capture["imu"], dtype=np.float32)
    return imu_block[:, :, acc_indices], imu_block[:, :, gyro_indices]

def _resolve_selected_sensors(*, requested_sensors: Sequence[str] | None, real_sensor_names: Sequence[str], synthetic_sensor_names: Sequence[str], pose_proxy_sensor_names: Sequence[str]) -> list[str]:
    
    shared = [
        sensor_name 
        for sensor_name in real_sensor_names
        if sensor_name in set(synthetic_sensor_names) and sensor_name in set(pose_proxy_sensor_names)
    ]
    
    if len(shared) == 0:
        raise ValueError("There are no sensors shared by the real, synthetic, and pose-proxy streams.")
    if requested_sensors is None:
        return shared

    requested = [str(sensor_name) for sensor_name in requested_sensors]
    missing = [sensor_name for sensor_name in requested if sensor_name not in set(shared)]
    if len(missing) > 0:
        raise ValueError(f"Requested sensors are not available in every modality: {missing}")
    return requested

def _select_sensor_block(values: np.ndarray, sensor_names: Sequence[str], selected_sensors: Sequence[str]) -> np.ndarray:
    indices = [list(sensor_names).index(sensor_name) for sensor_name in selected_sensors]
    return np.asarray(values[:, indices, :], dtype=np.float32)

def _trim_real_capture(real_capture: dict[str, Any], time_range_sec: tuple[float, float]) -> dict[str, Any]:
    ts = real_capture["timestamps_sec"]
    t0, t1 = float(time_range_sec[0]), float(time_range_sec[1])
    mask = (ts >= t0) & (ts <= t1)
    if not mask.any():
        return real_capture
    ts_trimmed = ts[mask]
    return {
        **real_capture,
        "timestamps_sec": ts_trimmed - ts_trimmed[0],
        "imu": real_capture["imu"][mask]
    }

# Loads pose3d, real IMU, and synthetic IMU for a single capture row
def load_capture_modalities(capture_row: pd.Series | dict[str, Any], *, synthetic_variant: str = "raw") -> dict[str, Any]:
    row = pd.Series(capture_row)
    pose_sequence = load_pose_sequence3d(str(row["pose3d_npz_path"]))
    real_capture = load_real_capture(row["clip_dir"])
    time_range = row.get("real_imu_time_range_sec")
    if time_range is not None:
        real_capture = _trim_real_capture(real_capture, time_range)
    synthetic_capture = _load_virtual_capture_from_path(_resolve_synthetic_npz_path(row, synthetic_variant=synthetic_variant))
    return {
        "pose_sequence": pose_sequence,
        "real_capture": real_capture,
        "synthetic_capture": synthetic_capture
    }

# Full per-capture pipeline: align modalities → extract features → segment windows → assemble metadata
def prepare_capture_windows(capture_row: pd.Series | dict[str, Any], *, config: WindowedDatasetConfig | None = None) -> dict[str, Any]:
    row = pd.Series(capture_row).copy()
    resolved_config = WindowedDatasetConfig() if config is None else config
    modalities = load_capture_modalities(row, synthetic_variant=resolved_config.synthetic_variant)
    pose_sequence = modalities["pose_sequence"]
    real_capture = modalities["real_capture"]
    synthetic_capture = modalities["synthetic_capture"]

    pose_features = build_pose_feature_tensor(pose_sequence)
    pose_proxy = build_pose_sensor_proxy(pose_sequence)
    real_acc, real_gyro = _split_real_imu_channels(real_capture)
    selected_sensors = _resolve_selected_sensors(
        requested_sensors=resolved_config.selected_sensors,
        real_sensor_names=real_capture["sensor_names"],
        synthetic_sensor_names=synthetic_capture["sensor_names"],
        pose_proxy_sensor_names=pose_proxy["sensor_names"]
    )

    pose_proxy_acc = _select_sensor_block(pose_proxy["acc"], pose_proxy["sensor_names"], selected_sensors)
    pose_proxy_gyro = _select_sensor_block(pose_proxy["gyro"], pose_proxy["sensor_names"], selected_sensors)
    real_acc_selected = _select_sensor_block(real_acc, real_capture["sensor_names"], selected_sensors)
    real_gyro_selected = _select_sensor_block(real_gyro, real_capture["sensor_names"], selected_sensors)
    synthetic_acc_selected = _select_sensor_block(synthetic_capture["acc"], synthetic_capture["sensor_names"], selected_sensors)
    synthetic_gyro_selected = _select_sensor_block(synthetic_capture["gyro"], synthetic_capture["sensor_names"], selected_sensors)

    pose_proxy_orientation = resolve_imu_orientation_features(
        pose_proxy_acc,
        pose_proxy_gyro,
        feature_mode=resolved_config.imu_feature_mode
    )
    
    real_orientation = resolve_imu_orientation_features(
        real_acc_selected,
        real_gyro_selected,
        feature_mode=resolved_config.imu_feature_mode
    )
    
    synthetic_orientation = resolve_imu_orientation_features(
        synthetic_acc_selected,
        synthetic_gyro_selected,
        feature_mode=resolved_config.imu_feature_mode
    )

    pose_summary = build_motion_summary_signal(pose_proxy_acc, pose_proxy_orientation["values"])
    real_summary = build_motion_summary_signal(real_acc_selected, real_orientation["values"])

    # Align real IMU to pose timeline using cross-correlation + DTW
    real_alignment = align_target_to_reference(
        pose_features["timestamps_sec"],
        pose_features["values"],
        real_capture["timestamps_sec"],
        np.concatenate([real_acc_selected, real_orientation["values"]], axis=2),
        reference_summary=pose_summary,
        target_summary=real_summary,
        resample_method=resolved_config.alignment_resample_method,
        max_lag_samples=resolved_config.alignment_max_lag_samples,
        dtw_radius=resolved_config.alignment_dtw_radius
    )
    
    aligned_timestamps = np.asarray(real_alignment["timestamps_sec"], dtype=np.float32)
    aligned_pose_values = np.asarray(real_alignment["reference_values"], dtype=np.float32)
    aligned_real_raw = np.asarray(real_alignment["aligned_target_values"], dtype=np.float32)
    aligned_real_acc = aligned_real_raw[:, :, :3]
    aligned_real_gyro = aligned_real_raw[:, :, 3:]

    # Resample synthetic IMU to the aligned timeline (no DTW needed — same source as pose)
    aligned_synthetic_raw = resample_values_to_reference(
        synthetic_capture["timestamps_sec"],
        np.concatenate([synthetic_acc_selected, synthetic_orientation["values"]], axis=2),
        aligned_timestamps,
        method=resolved_config.alignment_resample_method
    )
    
    aligned_synthetic_acc = aligned_synthetic_raw[:, :, :3]
    aligned_synthetic_gyro = aligned_synthetic_raw[:, :, 3:]

    real_imu_features = build_imu_feature_tensor(
        aligned_real_acc,
        aligned_real_gyro,
        aligned_timestamps,
        feature_mode=resolved_config.imu_feature_mode
    )
    
    synthetic_imu_features = build_imu_feature_tensor(
        aligned_synthetic_acc,
        aligned_synthetic_gyro,
        aligned_timestamps,
        feature_mode=resolved_config.imu_feature_mode
    )
    
    quality_vector = extract_quality_vector(
        row.get("quality_report"),
        pose_imu_alignment=real_alignment,
        imu_feature_mode=resolved_config.imu_feature_mode
    )

    pose_window_bundle = segment_signal_windows(
        aligned_pose_values,
        window_type="n_samples",
        window_size=int(resolved_config.window_size),
        stride_or_overlap_mode="overlap",
        overlap=float(resolved_config.overlap)
    )
    
    real_window_bundle = segment_signal_windows(
        real_imu_features["values"],
        window_type="n_samples",
        window_size=int(resolved_config.window_size),
        stride_or_overlap_mode="overlap",
        overlap=float(resolved_config.overlap)
    )
    
    synthetic_window_bundle = segment_signal_windows(
        synthetic_imu_features["values"],
        window_type="n_samples",
        window_size=int(resolved_config.window_size),
        stride_or_overlap_mode="overlap",
        overlap=float(resolved_config.overlap)
    )
    
    if pose_window_bundle["windows"].shape[0] != real_window_bundle["windows"].shape[0] or pose_window_bundle["windows"].shape[0] != synthetic_window_bundle["windows"].shape[0]:
        raise ValueError("Pose, real IMU, and synthetic IMU window counts must match.")

    pose_windows = np.asarray(pose_window_bundle["windows"], dtype=np.float32)
    real_windows = np.asarray(real_window_bundle["windows"], dtype=np.float32)
    synthetic_windows = np.asarray(synthetic_window_bundle["windows"], dtype=np.float32)
    start_indices = np.asarray(pose_window_bundle["start_indices"], dtype=np.int32)

    if resolved_config.max_windows_per_capture is not None and pose_windows.shape[0] > int(resolved_config.max_windows_per_capture):
        rng = np.random.default_rng(int(resolved_config.random_state))
        chosen_indices = np.sort(
            rng.choice(
                pose_windows.shape[0],
                size=int(resolved_config.max_windows_per_capture),
                replace=False
        ))
        
        pose_windows = pose_windows[chosen_indices]
        real_windows = real_windows[chosen_indices]
        synthetic_windows = synthetic_windows[chosen_indices]
        start_indices = start_indices[chosen_indices]

    quality_windows = np.repeat(quality_vector["values"][None, :], pose_windows.shape[0], axis=0).astype(np.float32)
    metadata_rows = []
    for window_index, start_index in enumerate(start_indices.tolist()):
        metadata_rows.append(
            {
                "sample_id": f"{row['clip_id']}::window_{window_index:04d}",
                "capture_id": str(row["clip_id"]),
                "clip_id": str(row["clip_id"]),
                "domain": str(row["domain"]),
                "user_id": int(row["user_id"]),
                "tag_number": int(row["tag_number"]),
                "take_id": row.get("take_id"),
                "emotion": str(row["emotion"]),
                "modality": str(row["modality"]),
                "stimulus": str(row["stimulus"]),
                "flat_tag": str(row["flat_tag"]),
                "subject_group": str(row["subject_group"]),
                "window_index": int(window_index),
                "window_start_index": int(start_index),
                "window_size": int(resolved_config.window_size),
                "window_overlap": float(resolved_config.overlap),
                "quality_status": str(row.get("status", "unknown")),
                "synthetic_variant": str(resolved_config.synthetic_variant),
                "imu_feature_mode": str(real_imu_features["feature_mode"]),
                "selected_sensors": tuple(str(sensor_name) for sensor_name in selected_sensors),
                "pose_imu_lag_samples": int(real_alignment["lag_samples"]),
                "pose_imu_lag_seconds": float(real_alignment["lag_seconds"]),
                "pose_imu_correlation_before_dtw": real_alignment["correlation_before_dtw"],
                "pose_imu_correlation_after_dtw": real_alignment["correlation_after_dtw"],
                "pose_imu_dtw_normalized_distance": float(real_alignment["dtw_normalized_distance"]),
                "visible_joint_ratio": float(dict(row.get("quality_report", {})).get("visible_joint_ratio", 0.0) or 0.0),
                "mean_confidence": float(dict(row.get("quality_report", {})).get("mean_confidence", 0.0) or 0.0),
                "temporal_jitter_score": float(dict(row.get("quality_report", {})).get("temporal_jitter_score", 0.0) or 0.0),
                "root_drift_score": float(dict(row.get("quality_report", {})).get("root_drift_score", 0.0) or 0.0)
            }
        )

    return {
        "metadata": pd.DataFrame(metadata_rows),
        "pose_windows": pose_windows,
        "imu_real_windows": real_windows,
        "imu_synthetic_windows": synthetic_windows,
        "quality_windows": quality_windows,
        "alignment_report": {
            "clip_id": str(row["clip_id"]),
            "selected_sensors": list(selected_sensors),
            "imu_feature_mode": str(real_imu_features["feature_mode"]),
            "lag_samples": int(real_alignment["lag_samples"]),
            "lag_seconds": float(real_alignment["lag_seconds"]),
            "correlation_before_dtw": real_alignment["correlation_before_dtw"],
            "correlation_after_dtw": real_alignment["correlation_after_dtw"],
            "dtw_normalized_distance": float(real_alignment["dtw_normalized_distance"]),
            "aligned_frequency_hz": float(real_alignment["aligned_frequency_hz"]),
            "num_aligned_frames": int(aligned_pose_values.shape[0]),
        },
        "pose_feature_names": list(pose_features["channel_names"]),
        "imu_feature_names": list(real_imu_features["channel_names"]),
        "imu_feature_mode": str(real_imu_features["feature_mode"]),
        "quality_feature_names": list(quality_vector["feature_names"]),
        "joint_names": list(pose_features["joint_names"]),
        "selected_sensors": list(selected_sensors)
    }


def _encode_label_column(values: Sequence[str]) -> dict[str, Any]:
    categories = sorted({str(value) for value in values})
    mapping = {label: index for index, label in enumerate(categories)}
    encoded = np.asarray([mapping[str(value)] for value in values], dtype=np.int64)
    return {
        "encoded": encoded,
        "class_names": categories,
        "mapping": mapping,
    }

# Iterates all captures, runs prepare_capture_windows on each, and concatenates into a single dataset dict
def build_windowed_multimodal_dataset(output_root: Path | str, *, config: WindowedDatasetConfig | None = None, captures_df: pd.DataFrame | None = None) -> dict[str, Any]:
    resolved_config = WindowedDatasetConfig() if config is None else config
    capture_table = (
        build_classifier_capture_table(
            output_root,
            capture_blacklist=resolved_config.capture_blacklist,
            drop_blacklisted=resolved_config.drop_blacklisted_captures
        ) 
        if captures_df is None else 
        apply_capture_blacklist(
            captures_df,
            capture_blacklist=resolved_config.capture_blacklist,
            drop_blacklisted=resolved_config.drop_blacklisted_captures
        )
    )
    
    if capture_table.empty:
        raise RuntimeError("No capture is available to build the classifier dataset.")

    metadata_blocks: list[pd.DataFrame] = []
    pose_blocks: list[np.ndarray] = []
    real_blocks: list[np.ndarray] = []
    synthetic_blocks: list[np.ndarray] = []
    quality_blocks: list[np.ndarray] = []
    alignment_rows: list[dict[str, Any]] = []
    pose_feature_names: list[str] | None = None
    imu_feature_names: list[str] | None = None
    quality_feature_names: list[str] | None = None
    joint_names: list[str] | None = None
    selected_sensors: list[str] | None = None
    imu_feature_mode: str | None = None

    for _, capture_row in capture_table.iterrows():
        try:
            prepared = prepare_capture_windows(capture_row, config=resolved_config)
        except ValueError:
            continue
        if prepared["metadata"].empty:
            continue
        metadata_blocks.append(prepared["metadata"])
        pose_blocks.append(np.asarray(prepared["pose_windows"], dtype=np.float32))
        real_blocks.append(np.asarray(prepared["imu_real_windows"], dtype=np.float32))
        synthetic_blocks.append(np.asarray(prepared["imu_synthetic_windows"], dtype=np.float32))
        quality_blocks.append(np.asarray(prepared["quality_windows"], dtype=np.float32))
        alignment_rows.append(dict(prepared["alignment_report"]))
        if pose_feature_names is None:
            pose_feature_names = list(prepared["pose_feature_names"])
            imu_feature_names = list(prepared["imu_feature_names"])
            quality_feature_names = list(prepared["quality_feature_names"])
            joint_names = list(prepared["joint_names"])
            selected_sensors = list(prepared["selected_sensors"])
            imu_feature_mode = str(prepared["imu_feature_mode"])

    if len(metadata_blocks) == 0:
        raise RuntimeError("No valid capture produced aligned windows for classifier training.")

    metadata = pd.concat(metadata_blocks, axis=0, ignore_index=True)
    pose_windows = np.concatenate(pose_blocks, axis=0).astype(np.float32)
    imu_real_windows = np.concatenate(real_blocks, axis=0).astype(np.float32)
    imu_synthetic_windows = np.concatenate(synthetic_blocks, axis=0).astype(np.float32)
    quality_windows = np.concatenate(quality_blocks, axis=0).astype(np.float32)

    label_encoders = {
        "emotion": _encode_label_column(metadata["emotion"].astype(str).tolist()),
        "modality": _encode_label_column(metadata["modality"].astype(str).tolist()),
        "stimulus": _encode_label_column(metadata["stimulus"].astype(str).tolist()),
        "flat_tag": _encode_label_column(metadata["flat_tag"].astype(str).tolist()),
    }
    for label_name, encoder in label_encoders.items():
        metadata[f"{label_name}_id"] = encoder["encoded"]

    return {
        "capture_table": capture_table.reset_index(drop=True),
        "metadata": metadata.reset_index(drop=True),
        "pose_windows": pose_windows,
        "imu_real_windows": imu_real_windows,
        "imu_synthetic_windows": imu_synthetic_windows,
        "quality_windows": quality_windows,
        "alignment_summary": pd.DataFrame(alignment_rows),
        "label_encoders": label_encoders,
        "pose_feature_names": [] if pose_feature_names is None else pose_feature_names,
        "imu_feature_names": [] if imu_feature_names is None else imu_feature_names,
        "imu_feature_mode": "acc_gyro" if imu_feature_mode is None else imu_feature_mode,
        "quality_feature_names": [] if quality_feature_names is None else quality_feature_names,
        "joint_names": [] if joint_names is None else joint_names,
        "selected_sensors": [] if selected_sensors is None else selected_sensors,
        "config": resolved_config
    }