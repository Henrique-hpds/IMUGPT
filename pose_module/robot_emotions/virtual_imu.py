"""RobotEmotions-specific wrapper around the full virtual-IMU pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from pose_module.export.imusim_adapter import (
    DEFAULT_MAX_ACCELERATION_WARNING_M_S2,
    DEFAULT_MAX_GYRO_WARNING_RAD_S,
    _build_virtual_imu_quality_report,
)
from pose_module.imu_alignment import (
    build_alignment_config,
    fit_sensor_subject_transforms,
    load_alignment_runtime_settings,
    load_real_imu_sequence,
    run_geometric_alignment,
    save_transforms_json,
    sequence_from_virtual_imu,
)
from pose_module.io.cache import write_json_file
from pose_module.pipeline import run_virtual_imu_from_pose3d, run_virtual_imu_pipeline
from pose_module.processing.imu_calibration import (
    DEFAULT_CALIBRATION_PERCENTILE_RESOLUTION,
    DEFAULT_CALIBRATION_SIGNAL_MODE,
    build_calibration_reference_matrix,
    calibrate_virtual_imu_sequence,
)
from pose_module.processing.quality import merge_virtual_imu_quality_reports

from .extractor import (
    RobotEmotionsClipRecord,
    RobotEmotionsExtractor,
    resolve_pose_output_dir,
)


def run_robot_emotions_virtual_imu(
    dataset_root: str,
    output_dir: str,
    *,
    fps_target: int = 20,
    clip_ids: Optional[Sequence[str]] = None,
    save_debug: bool = True,
    save_debug_2d: Optional[bool] = None,
    save_debug_3d: Optional[bool] = None,
    env_name: str = "openmmlab",
    motionbert_env_name: Optional[str] = None,
    motionbert_window_size: int = 81,
    motionbert_window_overlap: float = 0.5,
    include_motionbert_confidence: bool = True,
    motionbert_device: str = "auto",
    allow_motionbert_fallback_backend: bool = False,
    sensor_layout_path: Optional[str] = None,
    imu_acc_noise_std_m_s2: Optional[float] = None,
    imu_gyro_noise_std_rad_s: Optional[float] = None,
    imu_random_seed: int = 0,
    estimate_sensor_frame: bool = False,
    estimate_sensor_names: Optional[Sequence[str]] = None,
    domains: Sequence[str] = ("10ms", "30ms"),
    pose3d_manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    extractor = RobotEmotionsExtractor(dataset_root, domains=tuple(str(domain) for domain in domains))
    records = extractor.select_records(
        clip_ids=None if clip_ids is None else list(clip_ids),
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    alignment_settings = load_alignment_runtime_settings(None)
    use_subject_level_alignment = bool(alignment_settings.get("enable", False)) and not bool(
        alignment_settings.get("fit_from_current_pair", False)
    )

    # Build clip_id → pose3d_npz_path index from an existing pose3d manifest, if provided.
    pose3d_index: Dict[str, str] = {}
    if pose3d_manifest_path is not None:
        manifest_lines = Path(pose3d_manifest_path).read_text(encoding="utf-8").splitlines()
        for line in manifest_lines:
            if not line.strip():
                continue
            entry = json.loads(line)
            npz_path = entry.get("artifacts", {}).get("pose3d_npz_path")
            if npz_path and entry.get("clip_id"):
                pose3d_index[str(entry["clip_id"])] = str(npz_path)

    manifest_entries = []
    processed_runs = []
    num_ok = 0
    num_warning = 0
    num_fail = 0
    for record in records:
        manifest_entry = extractor.ensure_exported_clip(record, output_root=output_root)
        activity_label = None
        try:
            cached_pose3d_path = pose3d_index.get(record.clip_id)
            if cached_pose3d_path is not None:
                pipeline_result = run_virtual_imu_from_pose3d(
                    clip_id=record.clip_id,
                    pose3d_npz_path=cached_pose3d_path,
                    output_dir=resolve_pose_output_dir(output_root, record),
                    sensor_layout_path=(
                        None if sensor_layout_path in (None, "") else str(sensor_layout_path)
                    ),
                    imu_acc_noise_std_m_s2=imu_acc_noise_std_m_s2,
                    imu_gyro_noise_std_rad_s=imu_gyro_noise_std_rad_s,
                    imu_random_seed=int(imu_random_seed),
                )
            else:
                pipeline_result = run_virtual_imu_pipeline(
                    clip_id=record.clip_id,
                    video_path=str(record.video_path.resolve()),
                    output_dir=resolve_pose_output_dir(output_root, record),
                    activity_label=activity_label,
                    fps_target=int(fps_target),
                    save_debug=bool(save_debug),
                    save_debug_2d=save_debug_2d,
                    save_debug_3d=save_debug_3d,
                    env_name=str(env_name),
                    video_metadata=manifest_entry.get("video", {}),
                    model_alias="vitpose-b",
                    motionbert_env_name=motionbert_env_name,
                    motionbert_window_size=int(motionbert_window_size),
                    motionbert_window_overlap=float(motionbert_window_overlap),
                    include_motionbert_confidence=bool(include_motionbert_confidence),
                    motionbert_device=str(motionbert_device),
                    allow_motionbert_fallback_backend=bool(allow_motionbert_fallback_backend),
                    sensor_layout_path=(
                        None if sensor_layout_path in (None, "") else str(sensor_layout_path)
                    ),
                    imu_acc_noise_std_m_s2=imu_acc_noise_std_m_s2,
                    imu_gyro_noise_std_rad_s=imu_gyro_noise_std_rad_s,
                    imu_random_seed=int(imu_random_seed),
                    estimate_sensor_frame=bool(estimate_sensor_frame),
                    estimate_sensor_names=(
                        None if estimate_sensor_names is None else tuple(str(name) for name in estimate_sensor_names)
                    ),
                )
            processed_runs.append(
                {
                    "record": record,
                    "manifest_entry": manifest_entry,
                    "pipeline_result": pipeline_result,
                    "activity_label": activity_label,
                    "pose_dir": resolve_pose_output_dir(output_root, record),
                }
            )
        except Exception as exc:
            entry = _build_virtual_imu_failure_entry(
                record=record,
                manifest_entry=manifest_entry,
                pose_dir=resolve_pose_output_dir(output_root, record),
                error=str(exc),
            )
            num_fail += 1
            manifest_entries.append(entry)

    if use_subject_level_alignment and len(processed_runs) > 0:
        _apply_subject_level_geometric_alignment(
            processed_runs=processed_runs,
            output_root=output_root,
            alignment_settings=alignment_settings,
        )

    for processed in processed_runs:
        entry = _build_virtual_imu_manifest_entry(
            record=processed["record"],
            manifest_entry=processed["manifest_entry"],
            pipeline_result=processed["pipeline_result"],
        )
        status = str(entry.get("status", "fail"))
        if status == "ok":
            num_ok += 1
        elif status == "warning":
            num_warning += 1
        else:
            num_fail += 1
        manifest_entries.append(entry)

    manifest_path = output_root / "virtual_imu_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in manifest_entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    summary = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "output_dir": str(output_root.resolve()),
        "domains": [str(domain) for domain in domains],
        "num_requested_clips": int(len(records)),
        "num_virtual_imu_entries": int(len(manifest_entries)),
        "num_ok": int(num_ok),
        "num_warning": int(num_warning),
        "num_fail": int(num_fail),
        "virtual_imu_manifest_path": str(manifest_path.resolve()),
        "sample_clip_ids": [record.clip_id for record in records[:5]],
    }
    write_json_file(summary, output_root / "virtual_imu_summary.json")
    return summary


def _apply_subject_level_geometric_alignment(
    *,
    processed_runs: Sequence[Mapping[str, Any]],
    output_root: Path,
    alignment_settings: Mapping[str, Any],
) -> None:
    grouped_runs: dict[str, list[Mapping[str, Any]]] = {}
    for processed in processed_runs:
        record = processed["record"]
        grouped_runs.setdefault(_subject_id_for_record(record), []).append(processed)

    alignment_config = build_alignment_config(alignment_settings)
    for subject_id, subject_runs in grouped_runs.items():
        try:
            transforms = _fit_subject_transforms(
                subject_runs=subject_runs,
                output_root=output_root,
                alignment_config=alignment_config,
            )
            save_transforms_json(
                transforms,
                _resolve_subject_transform_path(output_root, subject_runs[0]["record"]),
            )
            for processed in subject_runs:
                _apply_subject_transforms_to_pipeline_result(
                    processed=processed,
                    output_root=output_root,
                    transforms=transforms,
                )
        except Exception as exc:
            for processed in subject_runs:
                _mark_subject_alignment_warning(
                    processed=processed,
                    output_root=output_root,
                    error=str(exc),
                    subject_id=subject_id,
                )


def _fit_subject_transforms(
    *,
    subject_runs: Sequence[Mapping[str, Any]],
    output_root: Path,
    alignment_config: Any,
):
    real_sequences = []
    virtual_sequences = []
    for processed in subject_runs:
        record = processed["record"]
        pipeline_result = processed["pipeline_result"]
        raw_virtual_sequence = pipeline_result["imusim_result"]["virtual_imu_sequence"]
        subject_id = _subject_id_for_record(record)
        real_path = _resolve_record_real_imu_npz_path(
            record=record,
            manifest_entry=processed["manifest_entry"],
            output_root=output_root,
        )
        real_sequences.append(
            load_real_imu_sequence(
                real_path,
                subject_id=subject_id,
                capture_id=record.clip_id,
            )
        )
        virtual_sequences.append(
            sequence_from_virtual_imu(
                raw_virtual_sequence,
                subject_id=subject_id,
                capture_id=record.clip_id,
            )
        )
    return fit_sensor_subject_transforms(real_sequences, virtual_sequences, alignment_config)


def _apply_subject_transforms_to_pipeline_result(
    *,
    processed: Mapping[str, Any],
    output_root: Path,
    transforms: Mapping[tuple[str, str], Any],
) -> None:
    record = processed["record"]
    manifest_entry = processed["manifest_entry"]
    pipeline_result = processed["pipeline_result"]
    pose_dir = Path(processed["pose_dir"])
    pose_dir.mkdir(parents=True, exist_ok=True)

    virtual_sequence = pipeline_result["imusim_result"]["virtual_imu_sequence"]
    real_path = _resolve_record_real_imu_npz_path(
        record=record,
        manifest_entry=manifest_entry,
        output_root=output_root,
    )
    alignment_result = run_geometric_alignment(
        virtual_sequence,
        output_dir=pose_dir,
        real_imu_npz_path=real_path,
        subject_id=_subject_id_for_record(record),
        capture_id=record.clip_id,
        transforms=transforms,
    )
    final_virtual_sequence = alignment_result["aligned_virtual_imu_sequence"]

    final_virtual_path = pose_dir / "virtual_imu.npz"
    np.savez_compressed(final_virtual_path, **final_virtual_sequence.to_npz_payload())

    original_virtual_quality = dict(pipeline_result.get("virtual_imu_quality_report", {}))
    virtual_imu_quality_report = _build_virtual_imu_quality_report(
        imu_sequence=final_virtual_sequence,
        acc_noise_std_m_s2=float(original_virtual_quality.get("acc_noise_std_m_s2", 0.0)),
        gyro_noise_std_rad_s=float(original_virtual_quality.get("gyro_noise_std_rad_s", 0.0)),
        max_acceleration_warning_m_s2=float(DEFAULT_MAX_ACCELERATION_WARNING_M_S2),
        max_gyro_warning_rad_s=float(DEFAULT_MAX_GYRO_WARNING_RAD_S),
    )
    write_json_file(virtual_imu_quality_report, pose_dir / "virtual_imu_report.json")

    merged_quality = merge_virtual_imu_quality_reports(
        pose3d_quality=dict(pipeline_result["pose3d_quality_report"]),
        ik_quality=dict(pipeline_result["ik_quality_report"]),
        virtual_imu_quality=virtual_imu_quality_report,
        geometric_alignment_quality=alignment_result["quality_report"],
        frame_alignment_quality=dict(pipeline_result.get("frame_alignment_quality_report", {})),
    )
    write_json_file(merged_quality, pose_dir / "quality_report.json")

    pipeline_result["virtual_imu_sequence"] = final_virtual_sequence
    pipeline_result["virtual_imu_quality_report"] = virtual_imu_quality_report
    pipeline_result["quality_report"] = merged_quality
    pipeline_result["geometric_alignment_result"] = alignment_result
    pipeline_result["geometric_alignment_quality_report"] = alignment_result["quality_report"]
    pipeline_result["geometrically_aligned_virtual_imu_sequence"] = geometrically_aligned_sequence
    pipeline_result["artifacts"].update(alignment_result["artifacts"])
    pipeline_result["artifacts"]["virtual_imu_npz_path"] = str(final_virtual_path.resolve())
    pipeline_result["artifacts"]["quality_report_json_path"] = str((pose_dir / "quality_report.json").resolve())
    pipeline_result["imusim_result"]["virtual_imu_sequence"] = final_virtual_sequence
    pipeline_result["imusim_result"]["calibration_report"] = calibration_report
    pipeline_result["imusim_result"]["quality_report"] = virtual_imu_quality_report
    pipeline_result["imusim_result"]["artifacts"]["virtual_imu_npz_path"] = str(final_virtual_path.resolve())
    pipeline_result["imusim_result"]["artifacts"]["virtual_imu_report_json_path"] = str(
        (pose_dir / "virtual_imu_report.json").resolve()
    )
    if calibration_report is not None:
        pipeline_result["imusim_result"]["artifacts"]["virtual_imu_calibration_report_json_path"] = str(
            (pose_dir / "virtual_imu_calibration_report.json").resolve()
        )
        pipeline_result["artifacts"]["virtual_imu_calibration_report_json_path"] = str(
            (pose_dir / "virtual_imu_calibration_report.json").resolve()
        )


def _mark_subject_alignment_warning(
    *,
    processed: Mapping[str, Any],
    output_root: Path,
    error: str,
    subject_id: str,
) -> None:
    record = processed["record"]
    manifest_entry = processed["manifest_entry"]
    pipeline_result = processed["pipeline_result"]
    pose_dir = Path(processed["pose_dir"])
    real_path = _resolve_record_real_imu_npz_path(
        record=record,
        manifest_entry=manifest_entry,
        output_root=output_root,
    )
    alignment_quality = {
        "enabled": True,
        "status": "warning",
        "subject_id": subject_id,
        "capture_id": str(record.clip_id),
        "transform_source": None,
        "real_imu_npz_path": str(Path(real_path).resolve()),
        "estimated_sensor_names": [],
        "mean_acc_corr_before": None,
        "mean_acc_corr_after": None,
        "mean_gyro_corr_before": None,
        "mean_gyro_corr_after": None,
        "notes": [f"subject_geometric_alignment_failed:{error}"],
    }
    merged_quality = merge_virtual_imu_quality_reports(
        pose3d_quality=dict(pipeline_result["pose3d_quality_report"]),
        ik_quality=dict(pipeline_result["ik_quality_report"]),
        virtual_imu_quality=dict(pipeline_result["virtual_imu_quality_report"]),
        geometric_alignment_quality=alignment_quality,
        frame_alignment_quality=dict(pipeline_result.get("frame_alignment_quality_report", {})),
    )
    write_json_file(merged_quality, pose_dir / "quality_report.json")
    pipeline_result["quality_report"] = merged_quality
    pipeline_result["geometric_alignment_quality_report"] = alignment_quality
    pipeline_result["geometric_alignment_result"] = {
        "status": "warning",
        "enabled": False,
        "aligned_virtual_imu_sequence": pipeline_result["virtual_imu_sequence"],
        "transforms": {},
        "metrics_before": {},
        "metrics_after": {},
        "quality_report": alignment_quality,
        "artifacts": {
            "virtual_imu_geometric_aligned_npz_path": None,
            "imu_alignment_transforms_json_path": None,
            "imu_alignment_metrics_json_path": None,
            "imu_alignment_quality_report_json_path": None,
            "imu_alignment_config_path": None,
        },
    }
    pipeline_result["artifacts"]["quality_report_json_path"] = str((pose_dir / "quality_report.json").resolve())


def _resolve_record_real_imu_npz_path(
    *,
    record: RobotEmotionsClipRecord,
    manifest_entry: Mapping[str, Any],
    output_root: Path,
) -> Path:
    artifact_path = dict(manifest_entry.get("artifacts", {})).get("imu_npz_path")
    if artifact_path in (None, ""):
        raise ValueError(f"Missing input artifact imu_npz_path for clip {record.clip_id}")
    return Path(str(artifact_path))


def _resolve_subject_transform_path(output_root: Path, record: RobotEmotionsClipRecord) -> Path:
    return output_root / record.domain / f"user_{record.user_id:02d}" / "imu_alignment_transforms.json"


def _subject_id_for_record(record: RobotEmotionsClipRecord) -> str:
    return f"{record.domain}_user_{record.user_id:02d}"


def _build_virtual_imu_manifest_entry(
    *,
    record: RobotEmotionsClipRecord,
    manifest_entry: Mapping[str, Any],
    pipeline_result: Mapping[str, Any],
) -> Dict[str, Any]:
    pose_sequence_3d = pipeline_result["pose_sequence"]
    virtual_imu_sequence = pipeline_result["virtual_imu_sequence"]
    quality_report = dict(pipeline_result["quality_report"])
    artifacts = dict(pipeline_result["artifacts"])
    input_artifacts = dict(manifest_entry.get("artifacts", {}))
    return {
        "clip_id": str(record.clip_id),
        "dataset": "RobotEmotions",
        "status": str(quality_report["status"]),
        "domain": str(record.domain),
        "user_id": int(record.user_id),
        "tag_number": int(record.tag_number),
        "take_id": record.take_id,
        "labels": dict(manifest_entry.get("labels", {})),
        "source": dict(manifest_entry.get("source", {})),
        "video": dict(manifest_entry.get("video", {})),
        "input_artifacts": input_artifacts,
        "pose3d": {
            "fps": None if pose_sequence_3d.fps is None else float(pose_sequence_3d.fps),
            "fps_original": (
                None if pose_sequence_3d.fps_original is None else float(pose_sequence_3d.fps_original)
            ),
            "num_frames": int(pose_sequence_3d.num_frames),
            "num_joints": int(pose_sequence_3d.num_joints),
            "joint_names_3d": list(pose_sequence_3d.joint_names_3d),
            "source": str(pose_sequence_3d.source),
            "coordinate_space": str(pose_sequence_3d.coordinate_space),
        },
        "virtual_imu": {
            "fps": None if virtual_imu_sequence.fps is None else float(virtual_imu_sequence.fps),
            "num_frames": int(virtual_imu_sequence.num_frames),
            "num_sensors": int(virtual_imu_sequence.num_sensors),
            "sensor_names": list(virtual_imu_sequence.sensor_names),
            "source": str(virtual_imu_sequence.source),
        },
        "quality_report": quality_report,
        "pose3d_quality_report": dict(pipeline_result["pose3d_quality_report"]),
        "ik_quality_report": dict(pipeline_result["ik_quality_report"]),
        "virtual_imu_quality_report": dict(pipeline_result["virtual_imu_quality_report"]),
        "artifacts": artifacts,
    }


def _build_virtual_imu_failure_entry(
    *,
    record: RobotEmotionsClipRecord,
    manifest_entry: Mapping[str, Any],
    pose_dir: Path,
    error: str,
) -> Dict[str, Any]:
    pose_dir.mkdir(parents=True, exist_ok=True)
    failure_quality = {
        "clip_id": str(record.clip_id),
        "status": "fail",
        "notes": [str(error)],
    }
    write_json_file(failure_quality, pose_dir / "quality_report.json")
    return {
        "clip_id": str(record.clip_id),
        "dataset": "RobotEmotions",
        "status": "fail",
        "domain": str(record.domain),
        "user_id": int(record.user_id),
        "tag_number": int(record.tag_number),
        "take_id": record.take_id,
        "labels": dict(manifest_entry.get("labels", {})),
        "source": dict(manifest_entry.get("source", {})),
        "video": dict(manifest_entry.get("video", {})),
        "quality_report": failure_quality,
        "artifacts": {
            "quality_report_json_path": str((pose_dir / "quality_report.json").resolve()),
            "pose3d_npz_path": str((pose_dir / "pose3d.npz").resolve()) if (pose_dir / "pose3d.npz").exists() else None,
            "virtual_imu_npz_path": str((pose_dir / "virtual_imu.npz").resolve()) if (pose_dir / "virtual_imu.npz").exists() else None,
            "ik_sequence_npz_path": str((pose_dir / "ik_sequence.npz").resolve()) if (pose_dir / "ik_sequence.npz").exists() else None,
            "ik_bvh_path": str((pose_dir / "pose3d_ik.bvh").resolve()) if (pose_dir / "pose3d_ik.bvh").exists() else None,
        },
    }


def calibrate_virtual_imu_manifest(
    manifest_path: str,
    real_imu_reference_path: str,
    *,
    signal_mode: str = DEFAULT_CALIBRATION_SIGNAL_MODE,
    percentile_resolution: int = DEFAULT_CALIBRATION_PERCENTILE_RESOLUTION,
    per_class: bool = True,
    calibration_fraction: float = 1.0,
    activity_label_key: Optional[str] = None,
    in_place: bool = False,
    output_suffix: str = "_calibrated",
) -> Dict[str, Any]:
    """Apply percentile calibration to all virtual IMU clips listed in a manifest.

    Reads each ``virtual_imu_npz_path`` from the manifest, calibrates it against
    the real IMU reference, and writes the result either in-place (overwriting
    ``virtual_imu.npz``) or alongside the original (``virtual_imu_calibrated.npz``).

    ``real_imu_reference_path`` can be:
    - A single NPZ file — used directly as reference.
    - A directory — all ``*.npz`` files inside are collected and concatenated
      after applying ``calibration_fraction`` per clip.

    Returns a summary dict with per-clip status.
    """
    from pose_module.interfaces import VirtualIMUSequence

    manifest_path = Path(manifest_path)
    ref_path = Path(real_imu_reference_path)

    entries = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Build reference matrix once for all clips
    if ref_path.is_dir():
        clip_npz_paths = sorted(ref_path.rglob("*.npz"))
        if not clip_npz_paths:
            raise ValueError(f"No NPZ files found in reference directory: {ref_path}")
        ref_matrix = build_calibration_reference_matrix(
            clip_npz_paths=clip_npz_paths,
            target_sensor_names=[],  # resolved per-clip inside calibrate_virtual_imu_sequence
            signal_mode=signal_mode,
            calibration_fraction=calibration_fraction,
        )
        # Save to a temp NPZ so calibrate_virtual_imu_sequence can load it
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        tmp.close()
        np.savez(tmp.name, imu=ref_matrix)
        resolved_ref_path = tmp.name
        _tmp_to_delete = tmp.name
    else:
        resolved_ref_path = str(ref_path)
        _tmp_to_delete = None

    results = []
    num_ok = num_warning = num_fail = 0

    try:
        for entry in entries:
            clip_id = str(entry.get("clip_id", ""))
            artifacts = entry.get("artifacts", {})
            virtual_imu_path = artifacts.get("virtual_imu_npz_path")

            if not virtual_imu_path or not Path(virtual_imu_path).exists():
                results.append({"clip_id": clip_id, "status": "skip", "reason": "virtual_imu_npz_path missing or not found"})
                num_fail += 1
                continue

            activity_label = None
            if activity_label_key not in (None, ""):
                activity_label = entry.get("labels", {}).get(str(activity_label_key))

            try:
                with np.load(virtual_imu_path, allow_pickle=True) as payload:
                    seq = VirtualIMUSequence(
                        clip_id=clip_id,
                        fps=float(payload["fps"]) if "fps" in payload and payload["fps"] is not None else None,
                        sensor_names=[str(s) for s in np.asarray(payload["sensor_names"]).reshape(-1).tolist()],
                        acc=np.asarray(payload["acc"], dtype=np.float32),
                        gyro=np.asarray(payload["gyro"], dtype=np.float32),
                        timestamps_sec=np.asarray(payload["timestamps_sec"], dtype=np.float32),
                        source=str(payload.get("source", "virtual_imu")),
                    )

                result = calibrate_virtual_imu_sequence(
                    seq,
                    real_imu_reference_path=resolved_ref_path,
                    activity_label=activity_label,
                    signal_mode=signal_mode,
                    percentile_resolution=percentile_resolution,
                    per_class=per_class,
                )
                calibrated_seq = result["virtual_imu_sequence"]
                report = result["calibration_report"]

                if in_place:
                    out_path = Path(virtual_imu_path)
                else:
                    stem = Path(virtual_imu_path).stem
                    out_path = Path(virtual_imu_path).with_name(f"{stem}{output_suffix}.npz")

                np.savez_compressed(out_path, **calibrated_seq.to_npz_payload())

                report_path = out_path.with_suffix("").with_name(out_path.stem + "_calibration_report.json")
                write_json_file(report, report_path)

                status = str(report.get("status", "ok"))
                results.append({
                    "clip_id": clip_id,
                    "status": status,
                    "output_npz_path": str(out_path.resolve()),
                    "calibration_report_path": str(report_path.resolve()),
                })
                if status == "ok":
                    num_ok += 1
                else:
                    num_warning += 1
            except Exception as exc:
                results.append({"clip_id": clip_id, "status": "fail", "reason": str(exc)})
                num_fail += 1
    finally:
        if _tmp_to_delete is not None:
            os.unlink(_tmp_to_delete)

    return {
        "manifest_path": str(manifest_path.resolve()),
        "real_imu_reference_path": str(ref_path.resolve()),
        "signal_mode": signal_mode,
        "percentile_resolution": percentile_resolution,
        "per_class": per_class,
        "calibration_fraction": calibration_fraction,
        "in_place": in_place,
        "num_clips": len(entries),
        "num_ok": num_ok,
        "num_warning": num_warning,
        "num_fail": num_fail,
        "clips": results,
    }
