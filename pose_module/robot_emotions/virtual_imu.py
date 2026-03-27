"""RobotEmotions-specific wrapper around the full virtual-IMU pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from pose_module.io.cache import write_json_file
from pose_module.pipeline import run_virtual_imu_pipeline

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
    real_imu_reference_path: Optional[str] = None,
    real_imu_label_key: Optional[str] = None,
    real_imu_signal_mode: str = "acc",
    real_imu_percentile_resolution: int = 100,
    real_imu_per_class_calibration: bool = True,
    domains: Sequence[str] = ("10ms", "30ms"),
) -> Dict[str, Any]:
    extractor = RobotEmotionsExtractor(dataset_root, domains=tuple(str(domain) for domain in domains))
    records = extractor.select_records(
        clip_ids=None if clip_ids is None else list(clip_ids),
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    num_ok = 0
    num_warning = 0
    num_fail = 0
    for record in records:
        manifest_entry = extractor.ensure_exported_clip(record, output_root=output_root)
        activity_label = None
        if real_imu_label_key not in (None, ""):
            activity_label = manifest_entry.get("labels", {}).get(str(real_imu_label_key))
        try:
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
                real_imu_reference_path=(
                    None if real_imu_reference_path in (None, "") else str(real_imu_reference_path)
                ),
                real_imu_signal_mode=str(real_imu_signal_mode),
                real_imu_percentile_resolution=int(real_imu_percentile_resolution),
                real_imu_per_class_calibration=bool(real_imu_per_class_calibration),
            )
            entry = _build_virtual_imu_manifest_entry(
                record=record,
                manifest_entry=manifest_entry,
                pipeline_result=pipeline_result,
            )
            status = str(entry.get("status", "fail"))
            if status == "ok":
                num_ok += 1
            elif status == "warning":
                num_warning += 1
            else:
                num_fail += 1
        except Exception as exc:
            entry = _build_virtual_imu_failure_entry(
                record=record,
                manifest_entry=manifest_entry,
                pose_dir=resolve_pose_output_dir(output_root, record),
                error=str(exc),
            )
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
        "real_imu_reference_path": (
            None if real_imu_reference_path in (None, "") else str(Path(real_imu_reference_path).resolve())
        ),
        "real_imu_label_key": None if real_imu_label_key in (None, "") else str(real_imu_label_key),
        "sample_clip_ids": [record.clip_id for record in records[:5]],
    }
    write_json_file(summary, output_root / "virtual_imu_summary.json")
    return summary


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
        "input_artifacts": dict(manifest_entry.get("artifacts", {})),
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
