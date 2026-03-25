"""RobotEmotions-specific wrapper around the generic 3D pose pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from pose_module.io.cache import write_json_file
from pose_module.pipeline import run_pose3d_pipeline

from .extractor import (
    RobotEmotionsClipRecord,
    RobotEmotionsExtractor,
    resolve_pose_output_dir,
)


def run_robot_emotions_pose3d(
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
    domains: Sequence[str] = ("10ms", "30ms"),
) -> Dict[str, Any]:
    extractor = RobotEmotionsExtractor(dataset_root, domains=tuple(str(domain) for domain in domains))
    records = extractor.select_records(
        clip_ids=None if clip_ids is None else list(clip_ids),
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    pose3d_manifest_entries = []
    num_ok = 0
    num_warning = 0
    num_fail = 0
    for record in records:
        manifest_entry = extractor.ensure_exported_clip(record, output_root=output_root)
        try:
            pipeline_result = run_pose3d_pipeline(
                clip_id=record.clip_id,
                video_path=str(record.video_path.resolve()),
                output_dir=resolve_pose_output_dir(output_root, record),
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
            )
            pose_entry = _build_pose3d_manifest_entry(
                record=record,
                manifest_entry=manifest_entry,
                pipeline_result=pipeline_result,
            )
            status = str(pose_entry.get("status", "fail"))
            if status == "ok":
                num_ok += 1
            elif status == "warning":
                num_warning += 1
            else:
                num_fail += 1
        except Exception as exc:
            pose_entry = _build_pose3d_failure_entry(
                record=record,
                manifest_entry=manifest_entry,
                pose_dir=resolve_pose_output_dir(output_root, record),
                error=str(exc),
            )
            num_fail += 1
        pose3d_manifest_entries.append(pose_entry)

    pose3d_manifest_path = output_root / "pose3d_manifest.jsonl"
    with pose3d_manifest_path.open("w", encoding="utf-8") as handle:
        for entry in pose3d_manifest_entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    pose3d_summary = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "output_dir": str(output_root.resolve()),
        "domains": [str(domain) for domain in domains],
        "num_requested_clips": int(len(records)),
        "num_pose3d_entries": int(len(pose3d_manifest_entries)),
        "num_ok": int(num_ok),
        "num_warning": int(num_warning),
        "num_fail": int(num_fail),
        "pose3d_manifest_path": str(pose3d_manifest_path.resolve()),
        "sample_clip_ids": [record.clip_id for record in records[:5]],
    }
    write_json_file(pose3d_summary, output_root / "pose3d_summary.json")
    return pose3d_summary


def _build_pose3d_manifest_entry(
    *,
    record: RobotEmotionsClipRecord,
    manifest_entry: Mapping[str, Any],
    pipeline_result: Mapping[str, Any],
) -> Dict[str, Any]:
    pose_sequence_2d = pipeline_result["pose_sequence_2d"]
    pose_sequence_3d = pipeline_result["pose_sequence"]
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
        "pose2d": {
            "fps": None if pose_sequence_2d.fps is None else float(pose_sequence_2d.fps),
            "fps_original": None if pose_sequence_2d.fps_original is None else float(pose_sequence_2d.fps_original),
            "num_frames": int(pose_sequence_2d.num_frames),
            "joint_names_2d": list(pose_sequence_2d.joint_names_2d),
            "source": str(pose_sequence_2d.source),
        },
        "pose3d": {
            "fps": None if pose_sequence_3d.fps is None else float(pose_sequence_3d.fps),
            "fps_original": None if pose_sequence_3d.fps_original is None else float(pose_sequence_3d.fps_original),
            "num_frames": int(pose_sequence_3d.num_frames),
            "num_joints": int(pose_sequence_3d.num_joints),
            "joint_names_3d": list(pose_sequence_3d.joint_names_3d),
            "source": str(pose_sequence_3d.source),
            "coordinate_space": str(pose_sequence_3d.coordinate_space),
        },
        "quality_report": quality_report,
        "pose2d_quality_report": dict(pipeline_result["pose2d_quality_report"]),
        "motionbert_quality_report": dict(pipeline_result["motionbert_quality_report"]),
        "artifacts": artifacts,
    }


def _build_pose3d_failure_entry(
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
            "backend_run_json_path": str((pose_dir / "backend_run.json").resolve())
            if (pose_dir / "backend_run.json").exists()
            else None,
            "motionbert_run_json_path": str((pose_dir / "motionbert_run.json").resolve())
            if (pose_dir / "motionbert_run.json").exists()
            else None,
            "pose2d_npz_path": str((pose_dir / "pose2d.npz").resolve())
            if (pose_dir / "pose2d.npz").exists()
            else None,
            "pose3d_npz_path": str((pose_dir / "pose3d.npz").resolve())
            if (pose_dir / "pose3d.npz").exists()
            else None,
            "quality_report_json_path": str((pose_dir / "quality_report.json").resolve()),
            "raw_prediction_json_path": str((pose_dir / "raw_predictions.json").resolve())
            if (pose_dir / "raw_predictions.json").exists()
            else None,
            "debug_overlay_path": str((pose_dir / "debug_overlay.mp4").resolve())
            if (pose_dir / "debug_overlay.mp4").exists()
            else None,
            "debug_overlay_pose3d_raw_path": str((pose_dir / "debug_overlay_pose3d_raw.mp4").resolve())
            if (pose_dir / "debug_overlay_pose3d_raw.mp4").exists()
            else None,
        },
    }
