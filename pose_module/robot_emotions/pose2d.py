"""RobotEmotions-specific wrapper around the generic pose pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from pose_module.pipeline import run_pose2d_pipeline
from pose_module.io.cache import write_json_file

from .extractor import (
    RobotEmotionsClipRecord,
    RobotEmotionsExtractor,
    resolve_pose_output_dir,
)


def run_robot_emotions_pose2d(
    dataset_root: str,
    output_dir: str,
    *,
    fps_target: int = 20,
    clip_ids: Optional[Sequence[str]] = None,
    save_debug: bool = True,
    env_name: str = "openmmlab",
    domains: Sequence[str] = ("10ms", "30ms"),
) -> Dict[str, Any]:
    extractor = RobotEmotionsExtractor(dataset_root, domains=tuple(str(domain) for domain in domains))
    records = extractor.select_records(
        clip_ids=None if clip_ids is None else list(clip_ids),
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    pose_manifest_entries = []
    num_ok = 0
    num_warning = 0
    num_fail = 0
    for record in records:
        manifest_entry = extractor.ensure_exported_clip(record, output_root=output_root)
        try:
            pipeline_result = run_pose2d_pipeline(
                clip_id=record.clip_id,
                video_path=str(record.video_path.resolve()),
                output_dir=resolve_pose_output_dir(output_root, record),
                fps_target=int(fps_target),
                save_debug=bool(save_debug),
                env_name=str(env_name),
                video_metadata=manifest_entry.get("video", {}),
                model_alias="vitpose-b",
            )
            pose_entry = _build_pose_manifest_entry(
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
            pose_entry = _build_pose_failure_entry(
                record=record,
                manifest_entry=manifest_entry,
                pose_dir=resolve_pose_output_dir(output_root, record),
                error=str(exc),
            )
            num_fail += 1
        pose_manifest_entries.append(pose_entry)

    pose_manifest_path = output_root / "pose_manifest.jsonl"
    with pose_manifest_path.open("w", encoding="utf-8") as handle:
        for entry in pose_manifest_entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    pose_summary = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "output_dir": str(output_root.resolve()),
        "domains": [str(domain) for domain in domains],
        "num_requested_clips": int(len(records)),
        "num_pose_entries": int(len(pose_manifest_entries)),
        "num_ok": int(num_ok),
        "num_warning": int(num_warning),
        "num_fail": int(num_fail),
        "pose_manifest_path": str(pose_manifest_path.resolve()),
        "sample_clip_ids": [record.clip_id for record in records[:5]],
    }
    write_json_file(pose_summary, output_root / "pose_summary.json")
    return pose_summary


def _build_pose_manifest_entry(
    *,
    record: RobotEmotionsClipRecord,
    manifest_entry: Mapping[str, Any],
    pipeline_result: Mapping[str, Any],
) -> Dict[str, Any]:
    pose_sequence = pipeline_result["pose_sequence"]
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
        "pose2d": {
            "fps": None if pose_sequence.fps is None else float(pose_sequence.fps),
            "fps_original": None if pose_sequence.fps_original is None else float(pose_sequence.fps_original),
            "num_frames": int(pose_sequence.num_frames),
            "joint_names_2d": list(pose_sequence.joint_names_2d),
            "source": str(pose_sequence.source),
        },
        "quality_report": quality_report,
        "artifacts": dict(artifacts),
    }


def _build_pose_failure_entry(
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
            "quality_report_json_path": str((pose_dir / "quality_report.json").resolve()),
            "raw_prediction_json_path": str((pose_dir / "raw_predictions.json").resolve())
            if (pose_dir / "raw_predictions.json").exists()
            else None,
            "debug_overlay_path": str((pose_dir / "debug_overlay.mp4").resolve())
            if (pose_dir / "debug_overlay.mp4").exists()
            else None,
        },
    }
