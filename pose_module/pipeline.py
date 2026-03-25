"""Top-level orchestration for the pose pipeline stages implemented so far."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from pose_module.interfaces import Pose2DJob
from pose_module.io.cache import write_json_file
from pose_module.io.video_loader import frame_indices_to_timestamps, select_frame_indices
from pose_module.processing.quality import merge_stage53_quality_reports
from pose_module.tracking.person_selector import build_person_track_report, link_person_tracks
from pose_module.vitpose.adapter import (
    canonicalize_pose_sequence2d,
    load_raw_prediction_frames,
    write_pose_sequence_npz,
)
from pose_module.vitpose.estimator import run_backend_job


def run_pose2d_pipeline(
    *,
    clip_id: str,
    video_path: str,
    output_dir: str | Path,
    fps_target: int = 20,
    save_debug: bool = True,
    env_name: str = "auto",
    video_metadata: Optional[Mapping[str, Any]] = None,
    model_alias: str = "vitpose-b",
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_metadata = {} if video_metadata is None else dict(video_metadata)

    job = Pose2DJob(
        clip_id=str(clip_id),
        video_path=str(Path(video_path).resolve()),
        fps_target=int(fps_target),
        output_dir=str(output_dir.resolve()),
        save_debug=bool(save_debug),
        device_preference="auto",
        model_alias=str(model_alias),
        detector_category_ids=(0,),
        video_fps=_optional_float(video_metadata.get("fps")),
        video_num_frames=_optional_int(video_metadata.get("num_frames")),
        video_duration_sec=_optional_float(video_metadata.get("duration_sec")),
    )
    backend_run = run_backend_job(job=job, env_name=env_name, output_dir=output_dir)
    if backend_run.get("status") != "ok":
        raise RuntimeError(str(backend_run.get("error", "pose2d_backend_failed")))

    raw_prediction_json_path = backend_run["artifacts"]["raw_prediction_json_path"]
    frame_predictions = load_raw_prediction_frames(raw_prediction_json_path)
    tracks = link_person_tracks(frame_predictions)
    selected_track = tracks[0] if len(tracks) > 0 else None
    track_report = build_person_track_report(
        tracks,
        selected_track=selected_track,
        total_frames=int(len(frame_predictions)),
    )
    write_json_file(track_report, output_dir / "person_track.json")

    if selected_track is None:
        raise RuntimeError("No valid person track produced from backend predictions.")

    selected_frame_indices = np.asarray(backend_run.get("selected_frame_indices", []), dtype=np.int32)
    if selected_frame_indices.size == 0:
        selected_frame_indices, effective_fps, _ = select_frame_indices(
            len(frame_predictions),
            _optional_float(video_metadata.get("fps")),
            int(fps_target),
        )
    else:
        effective_fps = _optional_float(backend_run.get("effective_fps"))
    timestamps_sec = frame_indices_to_timestamps(
        selected_frame_indices,
        _optional_float(video_metadata.get("fps")),
    )

    pose_sequence, pose_quality = canonicalize_pose_sequence2d(
        clip_id=str(clip_id),
        selected_track=selected_track,
        selected_frame_indices=selected_frame_indices,
        timestamps_sec=timestamps_sec,
        effective_fps=effective_fps,
        fps_original=_optional_float(video_metadata.get("fps")),
        source=str(model_alias),
    )
    merged_quality = merge_stage53_quality_reports(
        clip_id=str(clip_id),
        backend_quality=backend_run.get("quality_report", {}),
        track_report=track_report,
        pose_quality=pose_quality,
    )
    write_json_file(merged_quality, output_dir / "quality_report.json")
    write_pose_sequence_npz(pose_sequence, output_dir / "pose2d.npz")

    return {
        "clip_id": str(clip_id),
        "pose_sequence": pose_sequence,
        "quality_report": merged_quality,
        "track_report": track_report,
        "backend_run": backend_run,
        "artifacts": {
            "pose2d_npz_path": str((output_dir / "pose2d.npz").resolve()),
            "person_track_json_path": str((output_dir / "person_track.json").resolve()),
            "quality_report_json_path": str((output_dir / "quality_report.json").resolve()),
            "backend_run_json_path": str((output_dir / "backend_run.json").resolve()),
            "raw_prediction_json_path": str(Path(raw_prediction_json_path).resolve()),
            "debug_overlay_path": backend_run["artifacts"].get("debug_overlay_path"),
        },
    }


def _optional_float(raw_value: Any) -> Optional[float]:
    if raw_value in (None, ""):
        return None
    return float(raw_value)


def _optional_int(raw_value: Any) -> Optional[int]:
    if raw_value in (None, ""):
        return None
    return int(raw_value)
