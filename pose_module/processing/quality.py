"""Quality-report helpers for the pose pipeline."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def merge_stage53_quality_reports(
    *,
    clip_id: str,
    backend_quality: Mapping[str, Any],
    track_report: Mapping[str, Any],
    pose_quality: Mapping[str, Any],
) -> Dict[str, Any]:
    notes = []
    notes.extend([str(value) for value in backend_quality.get("warnings", [])])
    notes.extend([str(value) for value in track_report.get("warnings", [])])
    notes.extend([str(value) for value in pose_quality.get("notes", [])])

    status = "ok"
    if track_report.get("status") == "fail" or pose_quality.get("status") == "fail":
        status = "fail"
    elif (
        track_report.get("status") == "warning"
        or pose_quality.get("status") == "warning"
        or len(notes) > 0
    ):
        status = "warning"

    return {
        "clip_id": str(clip_id),
        "status": str(status),
        "fps_original": backend_quality.get("fps_original", pose_quality.get("fps_original")),
        "fps": backend_quality.get("effective_fps", pose_quality.get("fps")),
        "frames_total": backend_quality.get("frames_total"),
        "frames_selected": backend_quality.get("frames_selected", pose_quality.get("num_selected_frames")),
        "frames_with_detections": track_report.get("frames_with_detections"),
        "frames_with_selected_track": pose_quality.get("frames_with_selected_track"),
        "selected_track_id": track_report.get("selected_track_id"),
        "selected_track_stability": track_report.get("selected_track_stability"),
        "visible_joint_ratio": pose_quality.get("visible_joint_ratio"),
        "mean_confidence": pose_quality.get("mean_confidence"),
        "notes": list(dict.fromkeys(notes)),
    }
