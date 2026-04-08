"""Top-level orchestration for the pose pipeline stages"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from pose_module.export.bvh import export_pose_sequence3d_to_bvh
from pose_module.export.ik_adapter import run_ik
from pose_module.export.imusim_adapter import (
    DEFAULT_IMU_CALIBRATION_REPORT_FILENAME,
    DEFAULT_IMU_REPORT_FILENAME,
    DEFAULT_MAX_ACCELERATION_WARNING_M_S2,
    DEFAULT_MAX_GYRO_WARNING_RAD_S,
    DEFAULT_RAW_IMU_SEQUENCE_FILENAME,
    DEFAULT_SENSOR_LAYOUT_PATH,
    _build_virtual_imu_quality_report,
    run_imusim
)

from pose_module.export.debug_video import render_pose_overlay_video, render_pose3d_side_by_side_video, resolve_debug_overlay_variant_path
from pose_module.imu_alignment import load_alignment_runtime_settings, run_geometric_alignment
from pose_module.interfaces import Pose2DJob, PoseSequence2D, PoseSequence3D, VirtualIMUSequence
from pose_module.io.cache import write_json_file

from pose_module.motionbert.adapter import write_pose_sequence3d_npz
from pose_module.motionbert.lifter import MotionBERTPredictor, run_motionbert_lifter
from pose_module.prompt_source import (
    LegacyT2MGPTBackend,
    PromptMotionBackend,
    build_prompt_adapter_quality_report,
    build_prompt_metadata,
    build_prompt_pose_quality_report,
    build_prompt_pose_sequence3d,
)

from pose_module.io.video_loader import frame_indices_to_timestamps, select_frame_indices
from pose_module.processing.cleaner2d import clean_pose_sequence2d
from pose_module.processing.lower_limb_stabilizer import run_lower_limb_stabilizer
from pose_module.processing.metric_normalizer import run_metric_normalizer
from pose_module.processing.imu_calibration import calibrate_virtual_imu_sequence
from pose_module.processing.quality import merge_pose2d_quality_reports, merge_pose3d_quality_reports, merge_virtual_imu_quality_reports

from pose_module.processing.root_estimator import run_root_trajectory_estimator
from pose_module.processing.sensor_frame_estimation import DEFAULT_TARGET_SENSOR_NAMES, estimate_sensor_frame_alignment

from pose_module.processing.skeleton_mapper import map_pose_sequence_to_imugpt22
from pose_module.tracking.person_selector import build_person_track_report, link_person_tracks
from pose_module.vitpose.estimator import run_backend_job
from pose_module.vitpose.adapter import canonicalize_pose_sequence2d, load_raw_prediction_frames, write_pose_sequence_npz

def run_pose2d_pipeline(*, clip_id: str, video_path: str, output_dir: str | Path, fps_target: int = 20, save_debug: bool = True, env_name: str = "openmmlab", video_metadata: Optional[Mapping[str, Any]] = None, model_alias: str = "vitpose-b",) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pose2d_output_dir = _resolve_stage_output_dir(output_dir, "pose2d")
    video_metadata = {} if video_metadata is None else dict(video_metadata)

    job = Pose2DJob(
        clip_id=str(clip_id),
        video_path=str(Path(video_path).resolve()),
        fps_target=int(fps_target),
        output_dir=str(pose2d_output_dir.resolve()),
        save_debug=bool(save_debug),
        device_preference="auto",
        model_alias=str(model_alias),
        detector_category_ids=(0,),
        video_fps=_optional_float(video_metadata.get("fps")),
        video_num_frames=_optional_int(video_metadata.get("num_frames")),
        video_duration_sec=_optional_float(video_metadata.get("duration_sec")),
    )
    
    # Run the 2D backend and recover a single person track.
    backend_run = run_backend_job(job=job, env_name=env_name, output_dir=pose2d_output_dir)
    if backend_run.get("status") != "ok":
        raise RuntimeError(str(backend_run.get("error", "pose2d_backend_failed")))

    raw_prediction_json_path = backend_run["artifacts"]["raw_prediction_json_path"]
    frame_predictions = load_raw_prediction_frames(raw_prediction_json_path)
    tracks = link_person_tracks(frame_predictions)
    selected_track = tracks[0] if len(tracks) > 0 else None
    track_report = build_person_track_report(tracks, selected_track=selected_track, total_frames=int(len(frame_predictions)))
    write_json_file(track_report, pose2d_output_dir / "person_track.json")

    if selected_track is None:
        raise RuntimeError("No valid person track produced from backend predictions.")

    # Rebuild timing information for the sampled frames.
    selected_frame_indices = np.asarray(backend_run.get("selected_frame_indices", []), dtype=np.int32)
    
    if selected_frame_indices.size == 0:
        selected_frame_indices, effective_fps, _ = select_frame_indices(len(frame_predictions), _optional_float(video_metadata.get("fps")), int(fps_target))
    else:
        effective_fps = _optional_float(backend_run.get("effective_fps"))
        
    timestamps_sec = frame_indices_to_timestamps(selected_frame_indices, _optional_float(video_metadata.get("fps")))

    # Canonicalize the poses and clean unstable detections.
    raw_pose_sequence, pose_quality = canonicalize_pose_sequence2d(
        clip_id=str(clip_id),
        selected_track=selected_track,
        selected_frame_indices=selected_frame_indices,
        timestamps_sec=timestamps_sec,
        effective_fps=effective_fps,
        fps_original=_optional_float(video_metadata.get("fps")),
        source=str(model_alias),
    )
    
    pose_sequence, cleaner_quality, cleaner_artifacts = clean_pose_sequence2d(raw_pose_sequence, track_report=track_report)
    merged_quality = merge_pose2d_quality_reports(
        clip_id=str(clip_id),
        backend_quality=backend_run.get("quality_report", {}),
        track_report=track_report,
        pose_quality=pose_quality,
        cleaner_quality=cleaner_quality,
    )
    
    raw_debug_overlay_path = None
    clean_debug_overlay_path = None
    
    if save_debug:
        # Render quick visual checks for the raw and cleaned poses.
        raw_debug_overlay_path = _render_debug_overlay_variant(
            video_path=str(video_path),
            output_path=resolve_debug_overlay_variant_path(pose2d_output_dir, variant="raw", enabled=True),
            pose_sequence=raw_pose_sequence,
            keypoints_xy=np.asarray(raw_pose_sequence.keypoints_xy, dtype=np.float32),
            overlay_variant="raw",
            merged_quality=merged_quality,
        )
        clean_debug_overlay_path = _render_debug_overlay_variant(
            video_path=str(video_path),
            output_path=resolve_debug_overlay_variant_path(pose2d_output_dir, variant="clean", enabled=True),
            pose_sequence=pose_sequence,
            keypoints_xy=_restore_clean_pose_pixels(
                pose_sequence.keypoints_xy,
                cleaner_artifacts["normalization_centers_xy"],
                cleaner_artifacts["normalization_scales"],
                pose_sequence.confidence,
            ),
            overlay_variant="clean",
            merged_quality=merged_quality,
        )

    # Persist the cleaned sequence and its reports.
    write_json_file(merged_quality, pose2d_output_dir / "quality_report.json")
    np.save(pose2d_output_dir / "2d_keypoints_raw.npy", np.asarray(raw_pose_sequence.keypoints_xy, dtype=np.float32))
    np.save(pose2d_output_dir / "2d_keypoints_clean.npy", np.asarray(pose_sequence.keypoints_xy, dtype=np.float32))
    write_pose_sequence_npz(pose_sequence, pose2d_output_dir / "pose2d.npz")

    return {
        "clip_id": str(clip_id),
        "pose_sequence": pose_sequence,
        "raw_pose_sequence": raw_pose_sequence,
        "quality_report": merged_quality,
        "cleaner_quality": cleaner_quality,
        "track_report": track_report,
        "backend_run": backend_run,
        "cleaner_artifacts": cleaner_artifacts,
        "artifacts": {
            "pose2d_npz_path": str((pose2d_output_dir / "pose2d.npz").resolve()),
            "pose2d_raw_keypoints_path": str((pose2d_output_dir / "2d_keypoints_raw.npy").resolve()),
            "pose2d_clean_keypoints_path": str((pose2d_output_dir / "2d_keypoints_clean.npy").resolve()),
            "person_track_json_path": str((pose2d_output_dir / "person_track.json").resolve()),
            "quality_report_json_path": str((pose2d_output_dir / "quality_report.json").resolve()),
            "backend_run_json_path": str((pose2d_output_dir / "backend_run.json").resolve()),
            "raw_prediction_json_path": str(Path(raw_prediction_json_path).resolve()),
            "debug_overlay_path": backend_run["artifacts"].get("debug_overlay_path"),
            "debug_overlay_raw_path": None if raw_debug_overlay_path is None else str(raw_debug_overlay_path),
            "debug_overlay_clean_path": None if clean_debug_overlay_path is None else str(clean_debug_overlay_path),
        },
    }


def run_pose3d_pipeline(
        *,
        clip_id: str,
        video_path: str,
        output_dir: str | Path,
        fps_target: int = 20,
        save_debug: bool = True,
        save_debug_2d: Optional[bool] = None,
        save_debug_3d: Optional[bool] = None,
        env_name: str = "openmmlab",
        video_metadata: Optional[Mapping[str, Any]] = None,
        model_alias: str = "vitpose-b",
        motionbert_window_size: int = 81,
        motionbert_window_overlap: float = 0.5,
        include_motionbert_confidence: bool = True,
        motionbert_predictor: Optional[MotionBERTPredictor] = None,
        motionbert_backend_name: Optional[str] = None,
        motionbert_env_name: Optional[str] = None,
        motionbert_config_path: Optional[str] = None,
        motionbert_checkpoint_path: Optional[str] = None,
        motionbert_device: str = "auto",
        allow_motionbert_fallback_backend: bool = False) -> Dict[str, Any]:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pose3d_output_dir = _resolve_stage_output_dir(output_dir, "pose3d")
    motionbert_output_dir = _resolve_stage_output_dir(output_dir, "motionbert")
    save_debug_2d = bool(save_debug if save_debug_2d is None else save_debug_2d)
    save_debug_3d = bool(save_debug if save_debug_3d is None else save_debug_3d)

    pose2d_result = run_pose2d_pipeline(
        clip_id=str(clip_id),
        video_path=str(video_path),
        output_dir=output_dir,
        fps_target=int(fps_target),
        save_debug=bool(save_debug_2d),
        env_name=str(env_name),
        video_metadata=video_metadata,
        model_alias=str(model_alias),
    )
    # Feed the cleaned 2D sequence into the 3D lifter.
    motionbert_input_sequence = _build_motionbert_backend_input_sequence(pose2d_result)

    # Lift image-space keypoints into a raw 3D pose sequence.
    lifter_result = run_motionbert_lifter(
        motionbert_input_sequence,
        output_dir=motionbert_output_dir,
        window_size=int(motionbert_window_size),
        window_overlap=float(motionbert_window_overlap),
        include_confidence=bool(include_motionbert_confidence),
        predictor=motionbert_predictor,
        backend_name=motionbert_backend_name,
        config_path=motionbert_config_path,
        checkpoint=motionbert_checkpoint_path,
        device=str(motionbert_device),
        env_name=str(env_name if motionbert_env_name is None else motionbert_env_name),
        allow_fallback_backend=bool(allow_motionbert_fallback_backend),
        image_width=_optional_int(None if video_metadata is None else video_metadata.get("width")),
        image_height=_optional_int(None if video_metadata is None else video_metadata.get("height")),
    )
    
    raw_pose_sequence_3d = lifter_result["pose_sequence"]
    raw_motionbert_pose3d_path = pose3d_output_dir / "pose3d_motionbert17.npz"
    write_pose_sequence3d_npz(raw_pose_sequence_3d, raw_motionbert_pose3d_path)
    
    # Stabilize lower-limb motion before remapping the skeleton.
    lower_limb_result = run_lower_limb_stabilizer(raw_pose_sequence_3d)
    stabilized_pose_sequence_3d = lower_limb_result["pose_sequence"]
    stabilized_motionbert_pose3d_path = pose3d_output_dir / "pose3d_motionbert17_stabilized.npz"
    write_pose_sequence3d_npz(stabilized_pose_sequence_3d, stabilized_motionbert_pose3d_path)
    write_json_file(lower_limb_result["quality_report"], pose3d_output_dir / "lower_limb_stabilizer_report.json")
    
    # Convert the lifted joints into the project skeleton.
    mapped_pose_sequence_3d, mapper_quality, mapper_artifacts = map_pose_sequence_to_imugpt22(stabilized_pose_sequence_3d)
    
    # Recover metric scale and smooth local joint positions.
    metric_normalizer_result = run_metric_normalizer(mapped_pose_sequence_3d, lower_limb_correction_masks=lower_limb_result["artifacts"]["correction_masks"])
    metric_pose_sequence_3d = metric_normalizer_result["pose_sequence"]
    metric_normalization = metric_normalizer_result["normalization_result"]
    metric_quality = metric_normalizer_result["quality_report"]
    metric_artifacts = metric_normalizer_result["artifacts"]
    metric_pose_sequence_path = pose3d_output_dir / "pose3d_metric_local.npz"
    write_pose_sequence3d_npz(metric_pose_sequence_3d, metric_pose_sequence_path)
    np.save(pose3d_output_dir / "3d_keypoints_metric.npy", np.asarray(metric_normalization["joint_positions_smoothed"], dtype=np.float32))
    
    # Estimate global root motion and export the final motion.
    root_result = run_root_trajectory_estimator(metric_pose_sequence_3d, normalization_result=metric_normalization)
    root_pose_sequence_3d = root_result["pose_sequence"]
    write_pose_sequence3d_npz(root_pose_sequence_3d, pose3d_output_dir / "pose3d.npz")
    np.save(pose3d_output_dir / "root_translation.npy", np.asarray(root_result["root_translation_m"], dtype=np.float32))
    bvh_artifacts = export_pose_sequence3d_to_bvh(root_pose_sequence_3d, pose3d_output_dir / "pose3d.bvh")

    merged_quality = merge_pose3d_quality_reports(
        pose2d_quality=pose2d_result["quality_report"],
        lifter_quality=lifter_result["quality_report"],
        lower_limb_quality=lower_limb_result["quality_report"],
        mapper_quality=mapper_quality,
        normalizer_quality=metric_quality,
        root_quality=root_result["quality_report"],
    )
    
    pose3d_debug_overlay_path = None
    pose3d_stabilized_debug_overlay_path = None
    pose3d_imugpt22_debug_overlay_path = None
    
    if save_debug_3d:
        # Render side-by-side previews for key 3D variants.
        cleaner_artifacts = pose2d_result["cleaner_artifacts"]
        clean_keypoints_xy_pixels = _restore_clean_pose_pixels(
            pose2d_result["pose_sequence"].keypoints_xy,
            cleaner_artifacts["normalization_centers_xy"],
            cleaner_artifacts["normalization_scales"],
            pose2d_result["pose_sequence"].confidence
        )
        pose3d_debug_overlay_path = _render_pose3d_debug_overlay(
            video_path=str(video_path),
            output_path=resolve_debug_overlay_variant_path(pose3d_output_dir, variant="pose3d_raw", enabled=True),
            pose_sequence_2d=pose2d_result["pose_sequence"],
            clean_keypoints_xy=clean_keypoints_xy_pixels,
            pose_sequence_3d=raw_pose_sequence_3d,
            overlay_variant="pose3d_raw",
            merged_quality=merged_quality
        )
        pose3d_stabilized_debug_overlay_path = _render_pose3d_debug_overlay(
            video_path=str(video_path),
            output_path=resolve_debug_overlay_variant_path(pose3d_output_dir, variant="pose3d_stabilized", enabled=True),
            pose_sequence_2d=pose2d_result["pose_sequence"],
            clean_keypoints_xy=clean_keypoints_xy_pixels,
            pose_sequence_3d=stabilized_pose_sequence_3d,
            overlay_variant="pose3d_stabilized",
            merged_quality=merged_quality
        )
        pose3d_imugpt22_debug_overlay_path = _render_pose3d_debug_overlay(
            video_path=str(video_path),
            output_path=resolve_debug_overlay_variant_path(pose3d_output_dir, variant="pose3d_imugpt22", enabled=True),
            pose_sequence_2d=pose2d_result["pose_sequence"],
            clean_keypoints_xy=clean_keypoints_xy_pixels,
            pose_sequence_3d=metric_pose_sequence_3d,
            overlay_variant="pose3d_imugpt22",
            merged_quality=merged_quality
        )
    # Save the merged reports and artifact map for downstream stages.
    write_json_file(merged_quality, pose3d_output_dir / "quality_report.json")

    artifacts = dict(pose2d_result["artifacts"])
    artifacts.update(lifter_result["artifacts"])
    artifacts["pose3d_motionbert17_npz_path"] = str(raw_motionbert_pose3d_path.resolve())
    artifacts["pose3d_motionbert17_stabilized_npz_path"] = str(stabilized_motionbert_pose3d_path.resolve())
    artifacts["pose3d_metric_local_npz_path"] = str(metric_pose_sequence_path.resolve())
    artifacts["pose3d_npz_path"] = str((pose3d_output_dir / "pose3d.npz").resolve())
    artifacts["pose3d_metric_keypoints_path"] = str((pose3d_output_dir / "3d_keypoints_metric.npy").resolve())
    artifacts["root_translation_npy_path"] = str((pose3d_output_dir / "root_translation.npy").resolve())
    artifacts["pose3d_bvh_path"] = str(Path(bvh_artifacts["pose3d_bvh_path"]).resolve())
    artifacts["quality_report_json_path"] = str((pose3d_output_dir / "quality_report.json").resolve())
    artifacts["lower_limb_stabilizer_report_json_path"] = str((pose3d_output_dir / "lower_limb_stabilizer_report.json").resolve())
    artifacts["debug_overlay_pose3d_raw_path"] = (None if pose3d_debug_overlay_path is None else str(pose3d_debug_overlay_path))
    artifacts["debug_overlay_pose3d_stabilized_path"] = (None if pose3d_stabilized_debug_overlay_path is None else str(pose3d_stabilized_debug_overlay_path))
    artifacts["debug_overlay_pose3d_imugpt22_path"] = (None if pose3d_imugpt22_debug_overlay_path is None else str(pose3d_imugpt22_debug_overlay_path))

    return {
        "clip_id": str(clip_id),
        "pose2d_result": pose2d_result,
        "pose_sequence": root_pose_sequence_3d,
        "metric_pose_sequence": metric_pose_sequence_3d,
        "root_translation_m": root_result["root_translation_m"],
        "motionbert_pose_sequence": raw_pose_sequence_3d,
        "lower_limb_stabilized_pose_sequence": stabilized_pose_sequence_3d,
        "skeleton_mapped_pose_sequence": mapped_pose_sequence_3d,
        "pose_sequence_2d": pose2d_result["pose_sequence"],
        "raw_pose_sequence_2d": pose2d_result["raw_pose_sequence"],
        "quality_report": merged_quality,
        "pose2d_quality_report": pose2d_result["quality_report"],
        "motionbert_quality_report": lifter_result["quality_report"],
        "lower_limb_stabilizer_quality_report": lower_limb_result["quality_report"],
        "skeleton_mapper_quality_report": mapper_quality,
        "metric_normalization_quality_report": metric_quality,
        "root_trajectory_quality_report": root_result["quality_report"],
        "track_report": pose2d_result["track_report"],
        "backend_run": pose2d_result["backend_run"],
        "motionbert_run": lifter_result["run_report"],
        "lower_limb_stabilizer_artifacts": lower_limb_result["artifacts"],
        "skeleton_mapper_artifacts": mapper_artifacts,
        "metric_normalization_result": metric_normalization,
        "metric_normalization_artifacts": metric_artifacts,
        "root_trajectory_result": root_result["trajectory_result"],
        "root_trajectory_artifacts": root_result["artifacts"],
        "artifacts": artifacts,
    }


def run_pose3d_from_prompt(
    *,
    prompt_id: str,
    prompt_text: str,
    output_dir: str | Path,
    sample_id: Optional[str] = None,
    labels: Optional[Mapping[str, Any]] = None,
    seed: int = 0,
    fps: float = 20.0,
    duration_hint_sec: Optional[float] = None,
    action_detail: Optional[str] = None,
    stimulus_type: Optional[str] = None,
    reference_clip_id: Optional[str] = None,
    group_id: Optional[str] = None,
    notes: Optional[str] = None,
    source_metadata: Optional[Mapping[str, Any]] = None,
    prompt_backend: Optional[PromptMotionBackend] = None,
    export_bvh: bool = True,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_id = str(prompt_id if sample_id in (None, "") else sample_id)
    labels_dict = {} if labels is None else dict(labels)
    source_metadata_dict = {} if source_metadata is None else dict(source_metadata)

    backend = LegacyT2MGPTBackend() if prompt_backend is None else prompt_backend
    generation_result = backend.generate(
        prompt_text=str(prompt_text),
        seed=int(seed),
        fps=float(fps),
        duration_hint_sec=duration_hint_sec,
        output_dir=output_dir,
    )
    generation_backend = str(generation_result.get("generation_backend", "t2mgpt"))
    backend_report = dict(generation_result.get("backend_report", {}))

    raw_pose_sequence_3d = build_prompt_pose_sequence3d(
        prompt_id=str(prompt_id),
        clip_id=str(clip_id),
        joint_positions_xyz=generation_result["joint_positions_xyz"],
        fps=float(fps),
        source=f"{generation_backend}_prompt",
    )
    raw_pose_path = output_dir / "pose3d_prompt_raw.npz"
    write_pose_sequence3d_npz(raw_pose_sequence_3d, raw_pose_path)

    prompt_metadata = build_prompt_metadata(
        sample_id=str(clip_id),
        prompt_id=str(prompt_id),
        prompt_text=str(prompt_text),
        labels=labels_dict,
        seed=int(seed),
        fps=float(fps),
        num_generated_frames=int(raw_pose_sequence_3d.num_frames),
        generation_backend=generation_backend,
        action_detail=action_detail,
        stimulus_type=stimulus_type,
        reference_clip_id=reference_clip_id,
        duration_hint_sec=duration_hint_sec,
        notes=notes,
        group_id=group_id,
        source_metadata=source_metadata_dict,
    )
    prompt_metadata_path = output_dir / "prompt_metadata.json"
    write_json_file(prompt_metadata, prompt_metadata_path)
    if backend_report:
        backend_report_path = output_dir / "prompt_backend_report.json"
        write_json_file(backend_report, backend_report_path)
    else:
        backend_report_path = None

    adapter_quality = build_prompt_adapter_quality_report(
        pose_sequence=raw_pose_sequence_3d,
        generation_backend=generation_backend,
    )
    metric_normalizer_result = run_metric_normalizer(raw_pose_sequence_3d)
    metric_pose_sequence_3d = metric_normalizer_result["pose_sequence"]
    metric_normalization = metric_normalizer_result["normalization_result"]
    metric_quality = metric_normalizer_result["quality_report"]
    metric_artifacts = metric_normalizer_result["artifacts"]
    metric_pose_path = output_dir / "pose3d_metric_local.npz"
    write_pose_sequence3d_npz(metric_pose_sequence_3d, metric_pose_path)
    np.save(
        output_dir / "3d_keypoints_metric.npy",
        np.asarray(metric_normalization["joint_positions_smoothed"], dtype=np.float32),
    )

    root_result = run_root_trajectory_estimator(
        metric_pose_sequence_3d,
        normalization_result=metric_normalization,
    )
    root_pose_sequence_3d = root_result["pose_sequence"]
    final_pose_path = output_dir / "pose3d.npz"
    write_pose_sequence3d_npz(root_pose_sequence_3d, final_pose_path)
    np.save(
        output_dir / "root_translation.npy",
        np.asarray(root_result["root_translation_m"], dtype=np.float32),
    )

    bvh_artifacts = {"pose3d_bvh_path": None}
    if bool(export_bvh):
        bvh_artifacts = export_pose_sequence3d_to_bvh(root_pose_sequence_3d, output_dir / "pose3d.bvh")

    prompt_source_quality = build_prompt_pose_quality_report(
        root_pose_sequence_3d,
        generation_backend=generation_backend,
    )
    merged_quality = _build_prompt_pose3d_quality_report(
        pose_sequence=root_pose_sequence_3d,
        generation_backend=generation_backend,
        adapter_quality=adapter_quality,
        prompt_source_quality=prompt_source_quality,
        metric_quality=metric_quality,
        root_quality=root_result["quality_report"],
    )
    quality_report_path = output_dir / "quality_report.json"
    write_json_file(merged_quality, quality_report_path)

    artifacts = {
        "prompt_metadata_json_path": str(prompt_metadata_path.resolve()),
        "prompt_backend_report_json_path": (
            None if backend_report_path is None else str(backend_report_path.resolve())
        ),
        "pose3d_prompt_raw_npz_path": str(raw_pose_path.resolve()),
        "pose3d_metric_local_npz_path": str(metric_pose_path.resolve()),
        "pose3d_npz_path": str(final_pose_path.resolve()),
        "pose3d_metric_keypoints_path": str((output_dir / "3d_keypoints_metric.npy").resolve()),
        "root_translation_npy_path": str((output_dir / "root_translation.npy").resolve()),
        "pose3d_bvh_path": (
            None
            if bvh_artifacts.get("pose3d_bvh_path") is None
            else str(Path(bvh_artifacts["pose3d_bvh_path"]).resolve())
        ),
        "quality_report_json_path": str(quality_report_path.resolve()),
    }
    artifacts.update(
        {
            key: value
            for key, value in dict(generation_result.get("artifacts", {})).items()
            if key not in artifacts
        }
    )
    return {
        "clip_id": str(clip_id),
        "pose_sequence": root_pose_sequence_3d,
        "metric_pose_sequence": metric_pose_sequence_3d,
        "prompt_pose_sequence": raw_pose_sequence_3d,
        "root_translation_m": root_result["root_translation_m"],
        "quality_report": merged_quality,
        "prompt_adapter_quality_report": adapter_quality,
        "prompt_source_quality_report": prompt_source_quality,
        "metric_normalization_quality_report": metric_quality,
        "root_trajectory_quality_report": root_result["quality_report"],
        "prompt_metadata": prompt_metadata,
        "backend_report": backend_report,
        "metric_normalization_result": metric_normalization,
        "metric_normalization_artifacts": metric_artifacts,
        "root_trajectory_result": root_result["trajectory_result"],
        "root_trajectory_artifacts": root_result["artifacts"],
        "artifacts": artifacts,
    }


def run_virtual_imu_pipeline(
    *,
    clip_id: str,
    video_path: str,
    output_dir: str | Path,
    activity_label: Any = None,
    fps_target: int = 20,
    save_debug: bool = True,
    save_debug_2d: Optional[bool] = None,
    save_debug_3d: Optional[bool] = None,
    env_name: str = "openmmlab",
    video_metadata: Optional[Mapping[str, Any]] = None,
    model_alias: str = "vitpose-b",
    motionbert_window_size: int = 81,
    motionbert_window_overlap: float = 0.5,
    include_motionbert_confidence: bool = True,
    motionbert_predictor: Optional[MotionBERTPredictor] = None,
    motionbert_backend_name: Optional[str] = None,
    motionbert_env_name: Optional[str] = None,
    motionbert_config_path: Optional[str] = None,
    motionbert_checkpoint_path: Optional[str] = None,
    motionbert_device: str = "auto",
    allow_motionbert_fallback_backend: bool = False,
    sensor_layout_path: str | Path | None = None,
    imu_acc_noise_std_m_s2: Optional[float] = None,
    imu_gyro_noise_std_rad_s: Optional[float] = None,
    imu_random_seed: int = 0,
    real_imu_reference_path: str | Path | None = None,
    real_imu_signal_mode: str = "acc",
    real_imu_percentile_resolution: int = 100,
    real_imu_per_class_calibration: bool = True,
    defer_real_imu_calibration: bool = False,
    estimate_sensor_frame: bool = False,
    estimate_sensor_names: Sequence[str] | None = None,
) -> Dict[str, Any]:
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ik_output_dir = _resolve_stage_output_dir(output_dir, "ik")
    virtual_imu_output_dir = _resolve_stage_output_dir(output_dir, "virtual_imu")
    geometric_alignment_output_dir = _resolve_stage_output_dir(virtual_imu_output_dir, "geometric_alignment")
    sensor_frame_output_dir = _resolve_stage_output_dir(virtual_imu_output_dir, "sensor_frame")
    geometric_alignment_settings = load_alignment_runtime_settings(None)
    has_real_imu_reference = real_imu_reference_path not in (None, "")
    defer_real_imu_calibration = bool(defer_real_imu_calibration and has_real_imu_reference)
    
    postpone_real_imu_calibration_until_after_alignment = bool(
        has_real_imu_reference
        and not defer_real_imu_calibration
        and geometric_alignment_settings.get("enable", False)
    )
    
    imusim_real_imu_reference_path = (
        None
        if defer_real_imu_calibration or postpone_real_imu_calibration_until_after_alignment
        else real_imu_reference_path
    )

    pose3d_result = run_pose3d_pipeline(
        clip_id=str(clip_id),
        video_path=str(video_path),
        output_dir=output_dir,
        fps_target=int(fps_target),
        save_debug=bool(save_debug),
        save_debug_2d=save_debug_2d,
        save_debug_3d=save_debug_3d,
        env_name=str(env_name),
        video_metadata=video_metadata,
        model_alias=str(model_alias),
        motionbert_window_size=int(motionbert_window_size),
        motionbert_window_overlap=float(motionbert_window_overlap),
        include_motionbert_confidence=bool(include_motionbert_confidence),
        motionbert_predictor=motionbert_predictor,
        motionbert_backend_name=motionbert_backend_name,
        motionbert_env_name=motionbert_env_name,
        motionbert_config_path=motionbert_config_path,
        motionbert_checkpoint_path=motionbert_checkpoint_path,
        motionbert_device=str(motionbert_device),
        allow_motionbert_fallback_backend=bool(allow_motionbert_fallback_backend),
    )

    # Solve a kinematic body sequence from the final 3D pose.
    ik_result = run_ik(pose3d_result["pose_sequence"], output_dir=ik_output_dir)
    
    # Synthesize virtual IMU signals from the solved motion.
    imusim_result = run_imusim(
        ik_result["ik_sequence"],
        sensor_layout_path=DEFAULT_SENSOR_LAYOUT_PATH if sensor_layout_path is None else sensor_layout_path,
        output_dir=virtual_imu_output_dir,
        acc_noise_std_m_s2=imu_acc_noise_std_m_s2,
        gyro_noise_std_rad_s=imu_gyro_noise_std_rad_s,
        random_seed=int(imu_random_seed),
        real_imu_reference_path=imusim_real_imu_reference_path,
        real_imu_activity_label=activity_label,
        real_imu_signal_mode=str(real_imu_signal_mode),
        real_imu_percentile_resolution=int(real_imu_percentile_resolution),
        real_imu_per_class_calibration=bool(real_imu_per_class_calibration),
    )
    
    resolved_real_imu_npz_path = _resolve_real_imu_npz_path(output_dir=output_dir)
    
    # Optionally align the synthetic signals to a real IMU reference.
    if bool(geometric_alignment_settings.get("enable", False)):
        geometric_alignment_result = _run_optional_geometric_alignment(
            raw_virtual_imu_sequence=imusim_result["raw_virtual_imu_sequence"],
            output_dir=geometric_alignment_output_dir,
            real_imu_npz_path=resolved_real_imu_npz_path,
        )
    else:
        geometric_alignment_result = _build_disabled_geometric_alignment_result(
            raw_virtual_imu_sequence=imusim_result["raw_virtual_imu_sequence"],
            config_path=geometric_alignment_settings.get("config_path"),
        )
    post_geometric_virtual_imu_sequence = geometric_alignment_result["aligned_virtual_imu_sequence"]
    final_virtual_imu_sequence = imusim_result["virtual_imu_sequence"]
    final_virtual_imu_calibration_report = imusim_result["calibration_report"]
    
    pre_calibration_virtual_imu_sequence = (imusim_result["raw_virtual_imu_sequence"] if final_virtual_imu_calibration_report is not None else None)

    # Calibrate amplitudes after alignment when a real reference is available.
    if postpone_real_imu_calibration_until_after_alignment:
        calibration_result = calibrate_virtual_imu_sequence(
            post_geometric_virtual_imu_sequence,
            real_imu_reference_path=str(real_imu_reference_path),
            activity_label=activity_label,
            signal_mode=str(real_imu_signal_mode),
            percentile_resolution=int(real_imu_percentile_resolution),
            per_class=bool(real_imu_per_class_calibration),
        )
        
        final_virtual_imu_sequence = calibration_result["virtual_imu_sequence"]
        final_virtual_imu_calibration_report = dict(calibration_result["calibration_report"])
        pre_calibration_virtual_imu_sequence = post_geometric_virtual_imu_sequence
        
    elif bool(geometric_alignment_result.get("enabled")) or bool(defer_real_imu_calibration):
        final_virtual_imu_sequence = post_geometric_virtual_imu_sequence

    final_virtual_imu_quality_report = _sync_virtual_imu_pipeline_outputs(
        output_dir=virtual_imu_output_dir,
        imusim_result=imusim_result,
        virtual_imu_sequence=final_virtual_imu_sequence,
        calibration_report=final_virtual_imu_calibration_report,
        pre_calibration_virtual_imu_sequence=pre_calibration_virtual_imu_sequence,
    )

    # Optionally estimate sensor-frame corrections against real IMU data.
    frame_alignment_result = _build_disabled_sensor_frame_estimation_result(
        raw_virtual_imu_sequence=final_virtual_imu_sequence,
        target_sensor_names=DEFAULT_TARGET_SENSOR_NAMES if estimate_sensor_names is None else estimate_sensor_names,
    )
    if bool(estimate_sensor_frame):
        frame_alignment_result = _run_optional_sensor_frame_estimation(
            raw_virtual_imu_sequence=final_virtual_imu_sequence,
            target_sensor_names=DEFAULT_TARGET_SENSOR_NAMES
            if estimate_sensor_names is None
            else estimate_sensor_names,
            real_imu_npz_path=resolved_real_imu_npz_path,
            output_dir=sensor_frame_output_dir,
        )
    merged_quality = merge_virtual_imu_quality_reports(
        pose3d_quality=pose3d_result["quality_report"],
        ik_quality=ik_result["quality_report"],
        virtual_imu_quality=final_virtual_imu_quality_report,
        geometric_alignment_quality=geometric_alignment_result["quality_report"],
        frame_alignment_quality=frame_alignment_result["quality_report"],
    )
    # Save the consolidated reports and expose all generated artifacts.
    write_json_file(merged_quality, virtual_imu_output_dir / "quality_report.json")

    artifacts = dict(pose3d_result["artifacts"])
    artifacts.update(ik_result["artifacts"])
    artifacts.update(imusim_result["artifacts"])
    artifacts.update(geometric_alignment_result["artifacts"])
    artifacts.update(frame_alignment_result["artifacts"])
    artifacts["quality_report_json_path"] = str((virtual_imu_output_dir / "quality_report.json").resolve())

    return {
        "clip_id": str(clip_id),
        "pose3d_result": pose3d_result,
        "pose_sequence": pose3d_result["pose_sequence"],
        "virtual_imu_sequence": final_virtual_imu_sequence,
        "ik_sequence": ik_result["ik_sequence"],
        "quality_report": merged_quality,
        "pose3d_quality_report": pose3d_result["quality_report"],
        "ik_quality_report": ik_result["quality_report"],
        "virtual_imu_quality_report": final_virtual_imu_quality_report,
        "virtual_imu_calibration_report": final_virtual_imu_calibration_report,
        "geometric_alignment_quality_report": geometric_alignment_result["quality_report"],
        "geometric_alignment_result": geometric_alignment_result,
        "geometrically_aligned_virtual_imu_sequence": geometric_alignment_result["aligned_virtual_imu_sequence"],
        "aligned_virtual_imu_sequence": frame_alignment_result["aligned_virtual_imu_sequence"],
        "sensor_frame_estimation_report": frame_alignment_result["frame_estimation_report"],
        "sensor_frame_estimation_lag_report": frame_alignment_result["lag_report"],
        "frame_alignment_quality_report": frame_alignment_result["quality_report"],
        "ik_result": ik_result,
        "imusim_result": imusim_result,
        "frame_alignment_result": frame_alignment_result,
        "artifacts": artifacts,
    }

def _sync_virtual_imu_pipeline_outputs(
    *,
    output_dir: Path,
    imusim_result: Dict[str, Any],
    virtual_imu_sequence: VirtualIMUSequence,
    calibration_report: Mapping[str, Any] | None,
    pre_calibration_virtual_imu_sequence: VirtualIMUSequence | None,
) -> Dict[str, Any]:
    
    original_quality = dict(imusim_result.get("quality_report", {}))
    quality_report = _build_virtual_imu_quality_report(
        imu_sequence=virtual_imu_sequence,
        acc_noise_std_m_s2=float(original_quality.get("acc_noise_std_m_s2", 0.0)),
        gyro_noise_std_rad_s=float(original_quality.get("gyro_noise_std_rad_s", 0.0)),
        max_acceleration_warning_m_s2=float(DEFAULT_MAX_ACCELERATION_WARNING_M_S2),
        max_gyro_warning_rad_s=float(DEFAULT_MAX_GYRO_WARNING_RAD_S),
        calibration_report=None if calibration_report is None else dict(calibration_report),
    )

    final_virtual_imu_path = output_dir / "virtual_imu.npz"
    np.savez_compressed(final_virtual_imu_path, **virtual_imu_sequence.to_npz_payload())

    report_path = output_dir / DEFAULT_IMU_REPORT_FILENAME
    write_json_file(quality_report, report_path)

    artifacts = dict(imusim_result.get("artifacts", {}))
    artifacts["virtual_imu_npz_path"] = str(final_virtual_imu_path.resolve())
    artifacts["virtual_imu_report_json_path"] = str(report_path.resolve())

    if calibration_report is not None and pre_calibration_virtual_imu_sequence is not None:
        raw_path = output_dir / DEFAULT_RAW_IMU_SEQUENCE_FILENAME
        np.savez_compressed(raw_path, **pre_calibration_virtual_imu_sequence.to_npz_payload())
        calibration_report_path = output_dir / DEFAULT_IMU_CALIBRATION_REPORT_FILENAME
        write_json_file(dict(calibration_report), calibration_report_path)
        artifacts["virtual_imu_raw_npz_path"] = str(raw_path.resolve())
        artifacts["virtual_imu_calibration_report_json_path"] = str(calibration_report_path.resolve())
    else:
        artifacts["virtual_imu_raw_npz_path"] = None
        artifacts["virtual_imu_calibration_report_json_path"] = None

    imusim_result["virtual_imu_sequence"] = virtual_imu_sequence
    imusim_result["quality_report"] = quality_report
    imusim_result["calibration_report"] = None if calibration_report is None else dict(calibration_report)
    imusim_result["artifacts"] = artifacts
    return quality_report


def generate_pose_from_video(
    video_path: str,
    clip_id: str,
    *,
    fps_target: int = 20,
    save_debug: bool = True,
) -> Any:
    output_dir = Path("output") / str(clip_id) / "pose"
    result = run_pose3d_pipeline(
        clip_id=str(clip_id),
        video_path=str(video_path),
        output_dir=output_dir,
        fps_target=int(fps_target),
        save_debug=bool(save_debug),
    )
    return result["pose_sequence"]


def generate_virtual_imu_from_video(
    video_path: str,
    clip_id: str,
    *,
    activity_label: Any = None,
    fps_target: int = 20,
) -> Any:
    output_dir = Path("output") / str(clip_id) / "pose"
    result = run_virtual_imu_pipeline(
        clip_id=str(clip_id),
        video_path=str(video_path),
        output_dir=output_dir,
        activity_label=activity_label,
        fps_target=int(fps_target),
    )
    return result["virtual_imu_sequence"]


def generate_pose_from_prompt(
    prompt_text: str,
    prompt_id: str,
    *,
    seed: int = 0,
    fps: float = 20.0,
    output_dir: str | Path | None = None,
    prompt_backend: Optional[PromptMotionBackend] = None,
) -> PoseSequence3D:
    resolved_output_dir = (
        Path("output") / "synthetic" / str(prompt_id) / "pose"
        if output_dir is None
        else Path(output_dir)
    )
    result = run_pose3d_from_prompt(
        prompt_id=str(prompt_id),
        prompt_text=str(prompt_text),
        output_dir=resolved_output_dir,
        seed=int(seed),
        fps=float(fps),
        prompt_backend=prompt_backend,
    )
    return result["pose_sequence"]


def run_pose3d_from_video(**kwargs: Any) -> Dict[str, Any]:
    return run_pose3d_pipeline(**kwargs)


def _build_prompt_pose3d_quality_report(
    *,
    pose_sequence: PoseSequence3D,
    generation_backend: str,
    adapter_quality: Mapping[str, Any],
    prompt_source_quality: Mapping[str, Any],
    metric_quality: Mapping[str, Any],
    root_quality: Mapping[str, Any],
) -> Dict[str, Any]:
    statuses = [
        str(adapter_quality.get("status", "fail")),
        str(prompt_source_quality.get("status", "fail")),
        str(metric_quality.get("status", "fail")),
        str(root_quality.get("status", "fail")),
    ]
    if "fail" in statuses:
        status = "fail"
    elif "warning" in statuses:
        status = "warning"
    else:
        status = "ok"

    notes: list[str] = []
    for report in (adapter_quality, prompt_source_quality, metric_quality, root_quality):
        for note in report.get("notes", []):
            notes.append(str(note))
    notes = list(dict.fromkeys(notes))

    return {
        "clip_id": str(pose_sequence.clip_id),
        "status": str(status),
        "source_kind": "prompt",
        "modality_domain": "synthetic",
        "generation_backend": str(generation_backend),
        "fps": None if pose_sequence.fps is None else float(pose_sequence.fps),
        "fps_original": None if pose_sequence.fps_original is None else float(pose_sequence.fps_original),
        "num_frames": int(pose_sequence.num_frames),
        "num_joints": int(pose_sequence.num_joints),
        "coordinate_space": str(pose_sequence.coordinate_space),
        "nan_ratio": float(prompt_source_quality.get("nan_ratio", 1.0)),
        "bone_length_cv": float(prompt_source_quality.get("bone_length_cv", 0.0)),
        "root_motion_energy": float(prompt_source_quality.get("root_motion_energy", 0.0)),
        "foot_sliding_score": float(prompt_source_quality.get("foot_sliding_score", 0.0)),
        "prompt_adapter_status": str(adapter_quality.get("status")),
        "metric_normalizer_status": str(metric_quality.get("status")),
        "root_trajectory_status": str(root_quality.get("status")),
        "notes": list(notes),
    }


def _optional_float(raw_value: Any) -> Optional[float]:
    if raw_value in (None, ""):
        return None
    return float(raw_value)


def _optional_int(raw_value: Any) -> Optional[int]:
    if raw_value in (None, ""):
        return None
    return int(raw_value)


def _restore_clean_pose_pixels(
    normalized_keypoints_xy: np.ndarray,
    centers_xy: np.ndarray,
    scales: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    points_xy = (
        np.asarray(normalized_keypoints_xy, dtype=np.float32)
        * np.asarray(scales, dtype=np.float32)[:, None, None]
    ) + np.asarray(centers_xy, dtype=np.float32)[:, None, :]
    invalid_mask = np.asarray(confidence, dtype=np.float32) <= 0.0
    points_xy[invalid_mask] = np.nan
    return points_xy


def _build_motionbert_backend_input_sequence(pose2d_result: Mapping[str, Any]) -> PoseSequence2D:
    pose_sequence = pose2d_result["pose_sequence"]
    cleaner_artifacts = dict(pose2d_result.get("cleaner_artifacts", {}))
    clean_points_xy_pixels = cleaner_artifacts.get("clean_motionbert17_xy_pixels")
    if clean_points_xy_pixels is None and {
        "normalization_centers_xy",
        "normalization_scales",
    }.issubset(cleaner_artifacts.keys()):
        clean_points_xy_pixels = _restore_clean_pose_pixels(
            pose_sequence.keypoints_xy,
            cleaner_artifacts["normalization_centers_xy"],
            cleaner_artifacts["normalization_scales"],
            pose_sequence.confidence,
        )
    if clean_points_xy_pixels is None:
        clean_points_xy_pixels = np.asarray(pose_sequence.keypoints_xy, dtype=np.float32)

    return PoseSequence2D(
        clip_id=str(pose_sequence.clip_id),
        fps=None if pose_sequence.fps is None else float(pose_sequence.fps),
        fps_original=None if pose_sequence.fps_original is None else float(pose_sequence.fps_original),
        joint_names_2d=list(pose_sequence.joint_names_2d),
        keypoints_xy=np.asarray(clean_points_xy_pixels, dtype=np.float32),
        confidence=np.asarray(pose_sequence.confidence, dtype=np.float32),
        bbox_xywh=np.asarray(pose_sequence.bbox_xywh, dtype=np.float32),
        frame_indices=np.asarray(pose_sequence.frame_indices, dtype=np.int32),
        timestamps_sec=np.asarray(pose_sequence.timestamps_sec, dtype=np.float32),
        source=f"{pose_sequence.source}_pixels",
        observed_mask=np.asarray(pose_sequence.resolved_observed_mask(), dtype=bool),
        imputed_mask=np.asarray(pose_sequence.resolved_imputed_mask(), dtype=bool),
    )


def _resolve_real_imu_npz_path(*, output_dir: Path) -> str | None:
    for candidate in (
        output_dir / "imu.npz",
        output_dir.parent / "imu.npz",
        output_dir.parent.parent / "imu.npz",
    ):
        if candidate.exists():
            return str(candidate.resolve())
    return None


def _resolve_stage_output_dir(base_dir: Path, stage_name: str) -> Path:
    stage_output_dir = Path(base_dir) / str(stage_name)
    stage_output_dir.mkdir(parents=True, exist_ok=True)
    return stage_output_dir


def _run_optional_geometric_alignment(*, raw_virtual_imu_sequence: VirtualIMUSequence, output_dir: Path, real_imu_npz_path: str | None) -> Dict[str, Any]:
    try:
        return run_geometric_alignment(
            virtual_imu_sequence=raw_virtual_imu_sequence,
            output_dir=output_dir,
            real_imu_npz_path=real_imu_npz_path,
        )
    except Exception as exc:
        return _build_failed_geometric_alignment_result(
            raw_virtual_imu_sequence=raw_virtual_imu_sequence,
            output_dir=output_dir,
            error=str(exc),
            real_imu_npz_path=real_imu_npz_path,
        )


def _run_optional_sensor_frame_estimation(*, raw_virtual_imu_sequence: VirtualIMUSequence, target_sensor_names: Sequence[str], real_imu_npz_path: str | None, output_dir: Path) -> Dict[str, Any]:
    try:
        return estimate_sensor_frame_alignment(
            raw_virtual_imu_sequence,
            real_imu_npz_path=real_imu_npz_path,
            target_sensor_names=target_sensor_names,
            output_dir=output_dir,
        )
    except Exception as exc:
        return _build_failed_sensor_frame_estimation_result(
            raw_virtual_imu_sequence=raw_virtual_imu_sequence,
            target_sensor_names=target_sensor_names,
            output_dir=output_dir,
            error=str(exc),
            real_imu_npz_path=real_imu_npz_path,
        )


def _build_disabled_sensor_frame_estimation_result(*, raw_virtual_imu_sequence: VirtualIMUSequence, target_sensor_names: Sequence[str]) -> Dict[str, Any]:
    del raw_virtual_imu_sequence
    return {
        "status": "not_requested",
        "aligned_virtual_imu_sequence": None,
        "frame_estimation_report": None,
        "lag_report": None,
        "quality_report": {
            "enabled": False,
            "status": None,
            "target_sensor_names": [str(name) for name in target_sensor_names],
            "estimated_sensor_names": [],
            "real_imu_npz_path": None,
            "mean_gyro_corr_before": None,
            "mean_gyro_corr_after": None,
            "mean_acc_corr_before": None,
            "mean_acc_corr_after": None,
            "notes": [],
        },
        "artifacts": {
            "virtual_imu_frame_aligned_npz_path": None,
            "sensor_frame_estimation_report_json_path": None,
            "frame_alignment_quality_report_json_path": None,
        },
    }


def _build_disabled_geometric_alignment_result(*, raw_virtual_imu_sequence: VirtualIMUSequence, config_path: str | None) -> Dict[str, Any]:
    return {
        "status": "not_enabled",
        "enabled": False,
        "aligned_virtual_imu_sequence": VirtualIMUSequence(
            clip_id=str(raw_virtual_imu_sequence.clip_id),
            fps=None if raw_virtual_imu_sequence.fps is None else float(raw_virtual_imu_sequence.fps),
            sensor_names=list(raw_virtual_imu_sequence.sensor_names),
            acc=np.asarray(raw_virtual_imu_sequence.acc, dtype=np.float32),
            gyro=np.asarray(raw_virtual_imu_sequence.gyro, dtype=np.float32),
            timestamps_sec=np.asarray(raw_virtual_imu_sequence.timestamps_sec, dtype=np.float32),
            source=str(raw_virtual_imu_sequence.source),
        ),
        "transforms": {},
        "metrics_before": {},
        "metrics_after": {},
        "quality_report": {
            "enabled": False,
            "status": None,
            "subject_id": None,
            "capture_id": str(raw_virtual_imu_sequence.clip_id),
            "transform_source": None,
            "real_imu_npz_path": None,
            "estimated_sensor_names": [],
            "mean_acc_corr_before": None,
            "mean_acc_corr_after": None,
            "mean_gyro_corr_before": None,
            "mean_gyro_corr_after": None,
            "notes": [],
        },
        "artifacts": {
            "virtual_imu_geometric_aligned_npz_path": None,
            "imu_alignment_transforms_json_path": None,
            "imu_alignment_metrics_json_path": None,
            "imu_alignment_quality_report_json_path": None,
            "imu_alignment_config_path": config_path,
        },
    }


def _build_failed_geometric_alignment_result(*, raw_virtual_imu_sequence: VirtualIMUSequence, output_dir: Path, error: str, real_imu_npz_path: str | None) -> Dict[str, Any]:
    del output_dir
    aligned_virtual_imu_sequence = VirtualIMUSequence(
        clip_id=str(raw_virtual_imu_sequence.clip_id),
        fps=None if raw_virtual_imu_sequence.fps is None else float(raw_virtual_imu_sequence.fps),
        sensor_names=list(raw_virtual_imu_sequence.sensor_names),
        acc=np.asarray(raw_virtual_imu_sequence.acc, dtype=np.float32),
        gyro=np.asarray(raw_virtual_imu_sequence.gyro, dtype=np.float32),
        timestamps_sec=np.asarray(raw_virtual_imu_sequence.timestamps_sec, dtype=np.float32),
        source=str(raw_virtual_imu_sequence.source),
    )
    notes = [f"geometric_alignment_failed:{error}"]
    quality_report = {
        "enabled": True,
        "status": "warning",
        "subject_id": None,
        "capture_id": str(raw_virtual_imu_sequence.clip_id),
        "transform_source": None,
        "real_imu_npz_path": real_imu_npz_path,
        "estimated_sensor_names": [],
        "mean_acc_corr_before": None,
        "mean_acc_corr_after": None,
        "mean_gyro_corr_before": None,
        "mean_gyro_corr_after": None,
        "notes": list(notes),
    }
    return {
        "status": "warning",
        "enabled": False,
        "aligned_virtual_imu_sequence": aligned_virtual_imu_sequence,
        "transforms": {},
        "metrics_before": {},
        "metrics_after": {},
        "quality_report": quality_report,
        "artifacts": {
            "virtual_imu_geometric_aligned_npz_path": None,
            "imu_alignment_transforms_json_path": None,
            "imu_alignment_metrics_json_path": None,
            "imu_alignment_quality_report_json_path": None,
            "imu_alignment_config_path": None,
        },
    }


def _build_failed_sensor_frame_estimation_result(*, raw_virtual_imu_sequence: VirtualIMUSequence, target_sensor_names: Sequence[str], output_dir: Path, error: str, real_imu_npz_path: str | None) -> Dict[str, Any]:
    aligned_virtual_imu_sequence = VirtualIMUSequence(
        clip_id=str(raw_virtual_imu_sequence.clip_id),
        fps=None if raw_virtual_imu_sequence.fps is None else float(raw_virtual_imu_sequence.fps),
        sensor_names=list(raw_virtual_imu_sequence.sensor_names),
        acc=np.asarray(raw_virtual_imu_sequence.acc, dtype=np.float32),
        gyro=np.asarray(raw_virtual_imu_sequence.gyro, dtype=np.float32),
        timestamps_sec=np.asarray(raw_virtual_imu_sequence.timestamps_sec, dtype=np.float32),
        source=f"{raw_virtual_imu_sequence.source}_frame_aligned",
    )
    notes = [f"sensor_frame_estimation_failed:{error}"]
    frame_estimation_report = {
        "clip_id": str(raw_virtual_imu_sequence.clip_id),
        "status": "warning",
        "real_imu_npz_path": real_imu_npz_path,
        "target_sensor_names": [str(name) for name in target_sensor_names],
        "sensor_reports": {},
        "notes": list(notes),
    }
    lag_report = {
        "clip_id": str(raw_virtual_imu_sequence.clip_id),
        "status": "warning",
        "real_imu_npz_path": real_imu_npz_path,
        "target_sensor_names": [str(name) for name in target_sensor_names],
        "sensor_lags": {},
        "notes": list(notes),
    }
    quality_report = {
        "enabled": True,
        "clip_id": str(raw_virtual_imu_sequence.clip_id),
        "status": "warning",
        "real_imu_npz_path": real_imu_npz_path,
        "target_sensor_names": [str(name) for name in target_sensor_names],
        "estimated_sensor_names": [],
        "mean_gyro_corr_before": None,
        "mean_gyro_corr_after": None,
        "mean_acc_corr_before": None,
        "mean_acc_corr_after": None,
        "notes": list(notes),
    }
    artifacts = {
        "virtual_imu_frame_aligned_npz_path": None,
        "sensor_frame_estimation_report_json_path": None,
        "frame_alignment_quality_report_json_path": None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = output_dir / "virtual_imu_frame_aligned.npz"
    np.savez_compressed(aligned_path, **aligned_virtual_imu_sequence.to_npz_payload())
    artifacts["virtual_imu_frame_aligned_npz_path"] = str(aligned_path.resolve())
    frame_report_path = output_dir / "sensor_frame_estimation_report.json"
    write_json_file(frame_estimation_report, frame_report_path)
    artifacts["sensor_frame_estimation_report_json_path"] = str(frame_report_path.resolve())
    quality_report_path = output_dir / "frame_alignment_quality_report.json"
    write_json_file(quality_report, quality_report_path)
    artifacts["frame_alignment_quality_report_json_path"] = str(quality_report_path.resolve())
    return {
        "status": "warning",
        "aligned_virtual_imu_sequence": aligned_virtual_imu_sequence,
        "frame_estimation_report": frame_estimation_report,
        "lag_report": lag_report,
        "quality_report": quality_report,
        "artifacts": artifacts,
    }


def _render_debug_overlay_variant(*, video_path: str, output_path: Path | None, pose_sequence: Any, keypoints_xy: np.ndarray, overlay_variant: str, merged_quality: Dict[str, Any]) -> Path | None:
    if output_path is None:
        return None
    try:
        return render_pose_overlay_video(
            video_path=video_path,
            output_path=output_path,
            frame_indices=np.asarray(pose_sequence.frame_indices, dtype=np.int32),
            keypoints_xy=np.asarray(keypoints_xy, dtype=np.float32),
            confidence=np.asarray(pose_sequence.confidence, dtype=np.float32),
            joint_names=pose_sequence.joint_names_2d,
            bbox_xywh=np.asarray(pose_sequence.bbox_xywh, dtype=np.float32),
            fps=pose_sequence.fps,
            overlay_variant=str(overlay_variant),
        )
    except Exception as exc:
        notes = list(merged_quality.get("notes", []))
        notes.append(f"{overlay_variant}_debug_overlay_failed:{exc}")
        merged_quality["notes"] = list(dict.fromkeys(str(value) for value in notes))
        if merged_quality.get("status") == "ok":
            merged_quality["status"] = "warning"
        return None


def _render_pose3d_debug_overlay(*, video_path: str, output_path: Path | None, pose_sequence_2d: Any, clean_keypoints_xy: np.ndarray, pose_sequence_3d: Any, overlay_variant: str, merged_quality: Dict[str, Any]) -> Path | None:
    if output_path is None:
        return None
    try:
        return render_pose3d_side_by_side_video(
            video_path=video_path,
            output_path=output_path,
            frame_indices=np.asarray(pose_sequence_3d.frame_indices, dtype=np.int32),
            keypoints_xy=np.asarray(clean_keypoints_xy, dtype=np.float32),
            confidence_2d=np.asarray(pose_sequence_2d.confidence, dtype=np.float32),
            joint_names_2d=pose_sequence_2d.joint_names_2d,
            joint_positions_xyz=np.asarray(pose_sequence_3d.joint_positions_xyz, dtype=np.float32),
            confidence_3d=np.asarray(pose_sequence_3d.joint_confidence, dtype=np.float32),
            joint_names_3d=pose_sequence_3d.joint_names_3d,
            skeleton_parents=pose_sequence_3d.skeleton_parents,
            bbox_xywh=np.asarray(pose_sequence_2d.bbox_xywh, dtype=np.float32),
            fps=pose_sequence_3d.fps,
            coordinate_space=str(getattr(pose_sequence_3d, "coordinate_space", "camera")),
        )
    except Exception as exc:
        notes = list(merged_quality.get("notes", []))
        notes.append(f"{str(overlay_variant)}_debug_overlay_failed:{exc}")
        merged_quality["notes"] = list(dict.fromkeys(str(value) for value in notes))
        if merged_quality.get("status") == "ok":
            merged_quality["status"] = "warning"
        return None
