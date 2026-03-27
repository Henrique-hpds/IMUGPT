"""Stage 5.5: lift MotionBERT-ready 2D skeletons into a temporal 3D sequence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from pose_module.interfaces import (
    COCO_17_JOINT_NAMES,
    MOTIONBERT_17_JOINT_NAMES,
    MOTIONBERT_17_PARENT_INDICES,
    MotionBERTJob,
    PoseSequence2D,
    PoseSequence3D,
)
from pose_module.io.cache import load_json_file, tail_text, write_json_file
from pose_module.model_registry import resolve_local_motionbert_artifacts
from pose_module.motionbert.adapter import (
    build_motionbert_window_batch,
    canonicalize_motionbert_output,
    merge_motionbert_window_predictions,
    write_pose_sequence3d_npz,
)
from pose_module.openmmlab_runtime import select_openmmlab_launcher
from pose_module.vitpose.adapter import write_pose_sequence_npz


DEFAULT_WINDOW_SIZE = 81
DEFAULT_WINDOW_OVERLAP = 0.5
DEFAULT_INCLUDE_CONFIDENCE = True
DEFAULT_BACKEND_NAME = "mmpose_motionbert"
DEFAULT_FALLBACK_BACKEND_NAME = "motionbert_heuristic_baseline"
DEFAULT_SEQUENCE_BATCH_SIZE = 32

MotionBERTPredictor = Callable[[np.ndarray], Union[np.ndarray, Mapping[str, Any]]]

_H36M_17_JOINT_NAMES = (
    "root",
    "right_hip",
    "right_knee",
    "right_foot",
    "left_hip",
    "left_knee",
    "left_foot",
    "spine",
    "thorax",
    "neck_base",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)

_MB17_NAME_ALIASES = {
    "pelvis": ("pelvis", "root"),
    "left_hip": ("left_hip",),
    "right_hip": ("right_hip",),
    "spine": ("spine",),
    "left_knee": ("left_knee",),
    "right_knee": ("right_knee",),
    "thorax": ("thorax",),
    "left_ankle": ("left_ankle", "left_foot"),
    "right_ankle": ("right_ankle", "right_foot"),
    "neck": ("neck", "neck_base"),
    "head": ("head",),
    "left_shoulder": ("left_shoulder",),
    "right_shoulder": ("right_shoulder",),
    "left_elbow": ("left_elbow",),
    "right_elbow": ("right_elbow",),
    "left_wrist": ("left_wrist",),
    "right_wrist": ("right_wrist",),
}

_H36M17_NAME_ALIASES = {
    "root": ("root", "pelvis"),
    "right_hip": ("right_hip",),
    "right_knee": ("right_knee",),
    "right_foot": ("right_foot", "right_ankle"),
    "left_hip": ("left_hip",),
    "left_knee": ("left_knee",),
    "left_foot": ("left_foot", "left_ankle"),
    "spine": ("spine",),
    "thorax": ("thorax",),
    "neck_base": ("neck_base", "neck"),
    "head": ("head",),
    "left_shoulder": ("left_shoulder",),
    "left_elbow": ("left_elbow",),
    "left_wrist": ("left_wrist",),
    "right_shoulder": ("right_shoulder",),
    "right_elbow": ("right_elbow",),
    "right_wrist": ("right_wrist",),
}

_POSE_LIFTER_AXIS_ORDER = (0, 2, 1)
_POSE_LIFTER_AXIS_SIGN = (-1.0, 1.0, -1.0)
_DEFAULT_IMAGE_SIZE_HW = (256, 256)
_IMPUTED_2D_CONFIDENCE_FLOOR = 0.05
_LOWER_LIMB_IMPUTED_CONFIDENCE = 0.05
_LOWER_LIMB_JOINT_NAMES = (
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
_LOWER_LIMB_JOINT_INDICES = tuple(MOTIONBERT_17_JOINT_NAMES.index(name) for name in _LOWER_LIMB_JOINT_NAMES)

_DEPTH_PRIORS = {
    "pelvis": 0.00,
    "left_hip": 0.00,
    "right_hip": 0.00,
    "spine": 0.04,
    "left_knee": 0.03,
    "right_knee": 0.03,
    "thorax": 0.08,
    "left_ankle": 0.05,
    "right_ankle": 0.05,
    "neck": 0.10,
    "head": 0.14,
    "left_shoulder": 0.06,
    "right_shoulder": 0.06,
    "left_elbow": 0.08,
    "right_elbow": 0.08,
    "left_wrist": 0.10,
    "right_wrist": 0.10,
}


def run_motionbert_lifter(
    sequence: PoseSequence2D,
    *,
    output_dir: str | Path,
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_overlap: float = DEFAULT_WINDOW_OVERLAP,
    include_confidence: bool = DEFAULT_INCLUDE_CONFIDENCE,
    predictor: Optional[MotionBERTPredictor] = None,
    backend_name: Optional[str] = None,
    checkpoint: Optional[str] = None,
    config_path: Optional[str] = None,
    device: str = "auto",
    env_name: str = "openmmlab",
    allow_fallback_backend: bool = False,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_backend_name = str(
        backend_name or ("motionbert_callable_backend" if predictor is not None else DEFAULT_BACKEND_NAME)
    )
    resolved_artifacts = (
        None
        if predictor is not None
        else resolve_local_motionbert_artifacts(
            config_path=config_path,
            checkpoint_path=checkpoint,
        )
    )

    job = MotionBERTJob(
        clip_id=str(sequence.clip_id),
        output_dir=str(output_dir.resolve()),
        window_size=int(window_size),
        window_overlap=float(window_overlap),
        include_confidence=bool(include_confidence),
        backend_name=requested_backend_name,
        checkpoint=None if resolved_artifacts is None else str(resolved_artifacts.checkpoint_path),
        config_path=None if resolved_artifacts is None else str(resolved_artifacts.config_path),
        device=str(device),
        pose2d_source=str(sequence.source),
        image_width=None if image_width is None else int(image_width),
        image_height=None if image_height is None else int(image_height),
    )

    write_pose_sequence_npz(sequence, job.input_pose2d_path)

    if predictor is not None:
        result = _run_motionbert_callable_backend(
            sequence,
            job=job,
            predictor=predictor,
            backend_name=requested_backend_name,
        )
        write_json_file(result["run_report"], job.run_report_path)
        return result

    backend_run = run_motionbert_backend_job(
        job=job,
        env_name=str(env_name),
        output_dir=output_dir,
    )
    if backend_run.get("status") == "ok":
        return {
            "status": str(backend_run["status"]),
            "pose_sequence": _load_pose_sequence3d_npz(job.pose3d_npz_path),
            "quality_report": dict(backend_run["quality_report"]),
            "artifacts": dict(backend_run["artifacts"]),
            "run_report": dict(backend_run),
        }

    if not bool(allow_fallback_backend):
        raise RuntimeError(str(backend_run.get("error", "motionbert_backend_failed")))

    result = _run_motionbert_callable_backend(
        sequence,
        job=job,
        predictor=None,
        backend_name=DEFAULT_FALLBACK_BACKEND_NAME,
    )
    fallback_notes = list(result["quality_report"].get("notes", []))
    fallback_notes.append(
        "real_motionbert_backend_failed:" + str(backend_run.get("error", "unknown_backend_error"))
    )
    result["quality_report"]["notes"] = list(dict.fromkeys(fallback_notes))
    result["quality_report"]["status"] = "warning"
    result["run_report"] = {
        **result["run_report"],
        "status": "warning",
        "requested_backend": requested_backend_name,
        "fallback_trigger": str(backend_run.get("error", "unknown_backend_error")),
        "backend_attempt": dict(backend_run),
        "quality_report": dict(result["quality_report"]),
    }
    write_json_file(result["run_report"], job.run_report_path)
    return result


def run_motionbert_backend_job(
    *,
    job: MotionBERTJob,
    env_name: str,
    output_dir: str | Path,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    job_json_path = output_dir / "motionbert_backend_job.json"
    result_json_path = output_dir / "motionbert_backend_result.json"
    write_json_file(job.to_dict(), job_json_path)

    repo_root = Path(__file__).resolve().parents[2]
    launcher, probe_diagnostics = select_openmmlab_launcher(
        str(env_name),
        cwd=repo_root,
        probe_code=(
            "import mmpose, torch; from mmpose.apis import init_model; "
            "from mmengine.dataset import Compose; print('ok')"
        ),
    )
    if launcher is None:
        backend_run = {
            "status": "fail",
            "quality_report": {},
            "artifacts": {
                "pose3d_npz_path": str(job.pose3d_npz_path.resolve()),
                "pose3d_raw_keypoints_path": str(job.raw_keypoints_3d_path.resolve()),
                "motionbert_run_json_path": str(job.run_report_path.resolve()),
            },
            "error": "No Python launcher with mmpose pose-lifter support available.",
            "env_name": str(env_name),
            "backend": {
                "launcher": None,
                "probe_diagnostics": probe_diagnostics,
            },
            "returncode": 1,
        }
        write_json_file(backend_run, job.run_report_path)
        return backend_run

    command = list(launcher["prefix"]) + [
        "-m",
        "pose_module.motionbert.lifter",
        "--job-json",
        str(job_json_path.resolve()),
        "--result-json",
        str(result_json_path.resolve()),
    ]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(repo_root.resolve())
        if existing_pythonpath == ""
        else str(repo_root.resolve()) + os.pathsep + existing_pythonpath
    )
    completed = subprocess.run(
        command,
        cwd=str(repo_root.resolve()),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    if result_json_path.exists():
        backend_run = load_json_file(result_json_path)
    else:
        backend_run = {
            "status": "fail",
            "quality_report": {},
            "artifacts": {
                "pose3d_npz_path": str(job.pose3d_npz_path.resolve()),
                "pose3d_raw_keypoints_path": str(job.raw_keypoints_3d_path.resolve()),
                "motionbert_run_json_path": str(job.run_report_path.resolve()),
            },
            "error": "motionbert_backend_result_json_missing",
        }

    backend_run["env_name"] = str(env_name)
    backend_run["launcher"] = {
        "name": str(launcher["name"]),
        "python": str(launcher["python"]),
    }
    backend_run["probe_diagnostics"] = probe_diagnostics
    backend_run["command"] = command
    backend_run["returncode"] = int(completed.returncode)
    if completed.stdout.strip():
        backend_run["stdout_tail"] = tail_text(completed.stdout, max_chars=8000)
    if completed.stderr.strip():
        backend_run["stderr_tail"] = tail_text(completed.stderr, max_chars=8000)
    if completed.returncode != 0 and backend_run.get("status") == "ok":
        backend_run["status"] = "fail"
        backend_run["error"] = "motionbert_backend_process_failed"

    write_json_file(backend_run, job.run_report_path)
    return backend_run


def run_motionbert_backend(job: MotionBERTJob) -> Dict[str, Any]:
    import torch
    from mmengine.dataset import Compose, pseudo_collate
    from mmengine.registry import init_default_scope
    from mmengine.structures import InstanceData
    from mmpose.apis import init_model
    from mmpose.apis import convert_keypoint_definition, extract_pose_sequence
    from mmpose.apis.inference_3d import collate_pose_sequence
    from mmpose.structures import PoseDataSample

    if job.config_path in (None, "") or job.checkpoint in (None, ""):
        raise FileNotFoundError("MotionBERT config/checkpoint could not be resolved.")

    sequence = _load_pose_sequence2d_npz(job.input_pose2d_path)
    model = init_model(
        config=str(job.config_path),
        checkpoint=str(job.checkpoint),
        device=_resolve_device(str(job.device)),
    )
    dataset_meta = dict(model.dataset_meta or {})
    backend_joint_names = _resolve_lifter_joint_names(
        dataset_meta,
        joint_count=len(MOTIONBERT_17_JOINT_NAMES),
    )
    conversion_target_joint_names, conversion_weights = _build_lifter_input_conversion(
        src_joint_names=tuple(str(name) for name in sequence.joint_names_2d),
        dst_dataset_name=str(dataset_meta.get("dataset_name", "")),
        convert_keypoint_definition=convert_keypoint_definition,
    )
    image_size_hw = _resolve_image_size_hw(job=job, sequence=sequence)
    causal, effective_window_size, seq_step = _lift_dataset_params(model)
    filled_sequence_xy = _fill_missing_keypoints_for_lifter(
        sequence.keypoints_xy,
        sequence.confidence,
    )
    observed_mask_sequence = np.asarray(sequence.resolved_observed_mask(), dtype=bool)
    imputed_mask_sequence = np.asarray(sequence.resolved_imputed_mask(), dtype=bool)

    pred_frames: list[np.ndarray | None] = []
    pred_confidence: list[np.ndarray | None] = []
    pred_observed_masks: list[np.ndarray | None] = []
    pred_imputed_masks: list[np.ndarray | None] = []
    pose_history: list[list[Any]] = []

    for frame_index in range(sequence.num_frames):
        keypoints_xy = np.asarray(filled_sequence_xy[frame_index], dtype=np.float32)
        confidence = np.asarray(sequence.confidence[frame_index], dtype=np.float32)
        original_observed_mask = np.asarray(observed_mask_sequence[frame_index], dtype=bool)
        original_imputed_mask = np.asarray(imputed_mask_sequence[frame_index], dtype=bool)
        filled_valid_mask = np.isfinite(keypoints_xy).all(axis=1)

        if not np.any(filled_valid_mask):
            pose_history.append([])
            pred_frames.append(None)
            pred_confidence.append(None)
            pred_observed_masks.append(None)
            pred_imputed_masks.append(None)
            continue

        converted_keypoints_xy = _apply_linear_conversion(conversion_weights, keypoints_xy)
        effective_confidence, effective_imputed_mask = _apply_lifter_imputation_confidence_policy(
            confidence=confidence,
            original_observed_mask=original_observed_mask,
            original_imputed_mask=original_imputed_mask,
            filled_valid_mask=filled_valid_mask,
        )
        converted_mask = _convert_mask(filled_valid_mask, conversion_weights)
        converted_observed_mask = _convert_mask(original_observed_mask, conversion_weights)
        converted_imputed_mask = _convert_mask(effective_imputed_mask, conversion_weights)
        converted_confidence = _convert_confidence(
            effective_confidence,
            weights=conversion_weights,
            converted_mask=converted_mask,
        )
        if not bool(job.include_confidence):
            converted_confidence[converted_mask] = 1.0

        if not np.any(converted_mask):
            pose_history.append([])
            pred_frames.append(None)
            pred_confidence.append(None)
            pred_observed_masks.append(None)
            pred_imputed_masks.append(None)
            continue

        pose_history.append(
            [
                _build_pose_lifter_2d_sample(
                    PoseDataSample=PoseDataSample,
                    InstanceData=InstanceData,
                    keypoints=converted_keypoints_xy,
                    confidence=converted_confidence,
                    mask=converted_mask,
                    track_id=0,
                )
            ]
        )

        pose_seq_2d = extract_pose_sequence(
            pose_history,
            frame_idx=frame_index,
            causal=causal,
            seq_len=effective_window_size,
            step=seq_step,
        )
        with torch.no_grad():
            lift_results = _inference_pose_lifter_model_with_fixed_targets(
                model,
                pose_seq_2d,
                collate_pose_sequence=collate_pose_sequence,
                Compose=Compose,
                pseudo_collate=pseudo_collate,
                init_default_scope=init_default_scope,
                PoseDataSample=PoseDataSample,
                image_size=image_size_hw,
                norm_pose_2d=True,
            )

        if not lift_results:
            pred_frames.append(None)
            pred_confidence.append(None)
            pred_observed_masks.append(None)
            pred_imputed_masks.append(None)
            continue

        first = lift_results[0]
        if not hasattr(first, "pred_instances"):
            pred_frames.append(None)
            pred_confidence.append(None)
            pred_observed_masks.append(None)
            pred_imputed_masks.append(None)
            continue

        pred_instances = first.pred_instances
        keypoints_3d = _postprocess_lifter_keypoints_3d(
            getattr(pred_instances, "keypoints", np.empty((0, 3), dtype=np.float32)),
            causal=causal,
        )
        if keypoints_3d.size == 0:
            pred_frames.append(None)
            pred_confidence.append(None)
            pred_observed_masks.append(None)
            pred_imputed_masks.append(None)
            continue

        canonical_keypoints_3d = _canonicalize_backend_prediction_array(
            keypoints_3d[None, ...],
            joint_names=backend_joint_names,
        )[0]
        canonical_input_confidence = _canonicalize_backend_vector_to_mb17(
            converted_confidence,
            joint_names=backend_joint_names,
        )
        canonical_observed_mask = _canonicalize_backend_vector_to_mb17(
            converted_observed_mask.astype(np.float32),
            joint_names=backend_joint_names,
        ) > 0.0
        canonical_imputed_mask = _canonicalize_backend_vector_to_mb17(
            converted_imputed_mask.astype(np.float32),
            joint_names=backend_joint_names,
        ) > 0.0
        canonical_input_mask = _canonicalize_backend_vector_to_mb17(
            converted_mask.astype(np.float32),
            joint_names=backend_joint_names,
        ) > 0.0

        score_values = getattr(pred_instances, "keypoint_scores", None)
        if score_values is None:
            canonical_scores = np.ones((len(MOTIONBERT_17_JOINT_NAMES),), dtype=np.float32)
        else:
            scores = _select_lifter_output_timestep(
                score_values,
                causal=causal,
                temporal_ndim=2,
            )
            canonical_scores = _canonicalize_backend_vector_to_mb17(
                scores.reshape(-1),
                joint_names=backend_joint_names,
            )

        combined_confidence = np.minimum(
            canonical_input_confidence.astype(np.float32, copy=False),
            canonical_scores.astype(np.float32, copy=False),
        )
        valid_3d_mask = (
            np.isfinite(canonical_keypoints_3d).all(axis=1)
            & canonical_input_mask
            & (combined_confidence > 0.0)
        )
        canonical_keypoints_3d[~valid_3d_mask] = np.nan
        combined_confidence[~valid_3d_mask] = 0.0
        canonical_observed_mask &= valid_3d_mask
        canonical_imputed_mask &= valid_3d_mask

        pred_frames.append(canonical_keypoints_3d.astype(np.float32, copy=False))
        pred_confidence.append(combined_confidence.astype(np.float32, copy=False))
        pred_observed_masks.append(canonical_observed_mask.astype(bool, copy=False))
        pred_imputed_masks.append(canonical_imputed_mask.astype(bool, copy=False))

    joint_positions_xyz = []
    joint_confidence = []
    joint_observed_mask = []
    joint_imputed_mask = []
    num_joints = len(MOTIONBERT_17_JOINT_NAMES)
    for frame_points_xyz, frame_confidence, frame_observed_mask, frame_imputed_mask in zip(
        pred_frames,
        pred_confidence,
        pred_observed_masks,
        pred_imputed_masks,
    ):
        if (
            frame_points_xyz is None
            or frame_confidence is None
            or frame_observed_mask is None
            or frame_imputed_mask is None
        ):
            joint_positions_xyz.append(np.full((num_joints, 3), np.nan, dtype=np.float32))
            joint_confidence.append(np.zeros((num_joints,), dtype=np.float32))
            joint_observed_mask.append(np.zeros((num_joints,), dtype=bool))
            joint_imputed_mask.append(np.zeros((num_joints,), dtype=bool))
            continue
        joint_positions_xyz.append(np.asarray(frame_points_xyz, dtype=np.float32))
        joint_confidence.append(np.asarray(frame_confidence, dtype=np.float32))
        joint_observed_mask.append(np.asarray(frame_observed_mask, dtype=bool))
        joint_imputed_mask.append(np.asarray(frame_imputed_mask, dtype=bool))

    joint_positions_xyz = np.stack(joint_positions_xyz, axis=0).astype(np.float32)
    joint_confidence = np.stack(joint_confidence, axis=0).astype(np.float32)
    joint_observed_mask = np.stack(joint_observed_mask, axis=0).astype(bool)
    joint_imputed_mask = np.stack(joint_imputed_mask, axis=0).astype(bool)
    pose_sequence = PoseSequence3D(
        clip_id=str(sequence.clip_id),
        fps=None if sequence.fps is None else float(sequence.fps),
        fps_original=None if sequence.fps_original is None else float(sequence.fps_original),
        joint_names_3d=list(MOTIONBERT_17_JOINT_NAMES),
        joint_positions_xyz=joint_positions_xyz,
        joint_confidence=joint_confidence,
        skeleton_parents=list(MOTIONBERT_17_PARENT_INDICES),
        frame_indices=np.asarray(sequence.frame_indices, dtype=np.int32),
        timestamps_sec=np.asarray(sequence.timestamps_sec, dtype=np.float32),
        source=f"{sequence.source}_{job.backend_name}",
        coordinate_space="pose_lifter_aligned",
        observed_mask=joint_observed_mask,
        imputed_mask=joint_imputed_mask,
    )
    np.save(job.raw_keypoints_3d_path, np.asarray(pose_sequence.joint_positions_xyz, dtype=np.float32))
    write_pose_sequence3d_npz(pose_sequence, job.pose3d_npz_path)

    notes = []
    if int(effective_window_size) != int(job.window_size):
        notes.append(f"window_size_adjusted_to_pose_lifter_seq_len:{effective_window_size}")
    if int(seq_step) != max(1, int(round(float(job.window_size) * (1.0 - float(job.window_overlap))))):
        notes.append(f"pose_lifter_seq_step:{seq_step}")
    if backend_joint_names is not None:
        notes.append(
            "backend_joint_order_canonicalized_from:"
            + ",".join(str(name) for name in backend_joint_names)
        )
    notes.append("official_pose_lifter_api_used")
    notes.append("input_sequence_converted_before_lifting:" + ",".join(conversion_target_joint_names))
    notes.append(
        "coordinate_transform_applied:"
        + ",".join(str(value) for value in _POSE_LIFTER_AXIS_ORDER)
        + "/"
        + ",".join(str(value) for value in _POSE_LIFTER_AXIS_SIGN)
    )

    quality_report = _build_motionbert_quality_report(
        pose_sequence=pose_sequence,
        backend_name=str(job.backend_name),
        include_confidence=bool(job.include_confidence),
        fallback_backend_used=False,
        requested_window_overlap=float(job.window_overlap),
        effective_window_size=int(effective_window_size),
        num_windows=int(sequence.num_frames),
        input_channels=2,
        notes=notes,
    )

    return {
        "status": str(quality_report["status"]),
        "quality_report": quality_report,
        "artifacts": {
            "pose3d_npz_path": str(job.pose3d_npz_path.resolve()),
            "pose3d_raw_keypoints_path": str(job.raw_keypoints_3d_path.resolve()),
            "motionbert_run_json_path": str(job.run_report_path.resolve()),
        },
        "backend": {
            "name": str(job.backend_name),
            "config_path": str(job.config_path),
            "checkpoint_path": str(job.checkpoint),
            "device": _resolve_device(str(job.device)),
            "mode": "mmpose_pose_lifter_api",
            "effective_window_size": int(effective_window_size),
            "seq_step": int(seq_step),
            "image_size_hw": [int(image_size_hw[0]), int(image_size_hw[1])],
            "lifter_dataset_name": str(dataset_meta.get("dataset_name", "")),
        },
        "error": None,
    }


def _inference_pose_lifter_model_with_fixed_targets(
    model: Any,
    pose_results_2d: Sequence[Sequence[Any]],
    *,
    collate_pose_sequence: Callable[..., Any],
    Compose: Any,
    pseudo_collate: Callable[..., Any],
    init_default_scope: Callable[..., Any],
    PoseDataSample: Any,
    image_size: tuple[int, int] | None = None,
    norm_pose_2d: bool = False,
) -> list[Any]:
    """Local copy of MMPose pose-lifter inference with temporal target fix.

    MMPose's helper currently hardcodes `lifting_target` as shape `(1, K, 3)`,
    which breaks MotionBERT test-time inference when the temporal context is
    longer than one frame because `factor` is shaped as `(T,)`.
    """

    init_default_scope(model.cfg.get("default_scope", "mmpose"))
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    causal = bool(model.cfg.test_dataloader.dataset.get("causal", False))
    target_idx = -1 if causal else len(pose_results_2d) // 2

    dataset_info = model.dataset_meta
    if dataset_info is not None:
        if "stats_info" in dataset_info:
            bbox_center = dataset_info["stats_info"]["bbox_center"]
            bbox_scale = dataset_info["stats_info"]["bbox_scale"]
        elif norm_pose_2d:
            bbox_center = np.zeros((1, 2), dtype=np.float32)
            bbox_scale = 0.0
            num_bbox = 0
            for pose_res in pose_results_2d:
                for data_sample in pose_res:
                    for bbox in data_sample.pred_instances.bboxes:
                        bbox_center += np.array(
                            [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]],
                            dtype=np.float32,
                        )
                        bbox_scale += max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))
                        num_bbox += 1
            bbox_center /= max(num_bbox, 1)
            bbox_scale /= max(num_bbox, 1)
        else:
            bbox_center = None
            bbox_scale = None
    else:
        bbox_center = None
        bbox_scale = None

    pose_results_2d_copy = []
    for pose_res in pose_results_2d:
        pose_res_copy = []
        for data_sample in pose_res:
            data_sample_copy = PoseDataSample()
            data_sample_copy.gt_instances = data_sample.gt_instances.clone()
            data_sample_copy.pred_instances = data_sample.pred_instances.clone()
            data_sample_copy.track_id = data_sample.track_id
            kpts = data_sample.pred_instances.keypoints
            bboxes = data_sample.pred_instances.bboxes
            keypoints = []
            for k in range(len(kpts)):
                kpt = kpts[k]
                if norm_pose_2d:
                    bbox = bboxes[k]
                    center = np.array(
                        [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]],
                        dtype=np.float32,
                    )
                    scale = max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))
                    keypoints.append((kpt[:, :2] - center) / scale * bbox_scale + bbox_center)
                else:
                    keypoints.append(kpt[:, :2])
            data_sample_copy.pred_instances.set_field(np.array(keypoints), "keypoints")
            pose_res_copy.append(data_sample_copy)
        pose_results_2d_copy.append(pose_res_copy)

    pose_sequences_2d = collate_pose_sequence(
        pose_results_2d_copy,
        with_track_id=True,
        target_frame=target_idx,
    )
    if not pose_sequences_2d:
        return []

    data_list = []
    for pose_seq in pose_sequences_2d:
        data_info: Dict[str, Any] = {}
        keypoints_2d = pose_seq.pred_instances.keypoints
        keypoints_2d = np.squeeze(keypoints_2d, axis=0) if keypoints_2d.ndim == 4 else keypoints_2d
        t_steps, num_joints, _ = keypoints_2d.shape

        data_info["keypoints"] = keypoints_2d
        data_info["keypoints_visible"] = np.ones((t_steps, num_joints), dtype=np.float32)
        data_info["lifting_target"] = np.zeros((t_steps, num_joints, 3), dtype=np.float32)
        data_info["factor"] = np.zeros((t_steps,), dtype=np.float32)
        data_info["lifting_target_visible"] = np.ones((t_steps, num_joints, 1), dtype=np.float32)

        if image_size is not None:
            if len(image_size) != 2:
                raise ValueError("image_size must contain exactly two values.")
            data_info["camera_param"] = dict(w=image_size[0], h=image_size[1])

        data_info.update(model.dataset_meta)
        data_list.append(pipeline(data_info))

    if len(data_list) == 0:
        return []
    batch = pseudo_collate(data_list)
    return model.test_step(batch)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run MotionBERT pose-lifter backend.")
    parser.add_argument("--job-json", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    args = parser.parse_args(argv)

    job = MotionBERTJob.from_dict(load_json_file(args.job_json))
    try:
        result = run_motionbert_backend(job)
    except Exception as exc:
        result = {
            "status": "fail",
            "quality_report": {},
            "artifacts": {
                "pose3d_npz_path": str(job.pose3d_npz_path.resolve()),
                "pose3d_raw_keypoints_path": str(job.raw_keypoints_3d_path.resolve()),
                "motionbert_run_json_path": str(job.run_report_path.resolve()),
            },
            "backend": {
                "name": str(job.backend_name),
                "config_path": job.config_path,
                "checkpoint_path": job.checkpoint,
                "device": str(job.device),
            },
            "error": str(exc),
        }

    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return 0 if result.get("status") == "ok" else 1


def _run_motionbert_callable_backend(
    sequence: PoseSequence2D,
    *,
    job: MotionBERTJob,
    predictor: Optional[MotionBERTPredictor],
    backend_name: str,
) -> Dict[str, Any]:
    batch = build_motionbert_window_batch(
        sequence,
        window_size=int(job.window_size),
        window_overlap=float(job.window_overlap),
        include_confidence=bool(job.include_confidence),
    )
    raw_backend_output = (
        _heuristic_motionbert_predict(batch.inputs) if predictor is None else predictor(batch.inputs)
    )
    canonical_predictions = canonicalize_motionbert_output(
        raw_backend_output,
        expected_batch_size=batch.num_windows,
        expected_window_size=batch.window_size,
    )
    fused_predictions = merge_motionbert_window_predictions(
        canonical_predictions,
        batch,
        num_frames=int(sequence.num_frames),
    )

    pose_sequence = PoseSequence3D(
        clip_id=str(sequence.clip_id),
        fps=None if sequence.fps is None else float(sequence.fps),
        fps_original=None if sequence.fps_original is None else float(sequence.fps_original),
        joint_names_3d=list(MOTIONBERT_17_JOINT_NAMES),
        joint_positions_xyz=fused_predictions.astype(np.float32, copy=False),
        joint_confidence=np.asarray(sequence.confidence, dtype=np.float32),
        skeleton_parents=list(MOTIONBERT_17_PARENT_INDICES),
        frame_indices=np.asarray(sequence.frame_indices, dtype=np.int32),
        timestamps_sec=np.asarray(sequence.timestamps_sec, dtype=np.float32),
        source=f"{sequence.source}_{backend_name}",
        coordinate_space="camera",
        observed_mask=np.asarray(sequence.resolved_observed_mask(), dtype=bool),
        imputed_mask=np.asarray(sequence.resolved_imputed_mask(), dtype=bool),
    )

    np.save(job.raw_keypoints_3d_path, np.asarray(pose_sequence.joint_positions_xyz, dtype=np.float32))
    write_pose_sequence3d_npz(pose_sequence, job.pose3d_npz_path)

    quality_report = _build_motionbert_quality_report(
        pose_sequence=pose_sequence,
        backend_name=str(backend_name),
        include_confidence=bool(job.include_confidence),
        fallback_backend_used=predictor is None,
        requested_window_overlap=float(job.window_overlap),
        effective_window_size=int(job.window_size),
        num_windows=int(batch.num_windows),
        input_channels=int(batch.inputs.shape[-1]),
        notes=[],
    )
    artifacts = {
        "pose3d_npz_path": str(job.pose3d_npz_path.resolve()),
        "pose3d_raw_keypoints_path": str(job.raw_keypoints_3d_path.resolve()),
        "motionbert_run_json_path": str(job.run_report_path.resolve()),
    }
    run_report = {
        "status": str(quality_report["status"]),
        "quality_report": quality_report,
        "artifacts": dict(artifacts),
        "backend": {
            "name": str(backend_name),
            "mode": "callable" if predictor is not None else "heuristic_baseline",
            "num_windows": int(batch.num_windows),
            "window_size": int(batch.window_size),
        },
        "error": None,
    }
    return {
        "status": str(quality_report["status"]),
        "pose_sequence": pose_sequence,
        "quality_report": quality_report,
        "artifacts": artifacts,
        "run_report": run_report,
    }


def _extract_backend_window_predictions(
    results: Sequence[Any],
    *,
    expected_window_size: int,
    joint_names: Sequence[str] | None = None,
) -> List[np.ndarray]:
    predictions: List[np.ndarray] = []
    for result in results:
        pred_instances = getattr(result, "pred_instances", None)
        if pred_instances is None:
            raise RuntimeError("MotionBERT result missing pred_instances.")
        keypoints = None
        if hasattr(pred_instances, "keypoints"):
            keypoints = pred_instances.keypoints
        elif hasattr(pred_instances, "keypoints_3d"):
            keypoints = pred_instances.keypoints_3d
        if keypoints is None:
            raise RuntimeError("MotionBERT result missing keypoints/keypoints_3d.")

        array = np.asarray(keypoints, dtype=np.float32)
        while array.ndim > 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3 or array.shape[1:] != (len(MOTIONBERT_17_JOINT_NAMES), 3):
            raise RuntimeError(f"Unexpected MotionBERT keypoint shape: {array.shape}")
        if int(array.shape[0]) != int(expected_window_size):
            raise RuntimeError(
                f"Unexpected MotionBERT temporal window shape: {array.shape[0]} != {expected_window_size}"
            )
        predictions.append(
            _canonicalize_backend_prediction_array(
                array.astype(np.float32, copy=False),
                joint_names=joint_names,
            )
        )
    return predictions


def _resolve_lifter_joint_names(
    dataset_meta: Mapping[str, Any],
    joint_count: int,
) -> List[str]:
    keypoint_id2name = dataset_meta.get("keypoint_id2name")
    if isinstance(keypoint_id2name, Mapping) and len(keypoint_id2name) > 0:
        try:
            ordered_keys = sorted((int(key), str(value)) for key, value in keypoint_id2name.items())
            if len(ordered_keys) == int(joint_count):
                return [value for _, value in ordered_keys]
        except Exception:
            pass

    keypoint_names = dataset_meta.get("keypoint_names")
    if isinstance(keypoint_names, (list, tuple)) and len(keypoint_names) == int(joint_count):
        return [str(value) for value in keypoint_names]

    keypoint_name2id = dataset_meta.get("keypoint_name2id")
    if isinstance(keypoint_name2id, Mapping) and len(keypoint_name2id) > 0:
        try:
            ordered_items = sorted((int(value), str(key)) for key, value in keypoint_name2id.items())
            if len(ordered_items) == int(joint_count):
                return [name for _, name in ordered_items]
        except Exception:
            pass

    dataset_name = _normalize_lifter_dataset_name(str(dataset_meta.get("dataset_name", "")))
    if dataset_name == "h36m" and int(joint_count) == len(_H36M_17_JOINT_NAMES):
        return list(_H36M_17_JOINT_NAMES)
    return [f"joint_{index}" for index in range(int(joint_count))]


def _resolve_backend_joint_names(dataset_meta: Mapping[str, Any]) -> Optional[List[str]]:
    joint_count = int(dataset_meta.get("num_keypoints", len(_H36M_17_JOINT_NAMES)))
    return _resolve_lifter_joint_names(dataset_meta, joint_count)


def _normalize_lifter_dataset_name(name: str) -> str:
    token = str(name).strip().lower()
    if not token:
        return token
    if "wholebody" in token and "coco" in token:
        return "coco_wholebody"
    if "coco" in token:
        return "coco"
    if "aic" in token:
        return "aic"
    if "posetrack" in token:
        return "posetrack18"
    if "crowdpose" in token:
        return "crowdpose"
    if "h36m" in token or "human36m" in token:
        return "h36m"
    if "mb17" in token or "motionbert17" in token:
        return "mb17"
    return token


def _infer_sequence_dataset_name(joint_names: Sequence[str]) -> str:
    normalized_joint_names = tuple(str(name) for name in joint_names)
    if normalized_joint_names == tuple(COCO_17_JOINT_NAMES):
        return "coco"
    if normalized_joint_names == tuple(MOTIONBERT_17_JOINT_NAMES):
        return "mb17"
    if normalized_joint_names == tuple(_H36M_17_JOINT_NAMES):
        return "h36m"
    return ""


def _build_mb17_to_h36m_weights() -> np.ndarray:
    src_index = {name: idx for idx, name in enumerate(MOTIONBERT_17_JOINT_NAMES)}
    dst_index = {name: idx for idx, name in enumerate(_H36M_17_JOINT_NAMES)}
    weights = np.zeros((len(_H36M_17_JOINT_NAMES), len(MOTIONBERT_17_JOINT_NAMES)), dtype=np.float32)
    for dst_joint_name, aliases in _H36M17_NAME_ALIASES.items():
        dst_joint_index = dst_index[dst_joint_name]
        for alias in aliases:
            if alias in src_index:
                weights[dst_joint_index, src_index[alias]] = np.float32(1.0)
                break
    return weights


def _derive_linear_conversion_weights(
    *,
    src_joint_count: int,
    src_dataset_name: str,
    dst_dataset_name: str,
    convert_keypoint_definition: Callable[..., Any],
) -> np.ndarray:
    if not dst_dataset_name or src_dataset_name == dst_dataset_name:
        return np.eye(src_joint_count, dtype=np.float32)

    weights: np.ndarray | None = None
    for src_index in range(src_joint_count):
        probe = np.zeros((1, src_joint_count, 2), dtype=np.float32)
        probe[0, src_index, 0] = 1.0
        converted = convert_keypoint_definition(probe, src_dataset_name, dst_dataset_name)
        converted_arr = np.asarray(converted, dtype=np.float32)
        if converted_arr.ndim == 3:
            converted_arr = converted_arr[0]
        if converted_arr.ndim != 2 or converted_arr.shape[1] < 1:
            raise RuntimeError(
                "convert_keypoint_definition returned invalid shape while deriving mapping: "
                f"{converted_arr.shape}"
            )
        if weights is None:
            weights = np.zeros((converted_arr.shape[0], src_joint_count), dtype=np.float32)
        elif converted_arr.shape[0] != weights.shape[0]:
            raise RuntimeError(
                "convert_keypoint_definition returned inconsistent destination joint count across probes."
            )
        weights[:, src_index] = converted_arr[:, 0]

    if weights is None:
        raise RuntimeError("Unable to derive keypoint conversion mapping.")
    weights[np.abs(weights) < 1e-6] = 0.0
    return weights.astype(np.float32)


def _build_lifter_input_conversion(
    *,
    src_joint_names: Sequence[str],
    dst_dataset_name: str,
    convert_keypoint_definition: Callable[..., Any],
) -> tuple[tuple[str, ...], np.ndarray]:
    src_dataset_name = _infer_sequence_dataset_name(src_joint_names)
    dst_dataset_name_normalized = _normalize_lifter_dataset_name(dst_dataset_name)

    if dst_dataset_name_normalized == "h36m":
        if src_dataset_name == "mb17":
            return tuple(_H36M_17_JOINT_NAMES), _build_mb17_to_h36m_weights()
        if src_dataset_name == "h36m":
            return tuple(_H36M_17_JOINT_NAMES), np.eye(len(_H36M_17_JOINT_NAMES), dtype=np.float32)
        if src_dataset_name:
            return (
                tuple(_H36M_17_JOINT_NAMES),
                _derive_linear_conversion_weights(
                    src_joint_count=len(src_joint_names),
                    src_dataset_name=src_dataset_name,
                    dst_dataset_name=dst_dataset_name_normalized,
                    convert_keypoint_definition=convert_keypoint_definition,
                ),
            )

    if not dst_dataset_name_normalized or src_dataset_name == dst_dataset_name_normalized:
        return tuple(str(name) for name in src_joint_names), np.eye(len(src_joint_names), dtype=np.float32)

    raise RuntimeError(
        "Unsupported pose lifter input conversion path: "
        f"{src_dataset_name or 'unknown'} -> {dst_dataset_name_normalized or 'unknown'}."
    )


def _apply_linear_conversion(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(weights, dtype=np.float32)
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"`values` must be shaped as (J, D), got {array.shape}.")
    if matrix.ndim != 2:
        raise ValueError(f"`weights` must be shaped as (J_dst, J_src), got {matrix.shape}.")
    if array.shape[0] != matrix.shape[1]:
        raise ValueError(
            "Keypoint conversion shape mismatch: "
            f"values J_src={array.shape[0]} but weights expects {matrix.shape[1]}."
        )
    return np.sum(matrix[:, :, None] * array[None, :, :], axis=1, dtype=np.float32).astype(np.float32)


def _convert_mask(mask_src: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask_src, dtype=bool).reshape(-1)
    matrix = np.asarray(weights, dtype=np.float32)
    if mask.shape[0] != matrix.shape[1]:
        raise ValueError(
            "Source mask length mismatch for keypoint conversion: "
            f"expected {matrix.shape[1]}, got {mask.shape[0]}."
        )
    out = np.zeros((matrix.shape[0],), dtype=bool)
    for dst_index in range(matrix.shape[0]):
        contributors = np.flatnonzero(np.abs(matrix[dst_index]) > 1e-6)
        out[dst_index] = bool(np.all(mask[contributors])) if contributors.size > 0 else False
    return out


def _convert_confidence(
    confidence_src: np.ndarray,
    *,
    weights: np.ndarray,
    converted_mask: np.ndarray,
) -> np.ndarray:
    confidence = np.asarray(confidence_src, dtype=np.float32).reshape(-1)
    matrix = np.asarray(weights, dtype=np.float32)
    if confidence.shape[0] != matrix.shape[1]:
        raise ValueError(
            "Source confidence length mismatch for keypoint conversion: "
            f"expected {matrix.shape[1]}, got {confidence.shape[0]}."
        )
    out = np.zeros((matrix.shape[0],), dtype=np.float32)
    for dst_index in range(matrix.shape[0]):
        if not bool(converted_mask[dst_index]):
            continue
        contributors = np.flatnonzero(np.abs(matrix[dst_index]) > 1e-6)
        if contributors.size == 0:
            continue
        out[dst_index] = float(np.min(confidence[contributors]))
    return out.astype(np.float32)


def _bbox_from_keypoints(keypoints: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid_mask = np.asarray(mask, dtype=bool)
    points_xy = np.asarray(keypoints, dtype=np.float32)
    if not np.any(valid_mask):
        return np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    pts = points_xy[valid_mask]
    return np.asarray(
        [
            float(np.min(pts[:, 0])),
            float(np.min(pts[:, 1])),
            float(np.max(pts[:, 0])),
            float(np.max(pts[:, 1])),
        ],
        dtype=np.float32,
    )


def _build_pose_lifter_2d_sample(
    *,
    PoseDataSample: Any,
    InstanceData: Any,
    keypoints: np.ndarray,
    confidence: np.ndarray,
    mask: np.ndarray,
    track_id: int,
) -> Any:
    sample = PoseDataSample()
    pred_instances = InstanceData()
    pred_instances.set_field(np.asarray(keypoints[None, ...], dtype=np.float32), "keypoints")
    pred_instances.set_field(np.asarray(confidence[None, ...], dtype=np.float32), "keypoint_scores")
    pred_instances.set_field(_bbox_from_keypoints(keypoints, mask)[None, ...], "bboxes")

    gt_instances = InstanceData()
    gt_instances.set_field(np.zeros((1, 4), dtype=np.float32), "bboxes")

    sample.pred_instances = pred_instances
    sample.gt_instances = gt_instances
    sample.set_field(int(track_id), "track_id")
    return sample


def _lift_dataset_params(model: Any) -> tuple[bool, int, int]:
    causal = False
    seq_step = 1
    seq_len_candidates: list[int] = []

    try:
        dataset = model.cfg.test_dataloader.dataset
    except Exception:
        dataset = None

    if dataset is not None:
        causal = bool(dataset.get("causal", False))
        seq_step = max(int(dataset.get("seq_step", 1)), 1)

        # Pose-lifter configs in MMPose may expose temporal context through
        # `multiple_target` and/or the backbone `seq_len` while keeping the
        # dataset `seq_len` at 1 for sample packing. Prefer the longest valid
        # temporal context available so we do not silently fall back to
        # frame-by-frame lifting for temporal backbones such as MotionBERT.
        for raw_value in (
            dataset.get("multiple_target"),
            dataset.get("seq_len"),
        ):
            try:
                value = int(raw_value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                seq_len_candidates.append(value)

    model_seq_len = _resolve_model_sequence_length(model, fallback=0)
    if int(model_seq_len) > 0:
        seq_len_candidates.append(int(model_seq_len))

    if len(seq_len_candidates) == 0:
        return causal, 1, seq_step
    return causal, max(seq_len_candidates), seq_step


def _resolve_image_size_hw(*, job: MotionBERTJob, sequence: PoseSequence2D) -> tuple[int, int]:
    if job.image_width is not None and job.image_height is not None:
        return max(int(job.image_height), 1), max(int(job.image_width), 1)

    bbox_xywh = np.asarray(sequence.bbox_xywh, dtype=np.float32)
    if bbox_xywh.ndim == 2 and bbox_xywh.shape[1] >= 4 and bbox_xywh.shape[0] > 0:
        width = int(np.ceil(np.nanmax(bbox_xywh[:, 0] + bbox_xywh[:, 2])))
        height = int(np.ceil(np.nanmax(bbox_xywh[:, 1] + bbox_xywh[:, 3])))
        if width > 0 and height > 0:
            return max(height, 1), max(width, 1)

    keypoints_xy = np.asarray(sequence.keypoints_xy, dtype=np.float32)
    if keypoints_xy.size > 0:
        finite_mask = np.isfinite(keypoints_xy)
        if np.any(finite_mask[..., 0]) and np.any(finite_mask[..., 1]):
            width = int(np.ceil(np.nanmax(keypoints_xy[..., 0]))) + 1
            height = int(np.ceil(np.nanmax(keypoints_xy[..., 1]))) + 1
            if width > 0 and height > 0:
                return max(height, 1), max(width, 1)

    return _DEFAULT_IMAGE_SIZE_HW


def _fill_missing_keypoints_for_lifter(
    keypoints_xy: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    points = np.asarray(keypoints_xy, dtype=np.float32)
    conf = np.asarray(confidence, dtype=np.float32)
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError(f"Expected keypoints_xy shaped as [T, J, 2], got {points.shape}.")
    if conf.shape != points.shape[:2]:
        raise ValueError(
            "Expected confidence shaped as [T, J] for lifter input filling: "
            f"got points={points.shape} confidence={conf.shape}."
        )

    filled = points.copy()
    num_frames = int(filled.shape[0])
    if num_frames == 0:
        return filled

    frame_axis = np.arange(num_frames, dtype=np.float32)
    joint_valid = np.isfinite(filled).all(axis=2) & (conf > 0.0)
    for joint_index in range(filled.shape[1]):
        for dim_index in range(filled.shape[2]):
            dim_values = filled[:, joint_index, dim_index]
            dim_valid = joint_valid[:, joint_index] & np.isfinite(dim_values)
            if not np.any(dim_valid):
                filled[:, joint_index, dim_index] = 0.0
                continue
            filled[:, joint_index, dim_index] = np.interp(
                frame_axis,
                frame_axis[dim_valid],
                dim_values[dim_valid],
            )
    return filled.astype(np.float32, copy=False)


def _apply_lifter_imputation_confidence_policy(
    *,
    confidence: np.ndarray,
    original_observed_mask: np.ndarray,
    original_imputed_mask: np.ndarray,
    filled_valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    effective_confidence = np.asarray(confidence, dtype=np.float32).copy()
    observed_mask = np.asarray(original_observed_mask, dtype=bool)
    imputed_mask = np.asarray(original_imputed_mask, dtype=bool).copy()
    valid_mask = np.asarray(filled_valid_mask, dtype=bool)

    if (
        effective_confidence.shape != observed_mask.shape
        or effective_confidence.shape != valid_mask.shape
        or imputed_mask.shape != valid_mask.shape
    ):
        raise ValueError(
            "Lifter imputation policy expects confidence, observed_mask, imputed_mask, and "
            "filled_valid_mask with the same shape; got "
            f"{effective_confidence.shape}, {observed_mask.shape}, {imputed_mask.shape}, {valid_mask.shape}."
        )

    if imputed_mask.ndim != 1:
        raise ValueError(f"Lifter imputation policy expects 1D joint masks per frame, got {imputed_mask.shape}.")

    imputed_mask |= valid_mask & ~observed_mask
    effective_confidence[imputed_mask] = np.minimum(effective_confidence[imputed_mask], np.float32(0.15))
    effective_confidence[imputed_mask] = np.maximum(
        effective_confidence[imputed_mask],
        np.float32(_IMPUTED_2D_CONFIDENCE_FLOOR),
    )
    effective_confidence[list(_LOWER_LIMB_JOINT_INDICES)] = np.where(
        imputed_mask[list(_LOWER_LIMB_JOINT_INDICES)],
        np.float32(_LOWER_LIMB_IMPUTED_CONFIDENCE),
        effective_confidence[list(_LOWER_LIMB_JOINT_INDICES)],
    )
    return effective_confidence.astype(np.float32, copy=False), imputed_mask.astype(bool, copy=False)


def _select_lifter_output_timestep(
    values: np.ndarray,
    *,
    causal: bool,
    temporal_ndim: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    while array.ndim > temporal_ndim and array.shape[0] == 1:
        array = array[0]
    if array.ndim == temporal_ndim:
        target_index = -1 if bool(causal) else int(array.shape[0] // 2)
        array = array[target_index]
    return array.astype(np.float32, copy=False)


def _postprocess_lifter_keypoints_3d(
    keypoints: np.ndarray,
    *,
    causal: bool,
) -> np.ndarray:
    array = _select_lifter_output_timestep(
        keypoints,
        causal=causal,
        temporal_ndim=3,
    )
    if array.ndim != 2 or array.shape[1] < 3:
        return np.full((0, 3), np.nan, dtype=np.float32)

    transformed = array[:, list(_POSE_LIFTER_AXIS_ORDER)].astype(np.float32)
    transformed *= np.asarray(_POSE_LIFTER_AXIS_SIGN, dtype=np.float32)[None, :]

    finite_mask = np.isfinite(transformed[:, 2])
    if np.any(finite_mask):
        transformed[:, 2] -= np.float32(np.min(transformed[finite_mask, 2]))
    return transformed


def _resolve_mb17_reorder_indices(joint_names: Sequence[str]) -> List[int]:
    normalized_joint_names = [str(name).strip().lower() for name in joint_names]
    if normalized_joint_names == [name.lower() for name in MOTIONBERT_17_JOINT_NAMES]:
        return list(range(len(MOTIONBERT_17_JOINT_NAMES)))

    joint_name_to_index = {name: index for index, name in enumerate(normalized_joint_names)}
    ordered_indices = []
    missing_targets = []
    for target_joint_name in MOTIONBERT_17_JOINT_NAMES:
        resolved_index = None
        for alias in _MB17_NAME_ALIASES[str(target_joint_name)]:
            alias_key = str(alias).strip().lower()
            if alias_key in joint_name_to_index:
                resolved_index = int(joint_name_to_index[alias_key])
                break
        if resolved_index is None:
            missing_targets.append(str(target_joint_name))
        else:
            ordered_indices.append(resolved_index)

    if missing_targets:
        raise RuntimeError(
            "MotionBERT backend output joint names are incompatible with MB17: "
            + ",".join(missing_targets)
        )
    return ordered_indices


def _canonicalize_backend_vector_to_mb17(
    values: np.ndarray,
    *,
    joint_names: Sequence[str] | None,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if joint_names is None:
        return array
    ordered_indices = _resolve_mb17_reorder_indices(joint_names)
    if array.shape[0] != len(ordered_indices):
        raise RuntimeError(
            "MotionBERT backend vector length is incompatible with the reported joint names: "
            f"{array.shape[0]} vs {len(ordered_indices)}"
        )
    return array[ordered_indices].astype(np.float32, copy=False)


def _canonicalize_backend_prediction_array(
    prediction: np.ndarray,
    *,
    joint_names: Sequence[str] | None,
) -> np.ndarray:
    array = np.asarray(prediction, dtype=np.float32)
    if joint_names is None:
        return array

    ordered_indices = _resolve_mb17_reorder_indices(joint_names)
    return array[:, ordered_indices, :].astype(np.float32, copy=False)


def _resolve_model_sequence_length(model: Any, *, fallback: int) -> int:
    backbone = getattr(model, "backbone", None)
    seq_len = getattr(backbone, "seq_len", None)
    if seq_len is None:
        seq_len = model.cfg.get("model", {}).get("backbone", {}).get("seq_len")
    if seq_len in (None, 0):
        return int(fallback)
    return int(seq_len)


def _resolve_device(device_preference: str) -> str:
    import torch

    preference = str(device_preference).strip().lower()
    if preference not in {"", "auto"}:
        return str(device_preference)
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _load_pose_sequence2d_npz(path: str | Path) -> PoseSequence2D:
    with np.load(Path(path), allow_pickle=False) as payload:
        return PoseSequence2D.from_npz_payload({key: payload[key] for key in payload.files})


def _load_pose_sequence3d_npz(path: str | Path) -> PoseSequence3D:
    with np.load(Path(path), allow_pickle=False) as payload:
        return PoseSequence3D.from_npz_payload({key: payload[key] for key in payload.files})


def _heuristic_motionbert_predict(window_inputs: np.ndarray) -> np.ndarray:
    xy = np.asarray(window_inputs[..., :2], dtype=np.float32)
    confidence = (
        np.asarray(window_inputs[..., 2], dtype=np.float32)
        if window_inputs.shape[-1] > 2
        else np.ones(window_inputs.shape[:-1], dtype=np.float32)
    )

    output = np.zeros((xy.shape[0], xy.shape[1], xy.shape[2], 3), dtype=np.float32)
    output[..., 0] = xy[..., 0]
    output[..., 1] = -xy[..., 1]

    temporal_delta = np.zeros_like(xy, dtype=np.float32)
    if xy.shape[1] > 1:
        temporal_delta[:, 1:] = xy[:, 1:] - xy[:, :-1]
    temporal_speed = np.linalg.norm(temporal_delta, axis=3)

    for joint_index, joint_name in enumerate(MOTIONBERT_17_JOINT_NAMES):
        base_depth = float(_DEPTH_PRIORS[joint_name])
        lateral_bias = 0.15 * np.abs(xy[..., joint_index, 0])
        vertical_bias = 0.10 * np.maximum(-xy[..., joint_index, 1], 0.0)
        temporal_bias = 0.25 * temporal_speed[..., joint_index]
        joint_depth = (base_depth + lateral_bias + vertical_bias + temporal_bias) * confidence[..., joint_index]
        output[..., joint_index, 2] = joint_depth.astype(np.float32, copy=False)

    return output


def _build_motionbert_quality_report(
    *,
    pose_sequence: PoseSequence3D,
    backend_name: str,
    include_confidence: bool,
    fallback_backend_used: bool,
    requested_window_overlap: float,
    effective_window_size: int,
    num_windows: int,
    input_channels: int,
    notes: Sequence[str],
) -> Dict[str, Any]:
    joint_confidence = np.asarray(pose_sequence.joint_confidence, dtype=np.float32)
    visible_joint_ratio = (
        float(np.count_nonzero(joint_confidence > 0.0) / float(joint_confidence.size))
        if joint_confidence.size > 0
        else 0.0
    )
    valid_confidence = joint_confidence[joint_confidence > 0.0]
    mean_confidence = float(np.mean(valid_confidence)) if valid_confidence.size > 0 else 0.0
    depth_values = np.asarray(pose_sequence.joint_positions_xyz[..., 2], dtype=np.float32)
    depth_variation = float(np.std(depth_values)) if depth_values.size > 0 else 0.0
    window_coverage_ratio = 1.0 if pose_sequence.num_frames > 0 else 0.0

    report_notes = list(str(value) for value in notes)
    if fallback_backend_used:
        report_notes.append("fallback_backend_without_motionbert_weights")
    if depth_variation <= 1e-4:
        report_notes.append("depth_variation_too_low")
    if mean_confidence < 0.5:
        report_notes.append("mean_confidence_below_threshold")

    status = "ok"
    if depth_values.shape[0] != pose_sequence.num_frames:
        status = "fail"
    elif fallback_backend_used or mean_confidence < 0.5 or depth_variation <= 1e-4:
        status = "warning"

    return {
        "clip_id": str(pose_sequence.clip_id),
        "status": str(status),
        "fps": None if pose_sequence.fps is None else float(pose_sequence.fps),
        "fps_original": None if pose_sequence.fps_original is None else float(pose_sequence.fps_original),
        "num_frames": int(pose_sequence.num_frames),
        "num_joints": int(pose_sequence.num_joints),
        "input_joint_format": list(MOTIONBERT_17_JOINT_NAMES),
        "output_joint_format": list(MOTIONBERT_17_JOINT_NAMES),
        "coordinate_space": str(pose_sequence.coordinate_space),
        "backend_name": str(backend_name),
        "window_size": int(effective_window_size),
        "window_overlap": float(requested_window_overlap),
        "num_windows": int(num_windows),
        "input_channels": int(input_channels),
        "include_confidence_channel": bool(include_confidence),
        "visible_joint_ratio": float(visible_joint_ratio),
        "mean_confidence": float(mean_confidence),
        "depth_variation": float(depth_variation),
        "window_coverage_ratio": float(window_coverage_ratio),
        "notes": list(dict.fromkeys(report_notes)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
