"""Anchor-catalog builder from pose3d + window-level Qwen descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from pose_module.export.ik_adapter import run_ik
from pose_module.interfaces import IKSequence, PoseSequence3D

from .export import write_json, write_jsonl
from .kimodo_generation import (
    DEFAULT_KIMODO_GENERATION_MODEL,
    _load_kimodo_runtime,
    load_catalog_entries,
    select_catalog_entries,
)
from .windowing import (
    WindowSpec,
    load_pose_manifest_entries,
    load_pose_sequence3d,
    resolve_sequence_fps,
    resolve_source_times,
    select_pose_entries,
)

ROOT2D_MIN_DISPLACEMENT_M = 0.05
HEADING_MIN_DISPLACEMENT_M = 0.10
GROUND_HEIGHT_QUANTILE = 0.05
ROOT_ALIGNMENT_TOLERANCE_M = 1e-3
IK_ALIGNMENT_TOLERANCE_M = 1e-3
TIMESTAMP_ALIGNMENT_TOLERANCE_SEC = 1e-4
IK_SEQUENCE_FILENAME = "ik_sequence.npz"
CONSTRAINT_MODE = "ik_sequence"

SMPLX22_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
IMUGPT_TO_SMPLX_JOINT_MAP = (
    ("Pelvis", "pelvis"),
    ("Left_hip", "left_hip"),
    ("Right_hip", "right_hip"),
    ("Spine1", "spine1"),
    ("Left_knee", "left_knee"),
    ("Right_knee", "right_knee"),
    ("Spine2", "spine2"),
    ("Left_ankle", "left_ankle"),
    ("Right_ankle", "right_ankle"),
    ("Spine3", "spine3"),
    ("Left_foot", "left_foot"),
    ("Right_foot", "right_foot"),
    ("Neck", "neck"),
    ("Left_collar", "left_collar"),
    ("Right_collar", "right_collar"),
    ("Head", "head"),
    ("Left_shoulder", "left_shoulder"),
    ("Right_shoulder", "right_shoulder"),
    ("Left_elbow", "left_elbow"),
    ("Right_elbow", "right_elbow"),
    ("Left_wrist", "left_wrist"),
    ("Right_wrist", "right_wrist"),
)
SUPPORT_JOINT_NAMES = ("left_ankle", "right_ankle", "left_foot", "right_foot")


@dataclass(frozen=True)
class _CachedPoseClip:
    sequence: PoseSequence3D
    source_times: np.ndarray
    smplx_source_indices: np.ndarray
    root_translation_m: np.ndarray
    root_alignment_max_error_m: float
    ik_sequence: IKSequence
    ik_path: Path
    ik_smplx_source_indices: np.ndarray
    ik_root_alignment_max_error_m: float


def build_anchor_catalog(
    *,
    pose3d_manifest_path: str | Path,
    qwen_window_catalog_path: str | Path,
    output_dir: str | Path,
    model_name: str | None = DEFAULT_KIMODO_GENERATION_MODEL,
    clip_ids: Sequence[str] | None = None,
    constraint_keyframes: int = 8,
    runtime: Any = None,
) -> dict[str, Any]:
    """Build a Kimodo-ready anchor catalog from exact window-level Qwen entries."""

    if int(constraint_keyframes) <= 0:
        raise ValueError("constraint_keyframes must be >= 1.")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    pose_entries = select_pose_entries(load_pose_manifest_entries(pose3d_manifest_path), clip_ids=clip_ids)
    pose_by_clip = {entry.clip_id: entry for entry in pose_entries}
    qwen_entries = select_catalog_entries(
        load_catalog_entries(qwen_window_catalog_path),
        clip_ids=clip_ids,
    )

    if runtime is None:
        runtime = _load_kimodo_runtime()
    device = runtime.resolve_device()
    resolved_model_name = runtime.default_model if model_name is None else model_name
    model, resolved_model = runtime.load_model(
        resolved_model_name,
        device=device,
        default_family="Kimodo",
        return_resolved_name=True,
    )
    model_info = runtime.get_model_info(resolved_model)
    model_display_name = resolved_model if model_info is None else model_info.display_name
    target_skeleton = None if model_info is None else model_info.skeleton
    if not _supports_smplx_rich_constraints(
        target_model=model_display_name,
        target_skeleton=target_skeleton,
        model_skeleton=getattr(model, "skeleton", None),
    ):
        raise ValueError(
            "Rich window constraints are only supported for SMPLX targets; "
            f"got model={model_display_name!r}, skeleton={target_skeleton!r}."
        )
    target_fps = float(model.fps)

    cache: dict[str, _CachedPoseClip] = {}
    catalog_entries: list[dict[str, Any]] = []
    trace_paths: list[str] = []

    for qwen_entry in qwen_entries:
        prompt_id = qwen_entry.prompt_id
        reference_clip_id = qwen_entry.reference_clip_id
        if reference_clip_id is None:
            raise ValueError(f"Window catalog entry {prompt_id!r} is missing reference_clip_id.")
        if reference_clip_id not in pose_by_clip:
            raise ValueError(
                f"Window catalog entry {prompt_id!r} references unknown clip_id {reference_clip_id!r}."
            )
        if not qwen_entry.prompt_text.startswith("A person"):
            raise ValueError(f"Window catalog entry {prompt_id!r} must start with 'A person'.")
        if qwen_entry.duration_hint_sec is None or float(qwen_entry.duration_hint_sec) <= 0.0:
            raise ValueError(f"Window catalog entry {prompt_id!r} requires duration_hint_sec > 0.")
        if not qwen_entry.window:
            raise ValueError(f"Window catalog entry {prompt_id!r} is missing window metadata.")

        window = WindowSpec.from_dict(qwen_entry.window, prompt_id=prompt_id)
        pose_entry = pose_by_clip[reference_clip_id]
        pose_path = Path(pose_entry.pose3d_npz_path)

        if reference_clip_id not in cache:
            sequence = load_pose_sequence3d(pose_path)
            source_times = resolve_source_times(sequence)
            smplx_source_indices = _resolve_smplx_source_indices(sequence.joint_names_3d)
            root_translation_m = _require_root_translation(sequence, pose_path)
            root_alignment_max_error_m = _validate_pose_sequence_for_constraints(
                sequence=sequence,
                source_times=source_times,
                root_translation_m=root_translation_m,
                smplx_source_indices=smplx_source_indices,
                pose_path=pose_path,
            )
            ik_path = _resolve_or_materialize_ik_sequence_path(pose_path=pose_path, sequence=sequence)
            ik_sequence = _load_ik_sequence(ik_path)
            ik_smplx_source_indices = _resolve_smplx_source_indices(ik_sequence.joint_names_3d)
            ik_root_alignment_max_error_m = _validate_ik_sequence_alignment(
                sequence=sequence,
                source_times=source_times,
                root_translation_m=root_translation_m,
                ik_sequence=ik_sequence,
                pose_path=pose_path,
                ik_path=ik_path,
            )
            cache[reference_clip_id] = _CachedPoseClip(
                sequence=sequence,
                source_times=source_times,
                smplx_source_indices=smplx_source_indices,
                root_translation_m=root_translation_m,
                root_alignment_max_error_m=root_alignment_max_error_m,
                ik_sequence=ik_sequence,
                ik_path=ik_path,
                ik_smplx_source_indices=ik_smplx_source_indices,
                ik_root_alignment_max_error_m=ik_root_alignment_max_error_m,
            )

        cached = cache[reference_clip_id]
        window_dir = output_root / prompt_id
        window_dir.mkdir(parents=True, exist_ok=True)

        constraints_payload, traceability = _build_window_constraints_payload(
            cached=cached,
            pose_path=pose_path,
            window=window,
            target_fps=target_fps,
            target_model=model_display_name,
            target_skeleton=target_skeleton,
            constraint_keyframes=int(constraint_keyframes),
        )

        constraints_path = window_dir / "constraints.json"
        traceability_path = window_dir / "traceability.json"
        write_json(constraints_path, constraints_payload)
        write_json(traceability_path, traceability)
        trace_paths.append(str(traceability_path.resolve()))

        entry_payload = dict(qwen_entry.raw_entry)
        entry_payload["constraints_path"] = str(constraints_path.resolve())
        entry_payload["constraint_summary"] = {
            "constraint_mode": CONSTRAINT_MODE,
            "target_model": model_display_name,
            "target_skeleton": target_skeleton,
            "constraint_types": list(traceability["constraint_types"]),
            "constraint_frame_counts": dict(traceability["constraint_frame_counts"]),
            "source_frame_range": [
                int(traceability["source_frame_start"]),
                int(traceability["source_frame_end"]),
            ],
            "source_keyframes": list(traceability["source_keyframes"]),
            "target_keyframes": list(traceability["target_keyframes"]),
            "root2d_enabled": bool(traceability["root2d_enabled"]),
            "heading_enabled": bool(traceability["heading_enabled"]),
            "pose3d_source_path": traceability["source_pose3d_npz_path"],
            "root2d_min_displacement_m": float(traceability["root2d_min_displacement_m"]),
            "heading_min_displacement_m": float(traceability["heading_min_displacement_m"]),
        }
        entry_payload["window_id"] = prompt_id
        entry_payload["prompt_id"] = prompt_id
        entry_payload["reference_clip_id"] = reference_clip_id
        entry_payload["duration_hint_sec"] = float(window.duration_sec)
        catalog_entries.append(entry_payload)

    catalog_path = output_root / "kimodo_anchor_catalog.jsonl"
    summary_path = output_root / "kimodo_anchor_catalog.summary.json"
    write_jsonl(catalog_path, catalog_entries)

    summary = {
        "constraint_mode": CONSTRAINT_MODE,
        "pose3d_manifest_path": str(Path(pose3d_manifest_path).resolve()),
        "qwen_window_catalog_path": str(Path(qwen_window_catalog_path).resolve()),
        "output_dir": str(output_root.resolve()),
        "catalog_path": str(catalog_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "model_name": model_display_name,
        "resolved_model": resolved_model,
        "target_fps": target_fps,
        "constraint_keyframes": int(constraint_keyframes),
        "root2d_min_displacement_m": float(ROOT2D_MIN_DISPLACEMENT_M),
        "heading_min_displacement_m": float(HEADING_MIN_DISPLACEMENT_M),
        "ground_height_quantile": float(GROUND_HEIGHT_QUANTILE),
        "num_input_clips": int(len(pose_entries)),
        "num_input_windows": int(len(qwen_entries)),
        "num_ok": int(len(catalog_entries)),
        "num_fail": 0,
        "num_total_windows": int(len(catalog_entries)),
        "sample_prompt_ids": [entry["prompt_id"] for entry in catalog_entries[:5]],
        "sample_traceability_paths": trace_paths[:5],
    }
    write_json(summary_path, summary)
    return summary


def _supports_smplx_rich_constraints(
    *,
    target_model: str,
    target_skeleton: str | None,
    model_skeleton: Any,
) -> bool:
    if str(target_skeleton or "").upper() == "SMPLX":
        return True
    if "smplx" in str(target_model).lower():
        return True
    return "smplx" in str(getattr(model_skeleton, "name", "")).lower()


def _resolve_smplx_source_indices(joint_names_3d: Sequence[str]) -> np.ndarray:
    joint_names = [str(name) for name in joint_names_3d]
    joint_to_index: dict[str, int] = {}
    duplicates: set[str] = set()
    for index, joint_name in enumerate(joint_names):
        if joint_name in joint_to_index:
            duplicates.add(joint_name)
        joint_to_index[joint_name] = index
    if duplicates:
        raise ValueError(f"Joint source exposes duplicated joint names: {sorted(duplicates)}")

    source_indices: list[int] = []
    missing_source: list[str] = []
    resolved_target_names: list[str] = []
    for source_name, target_name in IMUGPT_TO_SMPLX_JOINT_MAP:
        if source_name not in joint_to_index:
            missing_source.append(source_name)
            continue
        source_indices.append(int(joint_to_index[source_name]))
        resolved_target_names.append(target_name)

    if missing_source:
        raise ValueError(
            "Joint source is missing IMUGPT22 joints required for SMPLX export: "
            f"{sorted(missing_source)}"
        )
    if tuple(resolved_target_names) != SMPLX22_JOINT_NAMES:
        raise ValueError(
            "The fixed IMUGPT22 -> SMPLX22 mapping no longer matches the expected SMPLX joint order."
        )
    return np.asarray(source_indices, dtype=np.int32)


def _require_root_translation(sequence: PoseSequence3D, pose_path: Path) -> np.ndarray:
    root_translation_m = sequence.resolved_root_translation_m()
    if root_translation_m is None:
        raise ValueError(
            f"Pose sequence {pose_path.resolve()} must expose root_translation_m for anchored constraints."
        )
    if not np.isfinite(root_translation_m).all():
        raise ValueError(f"Pose sequence {pose_path.resolve()} root_translation_m contains NaN/Inf values.")
    return root_translation_m.astype(np.float32, copy=False)


def _resolve_or_materialize_ik_sequence_path(*, pose_path: Path, sequence: PoseSequence3D) -> Path:
    ik_path = pose_path.with_name(IK_SEQUENCE_FILENAME)
    if ik_path.exists():
        return ik_path

    result = run_ik(
        sequence,
        output_dir=pose_path.parent,
        write_bvh=False,
        ik_sequence_filename=IK_SEQUENCE_FILENAME,
    )
    artifacts = result.get("artifacts", {})
    resolved_path = artifacts.get("ik_sequence_npz_path")
    if resolved_path:
        ik_path = Path(str(resolved_path))
    if not ik_path.exists():
        raise FileNotFoundError(
            f"Unable to resolve IK sequence for {pose_path.resolve()}; expected {ik_path.resolve()}."
        )
    return ik_path


def _load_ik_sequence(ik_path: Path) -> IKSequence:
    with np.load(ik_path, allow_pickle=False) as payload:
        return IKSequence.from_npz_payload({key: payload[key] for key in payload.files})


def _validate_pose_sequence_for_constraints(
    *,
    sequence: PoseSequence3D,
    source_times: np.ndarray,
    root_translation_m: np.ndarray,
    smplx_source_indices: np.ndarray,
    pose_path: Path,
) -> float:
    if sequence.num_frames <= 0:
        raise ValueError(f"Pose sequence {pose_path.resolve()} must contain at least one frame.")

    joint_positions = np.asarray(sequence.joint_positions_xyz, dtype=np.float32)
    if joint_positions.shape != (sequence.num_frames, len(sequence.joint_names_3d), 3):
        raise ValueError(
            f"Pose sequence {pose_path.resolve()} joint_positions_xyz has unexpected shape {joint_positions.shape}."
        )
    if not np.isfinite(joint_positions).all():
        raise ValueError(f"Pose sequence {pose_path.resolve()} joint_positions_xyz contains NaN/Inf values.")

    if source_times.shape != (sequence.num_frames,):
        raise ValueError(
            f"Pose sequence {pose_path.resolve()} timestamps must have shape [{sequence.num_frames}]."
        )
    if not np.isfinite(source_times).all():
        raise ValueError(f"Pose sequence {pose_path.resolve()} timestamps contain NaN/Inf values.")

    if root_translation_m.shape != (sequence.num_frames, 3):
        raise ValueError(
            f"Pose sequence {pose_path.resolve()} root_translation_m must have shape "
            f"[{sequence.num_frames}, 3]."
        )

    pelvis_positions = joint_positions[:, int(smplx_source_indices[0]), :]
    alignment_error = np.linalg.norm(pelvis_positions - root_translation_m, axis=1)
    max_alignment_error_m = float(np.max(alignment_error))
    if max_alignment_error_m > ROOT_ALIGNMENT_TOLERANCE_M:
        raise ValueError(
            f"Pose sequence {pose_path.resolve()} root_translation_m does not align with pelvis positions; "
            f"max error is {max_alignment_error_m:.6f} m."
        )
    return max_alignment_error_m


def _validate_ik_sequence_alignment(
    *,
    sequence: PoseSequence3D,
    source_times: np.ndarray,
    root_translation_m: np.ndarray,
    ik_sequence: IKSequence,
    pose_path: Path,
    ik_path: Path,
) -> float:
    if ik_sequence.num_frames != sequence.num_frames:
        raise ValueError(
            f"IK sequence {ik_path.resolve()} frame count {ik_sequence.num_frames} does not match "
            f"pose sequence {pose_path.resolve()} frame count {sequence.num_frames}."
        )

    if np.asarray(ik_sequence.frame_indices, dtype=np.int32).shape != (sequence.num_frames,):
        raise ValueError(
            f"IK sequence {ik_path.resolve()} frame_indices must have shape [{sequence.num_frames}]."
        )
    if not np.array_equal(
        np.asarray(ik_sequence.frame_indices, dtype=np.int32),
        np.asarray(sequence.frame_indices, dtype=np.int32),
    ):
        raise ValueError(
            f"IK sequence {ik_path.resolve()} frame_indices do not align with {pose_path.resolve()}."
        )

    timestamps_sec = np.asarray(ik_sequence.timestamps_sec, dtype=np.float32)
    if timestamps_sec.shape != (sequence.num_frames,):
        raise ValueError(
            f"IK sequence {ik_path.resolve()} timestamps must have shape [{sequence.num_frames}]."
        )
    if not np.isfinite(timestamps_sec).all():
        raise ValueError(f"IK sequence {ik_path.resolve()} timestamps contain NaN/Inf values.")
    if not np.allclose(
        timestamps_sec,
        source_times.astype(np.float32, copy=False),
        atol=TIMESTAMP_ALIGNMENT_TOLERANCE_SEC,
        rtol=0.0,
    ):
        raise ValueError(
            f"IK sequence {ik_path.resolve()} timestamps do not align with {pose_path.resolve()}."
        )

    local_joint_rotations = np.asarray(ik_sequence.local_joint_rotations, dtype=np.float32)
    if local_joint_rotations.ndim != 3:
        raise ValueError(
            f"IK sequence {ik_path.resolve()} local_joint_rotations must have shape [T, J, 4]."
        )
    if local_joint_rotations.shape[0] != sequence.num_frames:
        raise ValueError(
            f"IK sequence {ik_path.resolve()} local_joint_rotations must have {sequence.num_frames} frames."
        )
    if local_joint_rotations.shape[2] != 4:
        raise ValueError(
            f"IK sequence {ik_path.resolve()} local_joint_rotations must use quaternion_wxyz with shape [T, J, 4]."
        )
    if not np.isfinite(local_joint_rotations).all():
        raise ValueError(f"IK sequence {ik_path.resolve()} local_joint_rotations contain NaN/Inf values.")
    if str(ik_sequence.rotation_representation) != "quaternion_wxyz":
        raise ValueError(
            f"IK sequence {ik_path.resolve()} must use rotation_representation='quaternion_wxyz'."
        )

    ik_root_translation = np.asarray(ik_sequence.root_translation_m, dtype=np.float32)
    if ik_root_translation.shape != (sequence.num_frames, 3):
        raise ValueError(
            f"IK sequence {ik_path.resolve()} root_translation_m must have shape [{sequence.num_frames}, 3]."
        )
    if not np.isfinite(ik_root_translation).all():
        raise ValueError(f"IK sequence {ik_path.resolve()} root_translation_m contains NaN/Inf values.")

    root_alignment_error = np.linalg.norm(ik_root_translation - root_translation_m, axis=1)
    max_alignment_error_m = float(np.max(root_alignment_error))
    if max_alignment_error_m > IK_ALIGNMENT_TOLERANCE_M:
        raise ValueError(
            f"IK sequence {ik_path.resolve()} root translation does not align with {pose_path.resolve()}; "
            f"max error is {max_alignment_error_m:.6f} m."
        )
    return max_alignment_error_m


def _build_window_constraints_payload(
    *,
    cached: _CachedPoseClip,
    pose_path: Path,
    window: WindowSpec,
    target_fps: float,
    target_model: str,
    target_skeleton: str | None,
    constraint_keyframes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_window_bounds(window, num_frames=cached.sequence.num_frames)

    num_target_frames = max(1, int(round(float(window.duration_sec) * float(target_fps))))
    dense_target_frame_indices = np.arange(num_target_frames, dtype=np.int32)
    sparse_target_frame_indices = _select_uniform_target_keyframes(num_target_frames, constraint_keyframes)

    source_slice = slice(window.source_start_index, window.source_end_index)
    source_positions_window = np.asarray(
        cached.sequence.joint_positions_xyz[source_slice][:, cached.smplx_source_indices, :],
        dtype=np.float32,
    )
    source_root_window = np.asarray(
        cached.root_translation_m[source_slice],
        dtype=np.float32,
    )
    if source_root_window.shape[0] <= 0:
        raise ValueError(f"Window {window.prompt_id!r} does not contain source frames.")
    if source_positions_window.shape[0] != source_root_window.shape[0]:
        raise ValueError(f"Window {window.prompt_id!r} source pose and root windows are misaligned.")
    if not np.isfinite(source_positions_window).all():
        raise ValueError(f"Window {window.prompt_id!r} source pose contains NaN/Inf values.")
    if not np.isfinite(source_root_window).all():
        raise ValueError(f"Window {window.prompt_id!r} root translation contains NaN/Inf values.")

    source_times_window = (
        np.asarray(cached.source_times[source_slice], dtype=np.float32) - np.float32(window.start_sec)
    )
    source_root_2d = source_root_window[:, [0, 2]]
    dense_root_2d = _interpolate_dense_root2d(
        target_frames=dense_target_frame_indices,
        target_fps=target_fps,
        source_times_window=source_times_window,
        source_root_2d=source_root_2d,
    )

    sparse_source_indices = _map_target_frames_to_source_indices(
        target_frames=sparse_target_frame_indices,
        target_fps=target_fps,
        source_times=cached.source_times,
        window=window,
    )
    dense_source_indices = _map_target_frames_to_source_indices(
        target_frames=dense_target_frame_indices,
        target_fps=target_fps,
        source_times=cached.source_times,
        window=window,
    )

    sparse_source_indices = np.asarray(sparse_source_indices, dtype=np.int32)
    sparse_source_root = np.asarray(
        cached.ik_sequence.root_translation_m[sparse_source_indices],
        dtype=np.float32,
    )
    ground_height_m = _estimate_ground_height(source_positions_window)
    sparse_root_positions_grounded = sparse_source_root.copy()
    sparse_root_positions_grounded[:, 1] -= np.float32(ground_height_m)

    fullbody_local_rot = _wxyz_quaternions_to_axis_angle(
        np.asarray(
            cached.ik_sequence.local_joint_rotations[sparse_source_indices][:, cached.ik_smplx_source_indices, :],
            dtype=np.float32,
        )
    )

    constraints_payload: list[dict[str, Any]] = [
        {
            "type": "fullbody",
            "frame_indices": [int(value) for value in sparse_target_frame_indices.tolist()],
            "local_joints_rot": fullbody_local_rot.astype(np.float32, copy=False).tolist(),
            "root_positions": sparse_root_positions_grounded.astype(np.float32, copy=False).tolist(),
            "smooth_root_2d": sparse_root_positions_grounded[:, [0, 2]].astype(np.float32, copy=False).tolist(),
        }
    ]

    root2d_displacement_m = float(np.linalg.norm(source_root_2d[-1] - source_root_2d[0]))
    root2d_enabled = root2d_displacement_m >= ROOT2D_MIN_DISPLACEMENT_M
    heading_enabled = False

    constraint_types = ["fullbody"]
    constraint_frame_counts = {"fullbody": int(len(sparse_target_frame_indices))}
    if root2d_enabled:
        root2d_payload: dict[str, Any] = {
            "type": "root2d",
            "frame_indices": [int(value) for value in dense_target_frame_indices.tolist()],
            "smooth_root_2d": dense_root_2d.astype(np.float32, copy=False).tolist(),
        }
        if root2d_displacement_m >= HEADING_MIN_DISPLACEMENT_M:
            global_root_heading = _compute_root_heading(dense_root_2d)
            if global_root_heading is not None:
                root2d_payload["global_root_heading"] = global_root_heading.astype(np.float32, copy=False).tolist()
                heading_enabled = True
        constraints_payload.append(root2d_payload)
        constraint_types.append("root2d")
        constraint_frame_counts["root2d"] = int(len(dense_target_frame_indices))

    source_frame_ids = np.asarray(cached.sequence.frame_indices, dtype=np.int32)
    source_start_frame = int(source_frame_ids[window.source_start_index])
    source_end_frame = int(source_frame_ids[max(window.source_end_index - 1, window.source_start_index)])
    traceability = {
        "constraint_mode": CONSTRAINT_MODE,
        "prompt_id": window.prompt_id,
        "reference_clip_id": cached.sequence.clip_id,
        "source_pose3d_npz_path": str(pose_path.resolve()),
        "ik_sequence_path": str(cached.ik_path.resolve()),
        "source_coordinate_space": str(cached.sequence.coordinate_space),
        "root_source": "pose3d.root_translation_m",
        "ground_height_source": "pose3d.support_joint_heights_p05",
        "fullbody_root_source": "ik_sequence.root_translation_m_grounded",
        "fullbody_rotation_source": "ik_sequence.local_joint_rotations_quaternion_wxyz_to_axis_angle",
        "root_alignment_max_error_m": float(cached.root_alignment_max_error_m),
        "ik_root_alignment_max_error_m": float(cached.ik_root_alignment_max_error_m),
        "source_fps": resolve_sequence_fps(cached.sequence, cached.source_times),
        "target_model": target_model,
        "target_skeleton": target_skeleton,
        "target_fps": float(target_fps),
        "source_frame_start": source_start_frame,
        "source_frame_end": source_end_frame,
        "source_keyframes": [int(value) for value in source_frame_ids[sparse_source_indices].tolist()],
        "target_keyframes": [int(value) for value in sparse_target_frame_indices.tolist()],
        "constraint_types": list(constraint_types),
        "constraint_frame_counts": dict(constraint_frame_counts),
        "duration_hint_sec": float(window.duration_sec),
        "num_target_frames": int(num_target_frames),
        "constraint_keyframes_requested": int(constraint_keyframes),
        "constraint_keyframes_emitted": int(len(sparse_target_frame_indices)),
        "root2d_enabled": bool(root2d_enabled),
        "heading_enabled": bool(heading_enabled),
        "root2d_min_displacement_m": float(ROOT2D_MIN_DISPLACEMENT_M),
        "heading_min_displacement_m": float(HEADING_MIN_DISPLACEMENT_M),
        "root2d_net_displacement_m": float(root2d_displacement_m),
        "ground_height_m": float(ground_height_m),
        "ground_reference_joint_names": list(SUPPORT_JOINT_NAMES),
        "fullbody_joint_names": list(SMPLX22_JOINT_NAMES),
        "fullbody_source_indices": [int(value) for value in sparse_source_indices.tolist()],
        "root2d_source_indices": [int(value) for value in dense_source_indices.tolist()],
    }
    return constraints_payload, traceability


def _validate_window_bounds(window: WindowSpec, *, num_frames: int) -> None:
    if window.source_start_index < 0:
        raise ValueError(f"Window {window.prompt_id!r} has negative source_start_index.")
    if window.source_end_index <= window.source_start_index:
        raise ValueError(f"Window {window.prompt_id!r} must contain at least one source frame.")
    if window.source_end_index > num_frames:
        raise ValueError(
            f"Window {window.prompt_id!r} exceeds pose length: "
            f"{window.source_end_index} > {num_frames}."
        )


def _estimate_ground_height(source_positions_window: np.ndarray) -> float:
    support_indices = np.asarray(
        [int(SMPLX22_JOINT_NAMES.index(name)) for name in SUPPORT_JOINT_NAMES],
        dtype=np.int32,
    )
    support_heights = np.asarray(source_positions_window[:, support_indices, 1], dtype=np.float32).reshape(-1)
    finite_heights = support_heights[np.isfinite(support_heights)]
    if finite_heights.size <= 0:
        raise ValueError("Support-joint heights contain no finite values.")
    return float(np.quantile(finite_heights, GROUND_HEIGHT_QUANTILE))


def _wxyz_quaternions_to_axis_angle(quaternions_wxyz: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions_wxyz, dtype=np.float64)
    if quaternions.ndim != 3 or quaternions.shape[-1] != 4:
        raise ValueError("Expected quaternion local_joint_rotations with shape [T, J, 4].")
    norms = np.linalg.norm(quaternions, axis=2, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError("Quaternion local_joint_rotations contain zero-norm entries.")
    quaternions_xyzw = np.concatenate(
        [quaternions[..., 1:] / norms, quaternions[..., 0:1] / norms],
        axis=2,
    )
    axis_angle = Rotation.from_quat(quaternions_xyzw.reshape(-1, 4)).as_rotvec()
    axis_angle = axis_angle.reshape(quaternions.shape[:2] + (3,))
    if not np.isfinite(axis_angle).all():
        raise ValueError("Axis-angle conversion produced NaN/Inf values.")
    return axis_angle.astype(np.float32, copy=False)


def _select_uniform_target_keyframes(num_target_frames: int, requested_keyframes: int) -> np.ndarray:
    if num_target_frames <= 0:
        raise ValueError("num_target_frames must be >= 1.")
    if requested_keyframes <= 0:
        raise ValueError("requested_keyframes must be >= 1.")
    if num_target_frames == 1:
        return np.asarray([0], dtype=np.int32)

    count = min(int(requested_keyframes), int(num_target_frames))
    if count == num_target_frames:
        return np.arange(num_target_frames, dtype=np.int32)

    raw_positions = np.linspace(0, num_target_frames - 1, num=count, endpoint=True)
    indices = np.rint(raw_positions).astype(np.int32)
    indices[0] = 0
    indices[-1] = num_target_frames - 1
    indices = np.unique(indices)
    if indices.size == count:
        return indices.astype(np.int32, copy=False)

    all_frames = np.arange(num_target_frames, dtype=np.int32)
    chosen: list[int] = []
    used: set[int] = set()
    for target in raw_positions.tolist():
        candidate_order = sorted(all_frames.tolist(), key=lambda frame: (abs(frame - target), frame))
        for candidate in candidate_order:
            if candidate not in used:
                used.add(candidate)
                chosen.append(int(candidate))
                break
    chosen.sort()
    return np.asarray(chosen, dtype=np.int32)


def _interpolate_dense_root2d(
    *,
    target_frames: np.ndarray,
    target_fps: float,
    source_times_window: np.ndarray,
    source_root_2d: np.ndarray,
) -> np.ndarray:
    if source_root_2d.shape[0] != source_times_window.shape[0]:
        raise ValueError("source_root_2d and source_times_window must have the same length.")
    if source_root_2d.shape[0] <= 0:
        raise ValueError("Expected at least one source root frame for interpolation.")

    target_times = target_frames.astype(np.float32) / np.float32(target_fps)
    dense_root_2d = np.stack(
        [
            np.interp(target_times, source_times_window, source_root_2d[:, axis]).astype(np.float32, copy=False)
            for axis in range(2)
        ],
        axis=1,
    )
    if not np.isfinite(dense_root_2d).all():
        raise ValueError("Dense root2d interpolation produced NaN/Inf values.")
    return dense_root_2d.astype(np.float32, copy=False)


def _compute_root_heading(smooth_root_2d: np.ndarray) -> np.ndarray | None:
    if smooth_root_2d.shape[0] <= 1:
        return None

    diffs = np.zeros_like(smooth_root_2d, dtype=np.float32)
    diffs[0] = smooth_root_2d[1] - smooth_root_2d[0]
    diffs[-1] = smooth_root_2d[-1] - smooth_root_2d[-2]
    if smooth_root_2d.shape[0] > 2:
        diffs[1:-1] = 0.5 * (smooth_root_2d[2:] - smooth_root_2d[:-2])

    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    if float(np.max(norms)) <= 1e-6:
        return None

    headings = np.zeros_like(diffs, dtype=np.float32)
    first_valid = int(np.argmax((norms[:, 0] > 1e-6).astype(np.int32)))
    headings[first_valid] = diffs[first_valid] / norms[first_valid]
    for index in range(first_valid + 1, diffs.shape[0]):
        if float(norms[index, 0]) > 1e-6:
            headings[index] = diffs[index] / norms[index]
        else:
            headings[index] = headings[index - 1]
    for index in range(first_valid - 1, -1, -1):
        headings[index] = headings[index + 1]
    return headings


def _map_target_frames_to_source_indices(
    *,
    target_frames: np.ndarray,
    target_fps: float,
    source_times: np.ndarray,
    window: WindowSpec,
) -> np.ndarray:
    window_source_times = np.asarray(
        source_times[window.source_start_index : window.source_end_index],
        dtype=np.float32,
    )
    if window_source_times.size <= 0:
        raise ValueError(f"Window {window.prompt_id!r} does not contain source timestamps.")

    absolute_times = (target_frames.astype(np.float32) / np.float32(target_fps)) + np.float32(window.start_sec)
    mapped: list[int] = []
    for absolute_time in absolute_times.tolist():
        insert_at = int(np.searchsorted(window_source_times, np.float32(absolute_time), side="left"))
        if insert_at <= 0:
            nearest = 0
        elif insert_at >= int(window_source_times.size):
            nearest = int(window_source_times.size) - 1
        else:
            prev_index = insert_at - 1
            next_index = insert_at
            prev_delta = abs(float(window_source_times[prev_index]) - float(absolute_time))
            next_delta = abs(float(window_source_times[next_index]) - float(absolute_time))
            nearest = prev_index if prev_delta <= next_delta else next_index
        mapped.append(int(window.source_start_index + nearest))
    return np.asarray(mapped, dtype=np.int32)
