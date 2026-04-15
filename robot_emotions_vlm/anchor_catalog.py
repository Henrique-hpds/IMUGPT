"""Anchor-catalog builder from pose3d + window-level Qwen descriptions."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pose_module.export.bvh import _build_children_list, _collect_depth_first_order
from pose_module.export.ik_adapter import (
    _estimate_local_rotation_matrices,
    _rotation_matrices_to_wxyz_quaternions,
    run_ik,
)
from pose_module.interfaces import IKSequence, PoseSequence3D

from .export import write_json, write_jsonl
from .kimodo_generation import (
    DEFAULT_KIMODO_GENERATION_MODEL,
    _import_kimodo_package,
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
    constraint_skeleton = _resolve_smplx_constraint_skeleton(getattr(model, "skeleton", None))

    cache: dict[str, tuple[PoseSequence3D, np.ndarray, np.ndarray, str, IKSequence, Path]] = {}
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
            ik_path = _resolve_or_materialize_ik_sequence_path(
                pose_path=pose_path,
                sequence=sequence,
            )
            ik_sequence = _load_ik_sequence(ik_path)
            _validate_ik_sequence_alignment(
                sequence=sequence,
                source_times=source_times,
                ik_sequence=ik_sequence,
                pose_path=pose_path,
                ik_path=ik_path,
            )
            root_translation_xyz, root_source = _resolve_root_translation(ik_sequence)
            cache[reference_clip_id] = (
                sequence,
                source_times,
                root_translation_xyz,
                root_source,
                ik_sequence,
                ik_path,
            )
        sequence, source_times, root_translation_xyz, root_source, ik_sequence, ik_path = cache[reference_clip_id]

        window_dir = output_root / prompt_id
        window_dir.mkdir(parents=True, exist_ok=True)

        constraints_payload, traceability = _build_window_constraints_payload(
            sequence=sequence,
            pose_path=pose_path,
            source_times=source_times,
            root_translation_xyz=root_translation_xyz,
            root_source=root_source,
            ik_sequence=ik_sequence,
            ik_path=ik_path,
            window=window,
            target_fps=target_fps,
            target_model=model_display_name,
            target_skeleton=target_skeleton,
            constraint_keyframes=int(constraint_keyframes),
            constraint_skeleton=constraint_skeleton,
        )

        constraints_path = window_dir / "constraints.json"
        traceability_path = window_dir / "traceability.json"
        write_json(constraints_path, constraints_payload)
        write_json(traceability_path, traceability)
        trace_paths.append(str(traceability_path.resolve()))

        entry_payload = dict(qwen_entry.raw_entry)
        entry_payload["constraints_path"] = str(constraints_path.resolve())
        entry_payload["constraint_summary"] = {
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
            "ik_sequence_path": traceability["ik_sequence_path"],
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


def _resolve_or_materialize_ik_sequence_path(
    *,
    pose_path: Path,
    sequence: PoseSequence3D,
) -> Path:
    ik_path = pose_path.with_name("ik_sequence.npz")
    if ik_path.exists():
        return ik_path

    run_ik(
        sequence,
        output_dir=pose_path.parent,
        write_bvh=False,
    )
    if not ik_path.exists():
        raise ValueError(
            f"Failed to materialize {ik_path.name} next to {pose_path.resolve()} from pose3d. "
            "The anchor catalog requires an IK export compatible with the SMPLX constraint builder."
        )
    return ik_path


def _load_ik_sequence(path: Path) -> IKSequence:
    with np.load(path, allow_pickle=False) as payload:
        return IKSequence.from_npz_payload(payload)


def _validate_ik_sequence_alignment(
    *,
    sequence: PoseSequence3D,
    source_times: np.ndarray,
    ik_sequence: IKSequence,
    pose_path: Path,
    ik_path: Path,
) -> None:
    if str(ik_sequence.rotation_representation) != "quaternion_wxyz":
        raise ValueError(
            f"Unsupported IK rotation_representation {ik_sequence.rotation_representation!r} in {ik_path.resolve()}."
        )

    if ik_sequence.num_frames != sequence.num_frames:
        raise ValueError(
            f"IK sequence {ik_path.resolve()} has {ik_sequence.num_frames} frames but "
            f"{pose_path.resolve()} has {sequence.num_frames}."
        )

    if np.asarray(ik_sequence.local_joint_rotations).shape != (ik_sequence.num_frames, len(IMUGPT_TO_SMPLX_JOINT_MAP), 4):
        raise ValueError(
            f"IK sequence {ik_path.resolve()} must expose local_joint_rotations with shape "
            f"[T, {len(IMUGPT_TO_SMPLX_JOINT_MAP)}, 4]."
        )

    root_translation = np.asarray(ik_sequence.root_translation_m, dtype=np.float32)
    if root_translation.shape != (ik_sequence.num_frames, 3):
        raise ValueError(f"IK root_translation_m in {ik_path.resolve()} must have shape [T, 3].")
    if not np.isfinite(root_translation).all():
        raise ValueError(f"IK root_translation_m in {ik_path.resolve()} contains NaN/Inf values.")

    source_frame_ids = np.asarray(sequence.frame_indices, dtype=np.int32)
    ik_frame_ids = np.asarray(ik_sequence.frame_indices, dtype=np.int32)
    if source_frame_ids.shape != ik_frame_ids.shape or not np.array_equal(source_frame_ids, ik_frame_ids):
        raise ValueError(
            f"IK frame_indices in {ik_path.resolve()} do not match pose frame_indices from {pose_path.resolve()}."
        )

    ik_timestamps = np.asarray(ik_sequence.timestamps_sec, dtype=np.float32)
    if ik_timestamps.shape != source_times.shape or not np.allclose(ik_timestamps, source_times, atol=1e-4):
        raise ValueError(
            f"IK timestamps in {ik_path.resolve()} do not align with pose timestamps from {pose_path.resolve()}."
        )

    _resolve_smplx_joint_indices(ik_sequence)


def _resolve_root_translation(ik_sequence: IKSequence) -> tuple[np.ndarray, str]:
    root_xyz = np.asarray(ik_sequence.root_translation_m, dtype=np.float32)
    if not np.isfinite(root_xyz).all():
        raise ValueError("ik_sequence.root_translation_m contains NaN/Inf values.")
    return root_xyz, "ik_sequence.root_translation_m"


def _build_window_constraints_payload(
    *,
    sequence: PoseSequence3D,
    pose_path: Path,
    source_times: np.ndarray,
    root_translation_xyz: np.ndarray,
    root_source: str,
    ik_sequence: IKSequence,
    ik_path: Path,
    window: WindowSpec,
    target_fps: float,
    target_model: str,
    target_skeleton: str | None,
    constraint_keyframes: int,
    constraint_skeleton: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    num_target_frames = max(1, int(round(float(window.duration_sec) * float(target_fps))))
    dense_target_frame_indices = np.arange(num_target_frames, dtype=np.int32)
    sparse_target_frame_indices = _select_uniform_target_keyframes(num_target_frames, constraint_keyframes)

    source_positions_window = np.asarray(
        sequence.joint_positions_xyz[window.source_start_index : window.source_end_index],
        dtype=np.float32,
    )
    source_root_window = np.asarray(
        root_translation_xyz[window.source_start_index : window.source_end_index],
        dtype=np.float32,
    )
    if source_root_window.shape[0] <= 0:
        raise ValueError(f"Window {window.prompt_id!r} does not contain source frames.")
    if source_positions_window.shape[0] != source_root_window.shape[0]:
        raise ValueError(f"Window {window.prompt_id!r} source pose and root windows are misaligned.")
    if not np.isfinite(source_positions_window).all():
        raise ValueError(f"Window {window.prompt_id!r} source pose contains NaN/Inf values.")

    source_times_window = (
        np.asarray(source_times[window.source_start_index : window.source_end_index], dtype=np.float32)
        - np.float32(window.start_sec)
    )
    source_root_2d = source_root_window[:, [0, 2]]
    if not np.isfinite(source_root_2d).all():
        raise ValueError(f"Window {window.prompt_id!r} root2d contains NaN/Inf values.")

    dense_root_2d = np.stack(
        [
            np.interp(
                dense_target_frame_indices.astype(np.float32) / np.float32(target_fps),
                source_times_window,
                source_root_2d[:, axis],
            ).astype(np.float32, copy=False)
            for axis in range(2)
        ],
        axis=1,
    )

    sparse_source_indices = _map_target_frames_to_source_indices(
        target_frames=sparse_target_frame_indices,
        target_fps=target_fps,
        source_times=source_times,
        window=window,
    )
    dense_source_indices = _map_target_frames_to_source_indices(
        target_frames=dense_target_frame_indices,
        target_fps=target_fps,
        source_times=source_times,
        window=window,
    )

    smplx_joint_indices = _resolve_smplx_joint_indices(ik_sequence)
    sparse_source_positions = np.asarray(
        sequence.joint_positions_xyz[np.asarray(sparse_source_indices, dtype=np.int32)][:, smplx_joint_indices, :],
        dtype=np.float32,
    )
    ground_offset_m = np.float32(np.min(source_positions_window[..., 1]))
    sparse_source_positions_grounded = sparse_source_positions.copy()
    sparse_source_positions_grounded[..., 1] -= ground_offset_m
    root_positions_grounded = sparse_source_positions_grounded[:, 0, :].astype(np.float32, copy=False)
    fullbody_local_rot = _estimate_smplx22_fullbody_axis_angle(
        sparse_source_positions_grounded,
        constraint_skeleton=constraint_skeleton,
    )

    constraints_payload: list[dict[str, Any]] = [
        {
            "type": "end-effector",
            "frame_indices": [int(value) for value in sparse_target_frame_indices.tolist()],
            "local_joints_rot": fullbody_local_rot.astype(np.float32, copy=False).tolist(),
            "root_positions": root_positions_grounded.astype(np.float32, copy=False).tolist(),
            "smooth_root_2d": root_positions_grounded[:, [0, 2]].astype(np.float32, copy=False).tolist(),
            "joint_names": ["LeftFoot", "RightFoot", "LeftHand", "RightHand", "Hips"],
        }
    ]

    root2d_displacement_m = float(np.linalg.norm(source_root_2d[-1] - source_root_2d[0]))
    root2d_enabled = root2d_displacement_m >= ROOT2D_MIN_DISPLACEMENT_M
    heading_enabled = False

    constraint_types = ["end-effector"]
    constraint_frame_counts = {"end-effector": int(len(sparse_target_frame_indices))}
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

    source_frame_ids = np.asarray(sequence.frame_indices, dtype=np.int32)
    source_start_frame = int(source_frame_ids[window.source_start_index])
    source_end_frame = int(source_frame_ids[max(window.source_end_index - 1, window.source_start_index)])
    traceability = {
        "prompt_id": window.prompt_id,
        "reference_clip_id": sequence.clip_id,
        "source_pose3d_npz_path": str(pose_path.resolve()),
        "ik_sequence_path": str(ik_path.resolve()),
        "root_source": root_source,
        "fullbody_root_source": "pose3d_joint_positions_xyz_grounded[pelvis]",
        "fullbody_rotation_source": "pose3d_joint_positions_xyz_grounded_retargeted_to_smplx22",
        "source_fps": resolve_sequence_fps(sequence, source_times),
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
        "fullbody_joint_names": list(SMPLX22_JOINT_NAMES),
        "fullbody_source_indices": [int(value) for value in sparse_source_indices.tolist()],
        "root2d_source_indices": [int(value) for value in dense_source_indices.tolist()],
    }
    return constraints_payload, traceability


def _resolve_smplx_joint_indices(ik_sequence: IKSequence) -> np.ndarray:
    joint_names = [str(name) for name in ik_sequence.joint_names_3d]
    joint_to_index: dict[str, int] = {}
    duplicates: set[str] = set()
    for index, joint_name in enumerate(joint_names):
        if joint_name in joint_to_index:
            duplicates.add(joint_name)
        joint_to_index[joint_name] = index
    if duplicates:
        raise ValueError(f"IK sequence exposes duplicated joint names: {sorted(duplicates)}")

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
            "IK sequence is missing IMUGPT22 joints required for SMPLX export: "
            f"{sorted(missing_source)}"
        )
    if tuple(resolved_target_names) != SMPLX22_JOINT_NAMES:
        raise ValueError(
            "The fixed IMUGPT22 -> SMPLX22 mapping no longer matches the expected SMPLX joint order."
        )
    return np.asarray(source_indices, dtype=np.int32)


def _resolve_smplx_constraint_skeleton(model_skeleton: Any) -> Any:
    skeleton = model_skeleton
    if (
        skeleton is None
        or getattr(skeleton, "nbjoints", None) != len(SMPLX22_JOINT_NAMES)
        or getattr(skeleton, "neutral_joints", None) is None
    ):
        _import_kimodo_package()
        skeleton = importlib.import_module("kimodo.skeleton").build_skeleton(len(SMPLX22_JOINT_NAMES))

    joint_names = tuple(str(name) for name in getattr(skeleton, "bone_order_names", []))
    if joint_names != SMPLX22_JOINT_NAMES:
        raise ValueError(
            "The loaded SMPLX skeleton does not match the expected 22-joint body ordering required "
            "by the anchor catalog."
        )

    neutral_joints = _tensor_to_numpy(getattr(skeleton, "neutral_joints", None))
    if neutral_joints is None or neutral_joints.shape != (len(SMPLX22_JOINT_NAMES), 3):
        raise ValueError("The loaded SMPLX skeleton does not expose neutral_joints with shape [22, 3].")

    return skeleton


def _estimate_smplx22_fullbody_axis_angle(
    source_positions_grounded: np.ndarray,
    *,
    constraint_skeleton: Any,
) -> np.ndarray:
    posed_joints = np.asarray(source_positions_grounded, dtype=np.float32)
    if posed_joints.ndim != 3 or posed_joints.shape[1:] != (len(SMPLX22_JOINT_NAMES), 3):
        raise ValueError("Expected grounded SMPLX22 joint positions with shape [T, 22, 3].")
    if posed_joints.shape[0] <= 0:
        raise ValueError("Expected at least one grounded SMPLX22 frame.")
    if not np.isfinite(posed_joints).all():
        raise ValueError("Grounded SMPLX22 joint positions contain NaN/Inf values.")

    parents = _tensor_to_numpy(getattr(constraint_skeleton, "joint_parents", None))
    if parents is None or parents.shape != (len(SMPLX22_JOINT_NAMES),):
        raise ValueError("The loaded SMPLX skeleton does not expose joint_parents with shape [22].")
    parents_int = [int(parent) for parent in parents.tolist()]
    root_index = int(getattr(constraint_skeleton, "root_idx", 0))
    children = _build_children_list(parents_int)
    traversal_order = _collect_depth_first_order(children, root_index=root_index)
    offsets = _extract_skeleton_offsets_m(constraint_skeleton, parents_int)

    local_rotation_matrices, _ = _estimate_local_rotation_matrices(
        posed_joints,
        offsets=offsets,
        parents=parents_int,
        children=children,
        traversal_order=traversal_order,
        root_index=root_index,
    )
    return _quaternion_wxyz_to_axis_angle(_rotation_matrices_to_wxyz_quaternions(local_rotation_matrices))


def _extract_skeleton_offsets_m(constraint_skeleton: Any, parents: Sequence[int]) -> np.ndarray:
    neutral_joints = _tensor_to_numpy(getattr(constraint_skeleton, "neutral_joints", None))
    if neutral_joints is None or neutral_joints.shape != (len(SMPLX22_JOINT_NAMES), 3):
        raise ValueError("The loaded SMPLX skeleton does not expose neutral_joints with shape [22, 3].")

    offsets = np.zeros_like(neutral_joints, dtype=np.float32)
    for joint_index, parent_index in enumerate(parents):
        if int(parent_index) < 0:
            continue
        offsets[joint_index] = (
            neutral_joints[joint_index] - neutral_joints[int(parent_index)]
        ).astype(np.float32, copy=False)
    return offsets.astype(np.float32, copy=False)


def _tensor_to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float32)


def _quaternion_wxyz_to_axis_angle(quaternions: np.ndarray) -> np.ndarray:
    rotations = np.asarray(quaternions, dtype=np.float32)
    if rotations.ndim != 3 or rotations.shape[-1] != 4:
        raise ValueError("Expected quaternion rotations with shape [T, J, 4].")

    norms = np.linalg.norm(rotations, axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-8):
        raise ValueError("Quaternion rotations contain invalid norms.")

    unit_quats = rotations / norms
    negative_w = unit_quats[..., :1] < 0.0
    unit_quats = np.where(negative_w, -unit_quats, unit_quats)

    xyz = unit_quats[..., 1:]
    w = np.clip(unit_quats[..., 0], -1.0, 1.0)
    sin_half = np.linalg.norm(xyz, axis=-1)
    angles = 2.0 * np.arctan2(sin_half, w)

    axis_angle = np.zeros_like(xyz, dtype=np.float32)
    non_zero = sin_half > 1e-8
    axis_angle[non_zero] = (
        xyz[non_zero] / sin_half[non_zero][..., None]
    ) * angles[non_zero][..., None]
    axis_angle[~non_zero] = 2.0 * xyz[~non_zero]
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
