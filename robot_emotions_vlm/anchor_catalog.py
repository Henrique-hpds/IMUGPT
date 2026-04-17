"""Anchor-catalog builder from pose3d + window-level Qwen descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pose_module.interfaces import PoseSequence3D

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
CONSTRAINT_MODE = "pose3d"

SMPLX22_JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)
IMUGPT_TO_SMPLX_JOINT_MAP = (
    ("Pelvis", "pelvis"), ("Left_hip", "left_hip"), ("Right_hip", "right_hip"),
    ("Spine1", "spine1"), ("Left_knee", "left_knee"), ("Right_knee", "right_knee"),
    ("Spine2", "spine2"), ("Left_ankle", "left_ankle"), ("Right_ankle", "right_ankle"),
    ("Spine3", "spine3"), ("Left_foot", "left_foot"), ("Right_foot", "right_foot"),
    ("Neck", "neck"), ("Left_collar", "left_collar"), ("Right_collar", "right_collar"),
    ("Head", "head"), ("Left_shoulder", "left_shoulder"), ("Right_shoulder", "right_shoulder"),
    ("Left_elbow", "left_elbow"), ("Right_elbow", "right_elbow"),
    ("Left_wrist", "left_wrist"), ("Right_wrist", "right_wrist"),
)
CONSTRAINT_SOURCE_POSE3D_NPZ = "pose3d_npz"
SUPPORTED_CONSTRAINT_SOURCES = (CONSTRAINT_SOURCE_POSE3D_NPZ,)


def _rebase_root_positions_to_first_xz(
    root_positions_xyz: np.ndarray,
    root2d_xz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Translate root trajectories so the first frame starts at x=z=0."""

    root_positions = np.asarray(root_positions_xyz, dtype=np.float32).copy()
    root2d = np.asarray(root2d_xz, dtype=np.float32).copy()
    if root_positions.shape[0] <= 0:
        raise ValueError("Expected at least one root frame to rebase the root trajectory.")

    xz_offset = np.asarray(root_positions[0, [0, 2]], dtype=np.float32)
    root_positions[:, 0] -= xz_offset[0]
    root_positions[:, 2] -= xz_offset[1]
    root2d[:, 0] -= xz_offset[0]
    root2d[:, 1] -= xz_offset[1]
    return (
        root_positions.astype(np.float32, copy=False),
        root2d.astype(np.float32, copy=False),
        xz_offset.astype(np.float32, copy=False),
    )


@dataclass(frozen=True)
class _CachedPoseClip:
    sequence: PoseSequence3D
    source_times: np.ndarray
    root_translation_m: np.ndarray
    root_alignment_max_error_m: float


def build_anchor_catalog(
    *,
    pose3d_manifest_path: str | Path,
    qwen_window_catalog_path: str | Path,
    output_dir: str | Path,
    model_name: str | None = DEFAULT_KIMODO_GENERATION_MODEL,
    clip_ids: Sequence[str] | None = None,
    runtime: Any = None,
) -> dict[str, Any]:
    """Build a Kimodo-ready anchor catalog (root2d only) from window-level Qwen entries."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    pose_entries = select_pose_entries(load_pose_manifest_entries(pose3d_manifest_path), clip_ids=clip_ids)
    pose_by_clip = {entry.clip_id: entry for entry in pose_entries}
    qwen_entries = select_catalog_entries(
        load_catalog_entries(qwen_window_catalog_path),
        clip_ids=clip_ids,
    )

    # Resolve target FPS from the Kimodo model (needed to build the dense frame grid).
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

        if reference_clip_id not in cache:
            pose3d_npz_path = Path(pose_entry.pose3d_npz_path)
            sequence = load_pose_sequence3d(pose3d_npz_path)
            source_times = resolve_source_times(sequence)
            root_translation_m = _require_root_translation(sequence, pose3d_npz_path)
            cache[reference_clip_id] = _CachedPoseClip(
                sequence=sequence,
                source_times=source_times,
                root_translation_m=root_translation_m,
                root_alignment_max_error_m=0.0,
            )

        cached = cache[reference_clip_id]
        pose3d_npz_path = Path(pose_entry.pose3d_npz_path)
        window_dir = output_root / prompt_id
        window_dir.mkdir(parents=True, exist_ok=True)

        constraints_payload, traceability = _build_window_constraints_payload(
            cached=cached,
            pose3d_npz_path=pose3d_npz_path,
            window=window,
            target_fps=target_fps,
            target_model=model_display_name,
            target_skeleton=target_skeleton,
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
            "root2d_enabled": bool(traceability["root2d_enabled"]),
            "heading_enabled": bool(traceability["heading_enabled"]),
            "pose3d_source_path": traceability["pose3d_npz_path"],
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


def _require_root_translation(sequence: PoseSequence3D, pose_path: Path) -> np.ndarray:
    root_translation_m = sequence.resolved_root_translation_m()
    if root_translation_m is None:
        raise ValueError(
            f"Pose sequence {pose_path.resolve()} must expose root_translation_m for anchored constraints."
        )
    if not np.isfinite(root_translation_m).all():
        raise ValueError(f"Pose sequence {pose_path.resolve()} root_translation_m contains NaN/Inf values.")
    return root_translation_m.astype(np.float32, copy=False)


def _build_window_constraints_payload(
    *,
    cached: _CachedPoseClip,
    pose3d_npz_path: Path,
    window: WindowSpec,
    target_fps: float,
    target_model: str,
    target_skeleton: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_window_bounds(window, num_frames=cached.sequence.num_frames)

    num_target_frames = max(1, int(round(float(window.duration_sec) * float(target_fps))))
    dense_target_frame_indices = np.arange(num_target_frames, dtype=np.int32)

    source_slice = slice(window.source_start_index, window.source_end_index)
    source_root_window = np.asarray(
        cached.root_translation_m[source_slice],
        dtype=np.float32,
    )
    if source_root_window.shape[0] <= 0:
        raise ValueError(f"Window {window.prompt_id!r} does not contain source frames.")
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

    dense_source_indices = _map_target_frames_to_source_indices(
        target_frames=dense_target_frame_indices,
        target_fps=target_fps,
        source_times=cached.source_times,
        window=window,
    )

    root2d_displacement_m = float(np.linalg.norm(source_root_2d[-1] - source_root_2d[0]))
    root2d_motion_mode = "interpolated"
    if root2d_displacement_m < ROOT2D_MIN_DISPLACEMENT_M:
        # For near-static windows, constrain a smooth linear root path to avoid diffusion drift.
        if int(num_target_frames) <= 1:
            dense_root_2d = source_root_2d[[0]].astype(np.float32, copy=False)
        else:
            dense_root_2d = np.linspace(
                source_root_2d[0],
                source_root_2d[-1],
                num=int(num_target_frames),
                endpoint=True,
                dtype=np.float32,
            )
        root2d_motion_mode = "stabilized_linear"

    # Rebase so the window starts at XZ origin, then flip to Kimodo convention:
    # Pipeline: +X = anatomical right, +Z = forward.
    # Kimodo:   +X = anatomical left,  -Z = forward.  Negate both root2d columns.
    _, dense_root_2d_rebased, window_root_origin_xz = _rebase_root_positions_to_first_xz(
        source_root_window,
        dense_root_2d,
    )
    dense_root_2d_rebased = -dense_root_2d_rebased

    heading_enabled = False
    root2d_payload: dict[str, Any] = {
        "type": "root2d",
        "frame_indices": [int(value) for value in dense_target_frame_indices.tolist()],
        "smooth_root_2d": dense_root_2d_rebased.astype(np.float32, copy=False).tolist(),
    }
    if root2d_displacement_m >= HEADING_MIN_DISPLACEMENT_M:
        global_root_heading = _compute_root_heading(dense_root_2d_rebased)
        if global_root_heading is not None:
            root2d_payload["global_root_heading"] = global_root_heading.astype(np.float32, copy=False).tolist()
            heading_enabled = True
    constraints_payload: list[dict[str, Any]] = [root2d_payload]

    source_frame_ids = np.asarray(cached.sequence.frame_indices, dtype=np.int32)
    source_start_frame = int(source_frame_ids[window.source_start_index])
    source_end_frame = int(source_frame_ids[max(window.source_end_index - 1, window.source_start_index)])
    traceability = {
        "constraint_mode": CONSTRAINT_MODE,
        "prompt_id": window.prompt_id,
        "reference_clip_id": cached.sequence.clip_id,
        "pose3d_npz_path": str(pose3d_npz_path.resolve()),
        "source_coordinate_space": str(cached.sequence.coordinate_space),
        "root_source": "pose3d.root_translation_m",
        "ground_height_source": None,
        "root_alignment_max_error_m": float(cached.root_alignment_max_error_m),
        "source_fps": resolve_sequence_fps(cached.sequence, cached.source_times),
        "target_model": target_model,
        "target_skeleton": target_skeleton,
        "target_fps": float(target_fps),
        "source_frame_start": source_start_frame,
        "source_frame_end": source_end_frame,
        "constraint_types": ["root2d"],
        "constraint_frame_counts": {"root2d": int(len(dense_target_frame_indices))},
        "duration_hint_sec": float(window.duration_sec),
        "num_target_frames": int(num_target_frames),
        "root2d_enabled": True,
        "root2d_motion_mode": root2d_motion_mode,
        "heading_enabled": bool(heading_enabled),
        "root2d_min_displacement_m": float(ROOT2D_MIN_DISPLACEMENT_M),
        "heading_min_displacement_m": float(HEADING_MIN_DISPLACEMENT_M),
        "root2d_net_displacement_m": float(root2d_displacement_m),
        "window_root_origin_xz_m": window_root_origin_xz.astype(np.float32, copy=False).tolist(),
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
