"""Anchor-catalog builder from pose3d + window-level Qwen descriptions."""

from __future__ import annotations

import json
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


def build_anchor_catalog(
    *,
    pose3d_manifest_path: str | Path,
    qwen_window_catalog_path: str | Path,
    output_dir: str | Path,
    model_name: str | None = DEFAULT_KIMODO_GENERATION_MODEL,
    clip_ids: Sequence[str] | None = None,
    runtime: Any = None,
) -> dict[str, Any]:
    """Build a Kimodo-ready anchor catalog from exact window-level Qwen entries."""

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
    target_fps = float(model.fps)

    cache: dict[str, tuple[PoseSequence3D, np.ndarray, np.ndarray, str]] = {}
    catalog_entries: list[dict[str, Any]] = []
    trace_paths: list[str] = []
    num_ok = 0
    num_fail = 0

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
            sequence = load_pose_sequence3d(pose_entry.pose3d_npz_path)
            source_times = resolve_source_times(sequence)
            root_translation_xyz, root_source = _resolve_root_translation(sequence)
            cache[reference_clip_id] = (sequence, source_times, root_translation_xyz, root_source)
        sequence, source_times, root_translation_xyz, root_source = cache[reference_clip_id]

        window_dir = output_root / prompt_id
        window_dir.mkdir(parents=True, exist_ok=True)

        try:
            constraints_payload, traceability = _build_root2d_constraints_payload(
                sequence=sequence,
                pose_path=Path(pose_entry.pose3d_npz_path),
                source_times=source_times,
                root_translation_xyz=root_translation_xyz,
                root_source=root_source,
                window=window,
                target_fps=target_fps,
                target_model=model_display_name,
                target_skeleton=target_skeleton,
            )
        except Exception:
            num_fail += 1
            continue

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
            "source_frame_range": [
                int(traceability["source_frame_start"]),
                int(traceability["source_frame_end"]),
            ],
            "source_keyframes": list(traceability["source_keyframes"]),
            "target_keyframes": list(traceability["target_keyframes"]),
        }
        entry_payload["window_id"] = prompt_id
        entry_payload["prompt_id"] = prompt_id
        entry_payload["reference_clip_id"] = reference_clip_id
        entry_payload["duration_hint_sec"] = float(window.duration_sec)
        catalog_entries.append(entry_payload)
        num_ok += 1

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
        "num_input_clips": int(len(pose_entries)),
        "num_input_windows": int(len(qwen_entries)),
        "num_ok": int(num_ok),
        "num_fail": int(num_fail),
        "num_total_windows": int(len(catalog_entries)),
        "sample_prompt_ids": [entry["prompt_id"] for entry in catalog_entries[:5]],
        "sample_traceability_paths": trace_paths[:5],
    }
    write_json(summary_path, summary)
    return summary


def _resolve_root_translation(sequence: PoseSequence3D) -> tuple[np.ndarray, str]:
    root_translation = sequence.resolved_root_translation_m()
    if root_translation is not None:
        root_xyz = np.asarray(root_translation, dtype=np.float32)
        if not np.isfinite(root_xyz).all():
            raise ValueError("root_translation_m contains NaN/Inf values.")
        return root_xyz, "root_translation_m"

    joints = np.asarray(sequence.joint_positions_xyz, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[0] != sequence.num_frames or joints.shape[2] != 3:
        raise ValueError("joint_positions_xyz must have shape [T, J, 3].")
    if not np.isfinite(joints[:, 0, :]).all():
        raise ValueError("Pelvis fallback root trajectory contains NaN/Inf values.")
    return joints[:, 0, :], "joint_positions_xyz[:,0,:]"


def _build_root2d_constraints_payload(
    *,
    sequence: PoseSequence3D,
    pose_path: Path,
    source_times: np.ndarray,
    root_translation_xyz: np.ndarray,
    root_source: str,
    window: WindowSpec,
    target_fps: float,
    target_model: str,
    target_skeleton: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    num_target_frames = max(1, int(round(float(window.duration_sec) * float(target_fps))))
    target_frame_indices = np.arange(num_target_frames, dtype=np.int32)
    target_times = target_frame_indices.astype(np.float32) / np.float32(target_fps)

    source_root_window = np.asarray(
        root_translation_xyz[window.source_start_index : window.source_end_index],
        dtype=np.float32,
    )
    if source_root_window.shape[0] <= 0:
        raise ValueError(f"Window {window.prompt_id!r} does not contain source frames.")

    source_times_window = (
        np.asarray(source_times[window.source_start_index : window.source_end_index], dtype=np.float32)
        - np.float32(window.start_sec)
    )
    source_root_2d = source_root_window[:, [0, 2]]
    if not np.isfinite(source_root_2d).all():
        raise ValueError(f"Window {window.prompt_id!r} root2d contains NaN/Inf values.")

    smooth_root_2d = np.stack(
        [
            np.interp(target_times, source_times_window, source_root_2d[:, axis]).astype(np.float32, copy=False)
            for axis in range(2)
        ],
        axis=1,
    )
    global_root_heading = _compute_root_heading(smooth_root_2d)

    constraint_payload: dict[str, Any] = {
        "type": "root2d",
        "frame_indices": [int(value) for value in target_frame_indices.tolist()],
        "smooth_root_2d": smooth_root_2d.astype(np.float32, copy=False).tolist(),
    }
    if global_root_heading is not None:
        constraint_payload["global_root_heading"] = global_root_heading.astype(np.float32, copy=False).tolist()

    selected_source_frames = _map_target_frames_to_source_frames(
        target_frames=target_frame_indices,
        target_fps=target_fps,
        source_times=source_times,
        window=window,
        sequence=sequence,
    )
    source_frame_ids = np.asarray(sequence.frame_indices, dtype=np.int32)
    source_start_frame = int(source_frame_ids[window.source_start_index])
    source_end_frame = int(source_frame_ids[max(window.source_end_index - 1, window.source_start_index)])
    traceability = {
        "prompt_id": window.prompt_id,
        "reference_clip_id": sequence.clip_id,
        "source_pose3d_npz_path": str(pose_path.resolve()),
        "root_source": root_source,
        "source_fps": resolve_sequence_fps(sequence, source_times),
        "target_model": target_model,
        "target_skeleton": target_skeleton,
        "target_fps": float(target_fps),
        "source_frame_start": source_start_frame,
        "source_frame_end": source_end_frame,
        "source_keyframes": [int(value) for value in selected_source_frames.tolist()],
        "target_keyframes": [int(value) for value in target_frame_indices.tolist()],
        "constraint_types": ["root2d"],
        "duration_hint_sec": float(window.duration_sec),
        "num_target_frames": int(num_target_frames),
    }
    return [constraint_payload], traceability


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


def _map_target_frames_to_source_frames(
    *,
    target_frames: np.ndarray,
    target_fps: float,
    source_times: np.ndarray,
    window: WindowSpec,
    sequence: PoseSequence3D,
) -> np.ndarray:
    source_frame_ids = np.asarray(sequence.frame_indices, dtype=np.int32)
    absolute_times = (target_frames.astype(np.float32) / np.float32(target_fps)) + np.float32(window.start_sec)
    mapped: list[int] = []
    for absolute_time in absolute_times.tolist():
        insert_at = int(np.searchsorted(source_times, np.float32(absolute_time), side="left"))
        if insert_at <= 0:
            nearest = 0
        elif insert_at >= int(source_times.size):
            nearest = int(source_times.size) - 1
        else:
            prev_index = insert_at - 1
            next_index = insert_at
            prev_delta = abs(float(source_times[prev_index]) - float(absolute_time))
            next_delta = abs(float(source_times[next_index]) - float(absolute_time))
            nearest = prev_index if prev_delta <= next_delta else next_index
        mapped.append(int(source_frame_ids[nearest]))
    return np.asarray(mapped, dtype=np.int32)
