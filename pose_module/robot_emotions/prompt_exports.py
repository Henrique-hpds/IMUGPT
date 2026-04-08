"""Prompt-catalog construction and synthetic pose3d export for RobotEmotions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from pose_module.interfaces import IMUGPT_22_JOINT_NAMES, PoseSequence3D
from pose_module.io.cache import write_json_file
from pose_module.pipeline import run_pose3d_from_prompt
from pose_module.prompt_source import (
    PromptCatalogEntry,
    PromptMotionBackend,
    PromptSample,
    canonicalize_label_value,
    canonicalize_labels,
    iter_prompt_samples,
    load_prompt_catalog,
    write_prompt_catalog,
)

from .extractor import RobotEmotionsExtractor
from .metadata import PROTOCOL_ROWS


_PELVIS_INDEX = IMUGPT_22_JOINT_NAMES.index("Pelvis")
_NECK_INDEX = IMUGPT_22_JOINT_NAMES.index("Neck")
_LEFT_SHOULDER_INDEX = IMUGPT_22_JOINT_NAMES.index("Left_shoulder")
_RIGHT_SHOULDER_INDEX = IMUGPT_22_JOINT_NAMES.index("Right_shoulder")
_LEFT_ELBOW_INDEX = IMUGPT_22_JOINT_NAMES.index("Left_elbow")
_RIGHT_ELBOW_INDEX = IMUGPT_22_JOINT_NAMES.index("Right_elbow")
_LEFT_WRIST_INDEX = IMUGPT_22_JOINT_NAMES.index("Left_wrist")
_RIGHT_WRIST_INDEX = IMUGPT_22_JOINT_NAMES.index("Right_wrist")
_LEFT_ANKLE_INDEX = IMUGPT_22_JOINT_NAMES.index("Left_ankle")
_RIGHT_ANKLE_INDEX = IMUGPT_22_JOINT_NAMES.index("Right_ankle")


@dataclass(frozen=True)
class RobotEmotionsPromptPreset:
    action_detail: str
    lead: str
    default_cues: tuple[str, ...]


_PROMPT_PRESETS: tuple[RobotEmotionsPromptPreset, ...] = (
    RobotEmotionsPromptPreset(
        action_detail="neutral_standing_posture",
        lead="A person is standing in a neutral way",
        default_cues=("relaxed upright posture", "stable weight distribution", "minimal expressive motion"),
    ),
    RobotEmotionsPromptPreset(
        action_detail="neutral_seated_posture",
        lead="A person is sitting in a neutral way",
        default_cues=("upright seated posture", "calm upper-body motion", "low-amplitude movement"),
    ),
    RobotEmotionsPromptPreset(
        action_detail="neutral_walking",
        lead="A person is walking in a neutral way",
        default_cues=("even natural gait", "regular arm swings", "steady full-body rhythm"),
    ),
    RobotEmotionsPromptPreset(
        action_detail="sad_walking",
        lead="A person is walking sadly",
        default_cues=("lowered body energy", "slower arm swings", "subdued full-body motion"),
    ),
    RobotEmotionsPromptPreset(
        action_detail="sad_seated_stillness",
        lead="A person is sitting sadly",
        default_cues=("slightly collapsed posture", "reduced movement", "low arm activity"),
    ),
    RobotEmotionsPromptPreset(
        action_detail="sad_seated_reflection",
        lead="A person is sitting sadly and reflectively",
        default_cues=("inward upper-body posture", "slow low-amplitude arm motion", "subdued body energy"),
    ),
    RobotEmotionsPromptPreset(
        action_detail="sad_standing_stillness",
        lead="A person is standing sadly",
        default_cues=("lowered posture", "limited gestures", "quiet body dynamics"),
    ),
    RobotEmotionsPromptPreset(
        action_detail="happy_walking",
        lead="A person is walking happily",
        default_cues=("light lively gait", "open posture", "energetic full-body motion"),
    ),
    RobotEmotionsPromptPreset(
        action_detail="happy_seated_reaction",
        lead="A person is sitting and expressing happiness",
        default_cues=("animated upper-body reactions", "open chest", "bright expressive gestures"),
    ),
    RobotEmotionsPromptPreset(
        action_detail="happy_seated_storytelling",
        lead="A person is sitting and expressing happiness while recounting something positive",
        default_cues=("expressive arm gestures", "open posture", "lively upper-body motion"),
    ),
    RobotEmotionsPromptPreset(
        action_detail="happy_standing_arms_raised",
        lead="A person is standing and expressing happiness",
        default_cues=("expansive movement", "animated gestures", "celebratory upper-body expression"),
    ),
    RobotEmotionsPromptPreset(
        action_detail="angry_standing_gestures",
        lead="A person is standing angrily",
        default_cues=("tense posture", "abrupt arm motion", "forceful upper-body movement"),
    ),
    RobotEmotionsPromptPreset(
        action_detail="startled_seated_fear",
        lead="A person is sitting in fear",
        default_cues=("sudden reactive upper-body movement", "tense posture", "quick arm reactions"),
    ),
    RobotEmotionsPromptPreset(
        action_detail="running_away_in_fear",
        lead="A person suddenly turns and runs away in fear",
        default_cues=("urgent acceleration", "forward lean", "reactive arm motion"),
    ),
    RobotEmotionsPromptPreset(
        action_detail="bored_seated_restlessness",
        lead="A person is sitting in boredom",
        default_cues=("low body energy", "minimal expressive motion", "occasional restless shifts"),
    ),
    RobotEmotionsPromptPreset(
        action_detail="bored_standing_restlessness",
        lead="A person is standing in boredom",
        default_cues=("low-energy posture", "small restless adjustments", "minimal gestures"),
    ),
    RobotEmotionsPromptPreset(
        action_detail="searching_with_reaching",
        lead="A person is standing and searching for a hidden object",
        default_cues=("exploratory reaching gestures", "cautious whole-body motion", "small orientation changes"),
    ),
    RobotEmotionsPromptPreset(
        action_detail="searching_with_turning",
        lead="A person is standing and searching for a hidden object",
        default_cues=("exploratory trunk turns", "head-led orientation changes", "scanning upper-body motion"),
    ),
)


def build_robot_emotions_prompt_catalog(
    *,
    dataset_root: str,
    output_path: str | Path,
    real_pose3d_manifest_path: str | Path | None = None,
    domains: Sequence[str] = ("10ms", "30ms"),
    default_num_samples: int = 1,
    default_seed: int = 123,
    default_fps: float = 20.0,
) -> Dict[str, Any]:
    extractor = RobotEmotionsExtractor(dataset_root, domains=tuple(str(domain) for domain in domains))
    records = extractor.scan()
    condition_records = _group_records_by_protocol_row(records)
    condition_motion_stats = _collect_real_motion_stats(
        condition_records=condition_records,
        real_pose3d_manifest_path=real_pose3d_manifest_path,
    )

    catalog_entries: list[PromptCatalogEntry] = []
    num_enriched_conditions = 0
    for condition_index, protocol in enumerate(PROTOCOL_ROWS):
        preset = _PROMPT_PRESETS[condition_index]
        labels = canonicalize_labels(
            {
                "emotion": protocol["emotion"],
                "modality": protocol["modality"],
                "stimulus": protocol["stimulus"],
            }
        )
        reference_clip_ids = sorted(
            record.clip_id for record in condition_records.get(condition_index, [])
        )
        aggregated_stats = condition_motion_stats.get(condition_index)
        if aggregated_stats is not None:
            num_enriched_conditions += 1
        prompt_text = _compose_prompt_text(
            preset=preset,
            protocol=protocol,
            motion_stats=aggregated_stats,
            labels=labels,
        )
        duration_hint_sec = _estimate_duration_hint_sec(
            condition_index=condition_index,
            motion_stats=aggregated_stats,
        )
        prompt_id = _build_prompt_id(
            labels=labels,
            action_detail=preset.action_detail,
            condition_index=condition_index,
        )
        entry = PromptCatalogEntry(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            labels=labels,
            seed=int(default_seed),
            num_samples=int(default_num_samples),
            fps=float(default_fps),
            duration_hint_sec=duration_hint_sec,
            action_detail=str(preset.action_detail),
            stimulus_type=canonicalize_label_value(protocol.get("stimulus")),
            reference_clip_id=reference_clip_ids[0] if len(reference_clip_ids) > 0 else None,
            source_metadata={
                "dataset": "RobotEmotions",
                "condition_index": int(condition_index),
                "stimulus_details": str(protocol.get("stimulus_details", "")),
                "reference_clip_ids": reference_clip_ids,
                "motion_lexicon": None if aggregated_stats is None else dict(aggregated_stats),
            },
        )
        catalog_entries.append(entry)

    output_path = Path(output_path)
    write_prompt_catalog(catalog_entries, output_path)
    summary = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "output_path": str(output_path.resolve()),
        "domains": [str(domain) for domain in domains],
        "num_conditions": int(len(catalog_entries)),
        "num_enriched_conditions": int(num_enriched_conditions),
        "real_pose3d_manifest_path": (
            None if real_pose3d_manifest_path is None else str(Path(real_pose3d_manifest_path).resolve())
        ),
        "sample_prompt_ids": [entry.prompt_id for entry in catalog_entries[:5]],
    }
    summary_path = output_path.with_suffix(".summary.json")
    write_json_file(summary, summary_path)
    return summary


def run_robot_emotions_prompt_pose3d(
    *,
    prompt_catalog_path: str | Path,
    output_dir: str | Path,
    prompt_backend: Optional[PromptMotionBackend] = None,
    export_bvh: bool = True,
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    catalog_entries = load_prompt_catalog(prompt_catalog_path)

    manifest_entries = []
    samples = list(iter_prompt_samples(catalog_entries))
    num_ok = 0
    num_warning = 0
    num_fail = 0
    for sample in samples:
        pose_dir = output_root / "synthetic" / sample.sample_id / "pose"
        try:
            pipeline_result = run_pose3d_from_prompt(
                prompt_id=sample.prompt_id,
                sample_id=sample.sample_id,
                prompt_text=sample.prompt_text,
                output_dir=pose_dir,
                labels=sample.labels,
                seed=int(sample.seed),
                fps=float(sample.fps),
                duration_hint_sec=sample.duration_hint_sec,
                action_detail=sample.action_detail,
                stimulus_type=sample.stimulus_type,
                reference_clip_id=sample.reference_clip_id,
                group_id=sample.group_id,
                notes=sample.notes,
                source_metadata=sample.source_metadata,
                prompt_backend=prompt_backend,
                export_bvh=bool(export_bvh),
            )
            manifest_entry = _build_prompt_pose3d_manifest_entry(
                sample=sample,
                pipeline_result=pipeline_result,
            )
            status = str(manifest_entry.get("status", "fail"))
            if status == "ok":
                num_ok += 1
            elif status == "warning":
                num_warning += 1
            else:
                num_fail += 1
        except Exception as exc:
            manifest_entry = _build_prompt_pose3d_failure_entry(sample=sample, pose_dir=pose_dir, error=str(exc))
            num_fail += 1
        manifest_entries.append(manifest_entry)

    manifest_path = output_root / "prompt_pose3d_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in manifest_entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    summary = {
        "prompt_catalog_path": str(Path(prompt_catalog_path).resolve()),
        "output_dir": str(output_root.resolve()),
        "num_catalog_entries": int(len(catalog_entries)),
        "num_requested_samples": int(len(samples)),
        "num_ok": int(num_ok),
        "num_warning": int(num_warning),
        "num_fail": int(num_fail),
        "prompt_pose3d_manifest_path": str(manifest_path.resolve()),
        "sample_ids": [sample.sample_id for sample in samples[:5]],
    }
    write_json_file(summary, output_root / "prompt_pose3d_summary.json")
    return summary


def _build_prompt_pose3d_manifest_entry(
    *,
    sample: PromptSample,
    pipeline_result: Mapping[str, Any],
) -> Dict[str, Any]:
    pose_sequence = pipeline_result["pose_sequence"]
    quality_report = dict(pipeline_result["quality_report"])
    artifacts = dict(pipeline_result["artifacts"])
    return {
        "clip_id": str(sample.sample_id),
        "dataset": "RobotEmotionsPrompt",
        "status": str(quality_report["status"]),
        "source_kind": "prompt",
        "modality_domain": "synthetic",
        "prompt_id": str(sample.prompt_id),
        "prompt_text": str(sample.prompt_text),
        "sample_index": int(sample.sample_index),
        "seed": int(sample.seed),
        "labels": dict(sample.labels),
        "action_detail": None if sample.action_detail is None else str(sample.action_detail),
        "stimulus_type": None if sample.stimulus_type is None else str(sample.stimulus_type),
        "paired_real_clip_id": (
            None if sample.reference_clip_id is None else str(sample.reference_clip_id)
        ),
        "generation_backend": str(pipeline_result["prompt_metadata"]["generation_backend"]),
        "pose3d": {
            "fps": None if pose_sequence.fps is None else float(pose_sequence.fps),
            "fps_original": None if pose_sequence.fps_original is None else float(pose_sequence.fps_original),
            "num_frames": int(pose_sequence.num_frames),
            "num_joints": int(pose_sequence.num_joints),
            "joint_names_3d": list(pose_sequence.joint_names_3d),
            "source": str(pose_sequence.source),
            "coordinate_space": str(pose_sequence.coordinate_space),
            "has_root_translation": bool(pose_sequence.root_translation_m is not None),
        },
        "quality_report": quality_report,
        "prompt_adapter_quality_report": dict(pipeline_result["prompt_adapter_quality_report"]),
        "prompt_source_quality_report": dict(pipeline_result["prompt_source_quality_report"]),
        "metric_normalizer_quality_report": dict(pipeline_result["metric_normalization_quality_report"]),
        "root_trajectory_quality_report": dict(pipeline_result["root_trajectory_quality_report"]),
        "artifacts": artifacts,
    }


def _build_prompt_pose3d_failure_entry(*, sample: PromptSample, pose_dir: Path, error: str) -> Dict[str, Any]:
    pose_dir.mkdir(parents=True, exist_ok=True)
    quality_report = {
        "clip_id": str(sample.sample_id),
        "status": "fail",
        "source_kind": "prompt",
        "modality_domain": "synthetic",
        "notes": [str(error)],
    }
    write_json_file(quality_report, pose_dir / "quality_report.json")
    return {
        "clip_id": str(sample.sample_id),
        "dataset": "RobotEmotionsPrompt",
        "status": "fail",
        "source_kind": "prompt",
        "modality_domain": "synthetic",
        "prompt_id": str(sample.prompt_id),
        "prompt_text": str(sample.prompt_text),
        "sample_index": int(sample.sample_index),
        "seed": int(sample.seed),
        "labels": dict(sample.labels),
        "action_detail": None if sample.action_detail is None else str(sample.action_detail),
        "stimulus_type": None if sample.stimulus_type is None else str(sample.stimulus_type),
        "paired_real_clip_id": (
            None if sample.reference_clip_id is None else str(sample.reference_clip_id)
        ),
        "quality_report": quality_report,
        "artifacts": {
            "prompt_metadata_json_path": (
                str((pose_dir / "prompt_metadata.json").resolve())
                if (pose_dir / "prompt_metadata.json").exists()
                else None
            ),
            "pose3d_prompt_raw_npz_path": (
                str((pose_dir / "pose3d_prompt_raw.npz").resolve())
                if (pose_dir / "pose3d_prompt_raw.npz").exists()
                else None
            ),
            "pose3d_metric_local_npz_path": (
                str((pose_dir / "pose3d_metric_local.npz").resolve())
                if (pose_dir / "pose3d_metric_local.npz").exists()
                else None
            ),
            "pose3d_npz_path": (
                str((pose_dir / "pose3d.npz").resolve()) if (pose_dir / "pose3d.npz").exists() else None
            ),
            "pose3d_bvh_path": (
                str((pose_dir / "pose3d.bvh").resolve()) if (pose_dir / "pose3d.bvh").exists() else None
            ),
            "quality_report_json_path": str((pose_dir / "quality_report.json").resolve()),
        },
    }


def _group_records_by_protocol_row(records: Sequence[Any]) -> Dict[int, list[Any]]:
    grouped: Dict[int, list[Any]] = {}
    for record in records:
        condition_index = _resolve_protocol_row_index(record.domain, int(record.tag_number))
        if condition_index is None:
            continue
        grouped.setdefault(condition_index, []).append(record)
    return grouped


def _resolve_protocol_row_index(domain: str, tag_number: int) -> int | None:
    key = "tag_10ms" if str(domain) == "10ms" else "tag_30ms"
    for condition_index, protocol in enumerate(PROTOCOL_ROWS):
        if protocol.get(key) == int(tag_number):
            return int(condition_index)
    return None


def _collect_real_motion_stats(
    *,
    condition_records: Mapping[int, Sequence[Any]],
    real_pose3d_manifest_path: str | Path | None,
) -> Dict[int, Dict[str, Any]]:
    if real_pose3d_manifest_path in (None, ""):
        return {}
    manifest_entries = _load_jsonl(real_pose3d_manifest_path)
    manifest_by_clip_id = {
        str(entry["clip_id"]): entry
        for entry in manifest_entries
        if str(entry.get("status", "fail")) in {"ok", "warning"}
    }

    aggregated: Dict[int, Dict[str, Any]] = {}
    for condition_index, records in condition_records.items():
        stats = []
        for record in records:
            entry = manifest_by_clip_id.get(str(record.clip_id))
            if entry is None:
                continue
            pose_path = entry.get("artifacts", {}).get("pose3d_npz_path")
            if pose_path in (None, ""):
                continue
            pose_path = Path(str(pose_path))
            if not pose_path.exists():
                continue
            with np.load(pose_path, allow_pickle=False) as payload:
                sequence = PoseSequence3D.from_npz_payload(payload)
            stats.append(_compute_motion_stats(sequence))
        if len(stats) == 0:
            continue
        aggregated[condition_index] = _aggregate_motion_stats(stats)
    return aggregated


def _compute_motion_stats(sequence: PoseSequence3D) -> Dict[str, float]:
    positions = np.asarray(sequence.joint_positions_xyz, dtype=np.float32)
    fps = _resolve_fps(sequence)
    dt = max(float(1.0 / max(fps, 1e-6)), 1e-6)
    root = sequence.resolved_root_translation_m()
    if root is None:
        root = positions[:, _PELVIS_INDEX, :]
    shoulders = positions[:, [_LEFT_SHOULDER_INDEX, _RIGHT_SHOULDER_INDEX], :]
    wrists = positions[:, [_LEFT_WRIST_INDEX, _RIGHT_WRIST_INDEX], :]
    neck = positions[:, _NECK_INDEX, :]
    pelvis = positions[:, _PELVIS_INDEX, :]
    left_elbow_angle = _joint_angle_deg(
        positions[:, _LEFT_SHOULDER_INDEX, :],
        positions[:, _LEFT_ELBOW_INDEX, :],
        positions[:, _LEFT_WRIST_INDEX, :],
    )
    right_elbow_angle = _joint_angle_deg(
        positions[:, _RIGHT_SHOULDER_INDEX, :],
        positions[:, _RIGHT_ELBOW_INDEX, :],
        positions[:, _RIGHT_WRIST_INDEX, :],
    )
    left_arm_extent = np.linalg.norm(
        positions[:, _LEFT_WRIST_INDEX, :] - positions[:, _LEFT_SHOULDER_INDEX, :],
        axis=1,
    )
    right_arm_extent = np.linalg.norm(
        positions[:, _RIGHT_WRIST_INDEX, :] - positions[:, _RIGHT_SHOULDER_INDEX, :],
        axis=1,
    )
    root_velocity = np.diff(root[:, (0, 2)], axis=0) / np.float32(dt) if len(root) > 1 else np.zeros((0, 2))
    ankle_motion = positions[:, [_LEFT_ANKLE_INDEX, _RIGHT_ANKLE_INDEX], 0]
    ankle_diff = ankle_motion[:, 0] - ankle_motion[:, 1]
    cadence_hz = _estimate_cadence_hz(ankle_diff, fps=fps)
    trunk_vectors = neck - pelvis
    trunk_norm = np.linalg.norm(trunk_vectors, axis=1, keepdims=True)
    trunk_unit = trunk_vectors / np.maximum(trunk_norm, 1e-6)
    vertical_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    trunk_inclination_deg = np.degrees(
        np.arccos(np.clip(np.sum(trunk_unit * vertical_axis[None, :], axis=1), -1.0, 1.0))
    )
    body_velocity = (
        np.diff(positions, axis=0) / np.float32(dt)
        if positions.shape[0] > 1
        else np.zeros((0,) + positions.shape[1:], dtype=np.float32)
    )

    return {
        "num_frames": float(sequence.num_frames),
        "duration_sec": float(sequence.num_frames / max(fps, 1e-6)),
        "root_speed_mean": float(np.mean(np.linalg.norm(root_velocity, axis=1))) if len(root_velocity) > 0 else 0.0,
        "wrist_amplitude_mean": float(np.mean(np.linalg.norm(wrists - shoulders, axis=2))),
        "arm_raise_ratio": float(np.mean(wrists[..., 1] > shoulders[..., 1])),
        "trunk_inclination_deg": float(np.mean(trunk_inclination_deg)),
        "elbow_opening_deg": float(np.mean(np.concatenate([left_elbow_angle, right_elbow_angle], axis=0))),
        "step_cadence_hz": float(cadence_hz),
        "vertical_variation_m": float(np.ptp(root[:, 1])),
        "side_symmetry_score": float(
            1.0
            - np.mean(np.abs(left_arm_extent - right_arm_extent) / np.maximum(left_arm_extent + right_arm_extent, 1e-6))
        ),
        "movement_energy": float(
            np.mean(np.linalg.norm(body_velocity.reshape(body_velocity.shape[0], -1, 3), axis=2))
        )
        if body_velocity.shape[0] > 0
        else 0.0,
    }


def _aggregate_motion_stats(stats: Sequence[Mapping[str, float]]) -> Dict[str, Any]:
    keys = sorted(stats[0].keys())
    return {
        key: float(np.mean([float(item[key]) for item in stats]))
        for key in keys
    }


def _compose_prompt_text(
    *,
    preset: RobotEmotionsPromptPreset,
    protocol: Mapping[str, Any],
    motion_stats: Mapping[str, Any] | None,
    labels: Mapping[str, str],
) -> str:
    cues = list(preset.default_cues)
    if motion_stats is not None:
        cues.extend(_verbalize_motion_stats(labels=labels, motion_stats=motion_stats))
    cue_list = list(dict.fromkeys(cues))
    cue_list = cue_list[:4]

    prompt_parts = [preset.lead]
    if len(cue_list) > 0:
        prompt_parts.append(", ".join(cue_list))
    stimulus_context = _stimulus_context_clause(protocol.get("stimulus"))
    if stimulus_context is not None:
        prompt_parts.append(stimulus_context)
    return ", ".join(prompt_parts).rstrip(".") + "."


def _verbalize_motion_stats(
    *,
    labels: Mapping[str, str],
    motion_stats: Mapping[str, Any],
) -> list[str]:
    modality = str(labels["modality"])
    cues: list[str] = []
    root_speed_mean = float(motion_stats.get("root_speed_mean", 0.0))
    wrist_amplitude_mean = float(motion_stats.get("wrist_amplitude_mean", 0.0))
    arm_raise_ratio = float(motion_stats.get("arm_raise_ratio", 0.0))
    trunk_inclination_deg = float(motion_stats.get("trunk_inclination_deg", 0.0))
    vertical_variation_m = float(motion_stats.get("vertical_variation_m", 0.0))
    side_symmetry_score = float(motion_stats.get("side_symmetry_score", 1.0))
    movement_energy = float(motion_stats.get("movement_energy", 0.0))
    step_cadence_hz = float(motion_stats.get("step_cadence_hz", 0.0))

    if modality in {"walking", "running"}:
        if root_speed_mean >= 0.95:
            cues.append("fast, urgent steps" if modality == "running" else "fast light steps")
        elif root_speed_mean >= 0.45:
            cues.append("steady forward locomotion")
        else:
            cues.append("slow restrained steps")
        if step_cadence_hz >= 1.4:
            cues.append("quick step cadence")
    elif root_speed_mean <= 0.12:
        cues.append("small weight shifts")

    if wrist_amplitude_mean >= 0.65:
        cues.append("expansive arm motion" if modality not in {"walking", "running"} else "larger-than-neutral arm swings")
    elif wrist_amplitude_mean <= 0.30:
        cues.append("low-amplitude arm motion")

    if arm_raise_ratio >= 0.18:
        cues.append("repeatedly lifting the arms above shoulder level")

    if trunk_inclination_deg >= 18.0:
        cues.append("noticeable forward trunk lean")
    elif trunk_inclination_deg <= 8.0:
        cues.append("upright torso")

    if vertical_variation_m >= 0.08:
        cues.append("visible body bounce")

    if movement_energy >= 1.15:
        cues.append("energetic full-body motion")
    elif movement_energy <= 0.45:
        cues.append("subdued body energy")

    if side_symmetry_score <= 0.65:
        cues.append("slight side-to-side asymmetry")
    return cues


def _stimulus_context_clause(stimulus: Any) -> str | None:
    stimulus_value = canonicalize_label_value(stimulus)
    if stimulus_value == "none":
        return None
    if stimulus_value == "music":
        return "as if responding to music"
    if stimulus_value == "visual_methods":
        return "as if reacting to something being watched"
    if stimulus_value == "autobiographical_recall":
        return "as if recalling a personal episode"
    if stimulus_value == "simulation":
        return "as if responding to a simulated situation"
    return f"with context related to {stimulus_value.replace('_', ' ')}"


def _estimate_duration_hint_sec(
    *,
    condition_index: int,
    motion_stats: Mapping[str, Any] | None,
) -> float:
    if motion_stats is not None:
        duration_sec = float(motion_stats.get("duration_sec", 0.0))
        if duration_sec > 0.0:
            return float(max(3.0, min(duration_sec, 8.0)))
    if condition_index == 13:
        return 5.0
    return 6.0


def _build_prompt_id(
    *,
    labels: Mapping[str, str],
    action_detail: str,
    condition_index: int,
) -> str:
    return (
        f"robot_emotions_{labels['emotion']}_{labels['modality']}_{action_detail}_{condition_index + 1:02d}"
    )


def _joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    bc = np.asarray(c, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    ba_norm = ba / np.maximum(np.linalg.norm(ba, axis=1, keepdims=True), 1e-6)
    bc_norm = bc / np.maximum(np.linalg.norm(bc, axis=1, keepdims=True), 1e-6)
    cosine = np.clip(np.sum(ba_norm * bc_norm, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _estimate_cadence_hz(signal: np.ndarray, *, fps: float) -> float:
    signal = np.asarray(signal, dtype=np.float32)
    if signal.shape[0] <= 2:
        return 0.0
    demeaned = signal - np.mean(signal)
    zero_crossings = np.where(np.diff(np.signbit(demeaned)))[0]
    if zero_crossings.size <= 1:
        return 0.0
    duration_sec = float(signal.shape[0] / max(fps, 1e-6))
    if duration_sec <= 0.0:
        return 0.0
    return float((zero_crossings.size / 2.0) / duration_sec)


def _resolve_fps(sequence: PoseSequence3D) -> float:
    if sequence.fps is not None and float(sequence.fps) > 0.0:
        return float(sequence.fps)
    timestamps = np.asarray(sequence.timestamps_sec, dtype=np.float32)
    if timestamps.shape[0] <= 1:
        return 20.0
    deltas = np.diff(timestamps)
    valid = deltas > 1e-6
    if not np.any(valid):
        return 20.0
    return float(1.0 / np.median(deltas[valid]))


def _load_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    entries: list[Dict[str, Any]] = []
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            "Manifest file not found for prompt-catalog enrichment: "
            f"{resolved_path}. "
            "You can pass either a pose3d manifest or a virtual_imu manifest, "
            "as long as it contains artifacts.pose3d_npz_path."
        )
    with resolved_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "":
                continue
            entries.append(dict(json.loads(line)))
    return entries
