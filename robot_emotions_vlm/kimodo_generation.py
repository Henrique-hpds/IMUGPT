"""Batch Kimodo generation from RobotEmotions VLM prompt catalogs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Sequence

from .export import write_json, write_jsonl

DEFAULT_KIMODO_GENERATION_MODEL = "Kimodo-SMPLX-RP-v1"
RICH_CONSTRAINT_TYPES = frozenset({"fullbody", "end-effector"})


@dataclass(frozen=True)
class CatalogPromptEntry:
    """Normalized prompt catalog row used for Kimodo generation."""

    prompt_id: str
    window_id: str | None
    prompt_text: str
    labels: dict[str, Any]
    seed: int | None
    num_samples: int
    reference_clip_id: str | None
    duration_hint_sec: float | None
    constraints_path: str | None
    constraint_summary: dict[str, Any]
    window: dict[str, Any]
    source_metadata: dict[str, Any]
    raw_entry: dict[str, Any]

    @property
    def clip_id(self) -> str:
        return self.reference_clip_id or self.prompt_id


@dataclass(frozen=True)
class KimodoGenerationConfig:
    """Runtime settings for catalog-driven Kimodo generation."""

    model_name: str | None = DEFAULT_KIMODO_GENERATION_MODEL
    duration_sec: float = 5.0
    diffusion_steps: int = 100
    seed: int | None = None
    num_samples: int | None = None
    no_postprocess: bool = False
    bvh: bool = False
    cfg_type: str | None = None
    cfg_weight: tuple[float, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "duration_sec": float(self.duration_sec),
            "diffusion_steps": int(self.diffusion_steps),
            "seed": self.seed,
            "num_samples": self.num_samples,
            "no_postprocess": bool(self.no_postprocess),
            "bvh": bool(self.bvh),
            "cfg_type": self.cfg_type,
            "cfg_weight": None if self.cfg_weight is None else [float(value) for value in self.cfg_weight],
        }


def load_catalog_entries(catalog_path: str | Path) -> list[CatalogPromptEntry]:
    """Load and validate the Kimodo prompt catalog exported by robot_emotions_vlm."""

    entries: list[CatalogPromptEntry] = []
    path = Path(catalog_path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        prompt_id = str(payload.get("prompt_id", "")).strip()
        prompt_text = str(payload.get("prompt_text", "")).strip()
        if not prompt_id:
            raise ValueError(f"Catalog entry in {path} is missing prompt_id")
        if not prompt_text:
            raise ValueError(f"Catalog entry '{prompt_id}' is missing prompt_text")
        seed = payload.get("seed")
        entries.append(
            CatalogPromptEntry(
                prompt_id=prompt_id,
                window_id=_optional_string(payload.get("window_id")),
                prompt_text=prompt_text,
                labels=dict(payload.get("labels") or {}),
                seed=None if seed is None else int(seed),
                num_samples=int(payload.get("num_samples", 1)),
                reference_clip_id=_optional_string(payload.get("reference_clip_id")),
                duration_hint_sec=_optional_float(payload.get("duration_hint_sec")),
                constraints_path=_optional_string(payload.get("constraints_path")),
                constraint_summary=dict(payload.get("constraint_summary") or {}),
                window=dict(payload.get("window") or {}),
                source_metadata=dict(payload.get("source_metadata") or {}),
                raw_entry=dict(payload),
            )
        )
    return entries


def select_catalog_entries(
    entries: Sequence[CatalogPromptEntry],
    *,
    clip_ids: Sequence[str] | None = None,
    prompt_ids: Sequence[str] | None = None,
    window_ids: Sequence[str] | None = None,
) -> list[CatalogPromptEntry]:
    """Optionally filter catalog entries by clip id."""

    selected = list(entries)
    if clip_ids is not None:
        requested = {str(clip_id) for clip_id in clip_ids}
        selected = [entry for entry in selected if entry.clip_id in requested]
        found = {entry.clip_id for entry in entries}
        missing = sorted(requested.difference(found))
        if missing:
            raise ValueError(f"Unknown clip_id values requested in catalog: {missing}")
    if prompt_ids is not None:
        requested = {str(prompt_id) for prompt_id in prompt_ids}
        selected = [entry for entry in selected if entry.prompt_id in requested]
        found = {entry.prompt_id for entry in entries}
        missing = sorted(requested.difference(found))
        if missing:
            raise ValueError(f"Unknown prompt_id values requested in catalog: {missing}")
    if window_ids is not None:
        requested = {str(window_id) for window_id in window_ids}
        selected = [entry for entry in selected if (entry.window_id or entry.prompt_id) in requested]
        found = {(entry.window_id or entry.prompt_id) for entry in entries}
        missing = sorted(requested.difference(found))
        if missing:
            raise ValueError(f"Unknown window_id values requested in catalog: {missing}")
    return selected


def _resolve_postprocess_flag(
    *,
    entry: CatalogPromptEntry,
    resolved_model: str,
    config: KimodoGenerationConfig,
    runtime_notes: list[str],
) -> bool:
    del entry, runtime_notes
    return False if "g1" in resolved_model else (not config.no_postprocess)


def generate_kimodo_from_catalog(
    *,
    catalog_path: str | Path,
    output_dir: str | Path,
    clip_ids: Sequence[str] | None = None,
    prompt_ids: Sequence[str] | None = None,
    window_ids: Sequence[str] | None = None,
    model_name: str | None = DEFAULT_KIMODO_GENERATION_MODEL,
    duration_sec: float = 5.0,
    diffusion_steps: int = 100,
    seed: int | None = None,
    num_samples: int | None = None,
    no_postprocess: bool = False,
    bvh: bool = False,
    cfg_type: str | None = None,
    cfg_weight: Sequence[float] | None = None,
    runtime: Any = None,
) -> dict[str, Any]:
    """Generate one Kimodo motion per catalog entry and save a batch manifest."""

    entries = select_catalog_entries(
        load_catalog_entries(catalog_path),
        clip_ids=clip_ids,
        prompt_ids=prompt_ids,
        window_ids=window_ids,
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    config = KimodoGenerationConfig(
        model_name=model_name,
        duration_sec=float(duration_sec),
        diffusion_steps=int(diffusion_steps),
        seed=None if seed is None else int(seed),
        num_samples=None if num_samples is None else int(num_samples),
        no_postprocess=bool(no_postprocess),
        bvh=bool(bvh),
        cfg_type=cfg_type,
        cfg_weight=None if cfg_weight is None else tuple(float(value) for value in cfg_weight),
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

    manifest_entries: list[dict[str, Any]] = []
    num_ok = 0
    num_warning = 0
    num_fail = 0
    num_generated_samples = 0

    for entry in entries:
        prompt_output_dir = output_root / entry.prompt_id
        prompt_output_dir.mkdir(parents=True, exist_ok=True)

        effective_seed = entry.seed if config.seed is None else config.seed
        effective_num_samples = entry.num_samples if config.num_samples is None else config.num_samples
        if entry.constraints_path is not None and entry.duration_hint_sec is None:
            raise ValueError(
                f"Catalog entry {entry.prompt_id!r} must provide duration_hint_sec when constraints_path is set."
            )
        effective_duration_sec = (
            float(entry.duration_hint_sec)
            if entry.duration_hint_sec is not None
            else float(config.duration_sec)
        )
        effective_num_frames = max(1, int(round(effective_duration_sec * float(model.fps))))
        warnings: list[str] = []
        runtime_notes: list[str] = []
        declared_constraint_types = _resolve_declared_constraint_types(entry)
        use_postprocess = _resolve_postprocess_flag(
            entry=entry,
            resolved_model=resolved_model,
            config=config,
            runtime_notes=runtime_notes,
        )
        if _uses_rich_constraints(declared_constraint_types) and not _is_smplx_generation_target(
            model_display_name=model_display_name,
            resolved_model=resolved_model,
            target_skeleton=target_skeleton,
            runtime_skeleton=getattr(model, "skeleton", None),
        ):
            raise ValueError(
                "Rich constraints are only supported for SMPLX generation targets; "
                f"got model={model_display_name!r}, skeleton={target_skeleton!r}."
            )

        prompt_entry_path = prompt_output_dir / "prompt_entry.json"
        generation_config_path = prompt_output_dir / "generation_config.json"
        write_json(prompt_entry_path, entry.raw_entry)
        write_json(
            generation_config_path,
            {
                "prompt_id": entry.prompt_id,
                "window_id": entry.window_id,
                "reference_clip_id": entry.reference_clip_id,
                "resolved_model": resolved_model,
                "model_display_name": model_display_name,
                "runtime_config": config.to_dict(),
                "effective_seed": effective_seed,
                "effective_num_samples": effective_num_samples,
                "effective_duration_sec": effective_duration_sec,
                "effective_num_frames": effective_num_frames,
                "effective_post_processing": bool(use_postprocess),
                "fps": float(model.fps),
                "constraints_path": entry.constraints_path,
                "constraint_summary": dict(entry.constraint_summary),
                "declared_constraint_types": list(declared_constraint_types),
                "runtime_notes": list(runtime_notes),
                "target_skeleton": target_skeleton,
            },
        )

        try:
            if effective_seed is not None:
                runtime.seed_everything(int(effective_seed))

            cfg_kwargs = _resolve_cfg_kwargs(config.cfg_type, config.cfg_weight)
            constraint_lst = []
            if entry.constraints_path is not None:
                constraint_lst = runtime.load_constraints(
                    entry.constraints_path,
                    skeleton=model.skeleton,
                    device=device,
                )
            loaded_constraint_types = _describe_loaded_constraints(constraint_lst)
            output = model(
                entry.prompt_text,
                effective_num_frames,
                num_denoising_steps=int(config.diffusion_steps),
                constraint_lst=constraint_lst,
                num_samples=int(effective_num_samples),
                multi_prompt=False,
                post_processing=use_postprocess,
                return_numpy=True,
                **cfg_kwargs,
            )
            artifacts = runtime.save_outputs(
                output=output,
                output_stem=prompt_output_dir / "motion",
                resolved_model=resolved_model,
                skeleton=model.skeleton,
                fps=float(model.fps),
                export_bvh=bool(config.bvh),
                device=device,
            )
            status = "warning" if warnings else "ok"
            if status == "ok":
                num_ok += 1
            else:
                num_warning += 1
            error_text = None
        except Exception as exc:  # pragma: no cover - integration failure path
            status = "fail"
            error_text = f"{type(exc).__name__}: {exc}"
            artifacts = {}
            loaded_constraint_types = []
            num_fail += 1
            trace_path = prompt_output_dir / "error_trace.txt"
            trace_path.write_text(traceback.format_exc(), encoding="utf-8")
            artifacts["error_trace_txt_path"] = str(trace_path.resolve())

        base_artifacts = {
            "prompt_entry_json_path": str(prompt_entry_path.resolve()),
            "generation_config_json_path": str(generation_config_path.resolve()),
        }
        if status == "fail":
            manifest_entries.append(
                {
                    "clip_id": entry.clip_id,
                    "prompt_id": entry.prompt_id,
                    "window_id": entry.window_id,
                    "reference_clip_id": entry.reference_clip_id,
                    "sample_index": None,
                    "sample_id": None,
                    "status": status,
                    "error": error_text,
                    "prompt_text": entry.prompt_text,
                    "labels": dict(entry.labels),
                    "model_name": model_display_name,
                    "resolved_model": resolved_model,
                    "seed": effective_seed,
                    "num_samples_requested": int(effective_num_samples),
                    "duration_sec": float(effective_duration_sec),
                    "num_frames": int(effective_num_frames),
                    "fps": float(model.fps),
                    "diffusion_steps": int(config.diffusion_steps),
                    "constraints_path": entry.constraints_path,
                    "constraint_summary": dict(entry.constraint_summary),
                    "declared_constraint_types": list(declared_constraint_types),
                    "loaded_constraint_types": list(loaded_constraint_types),
                    "num_constraints_loaded": int(len(loaded_constraint_types)),
                    "post_processing": bool(use_postprocess),
                    "runtime_notes": list(runtime_notes),
                    "warnings": warnings,
                    "artifacts": {
                        **base_artifacts,
                        **artifacts,
                    },
                }
            )
            continue

        sample_entries = _build_sample_manifest_entries(
            entry=entry,
            model_display_name=model_display_name,
            resolved_model=resolved_model,
            effective_seed=effective_seed,
            effective_num_samples=int(effective_num_samples),
            effective_duration_sec=float(effective_duration_sec),
            effective_num_frames=int(effective_num_frames),
            fps=float(model.fps),
            diffusion_steps=int(config.diffusion_steps),
            post_processing=bool(use_postprocess),
            runtime_notes=runtime_notes,
            warnings=warnings,
            constraint_summary=dict(entry.constraint_summary),
            declared_constraint_types=declared_constraint_types,
            loaded_constraint_types=loaded_constraint_types,
            base_artifacts=base_artifacts,
            output_artifacts=artifacts,
        )
        manifest_entries.extend(sample_entries)
        num_generated_samples += len(sample_entries)

    manifest_path = output_root / "kimodo_generation_manifest.jsonl"
    summary_path = output_root / "kimodo_generation_summary.json"
    summary = {
        "catalog_path": str(Path(catalog_path).resolve()),
        "output_dir": str(output_root.resolve()),
        "model_name": model_display_name,
        "resolved_model": resolved_model,
        "num_total": int(len(entries)),
        "num_ok": int(num_ok),
        "num_warning": int(num_warning),
        "num_fail": int(num_fail),
        "num_generated_samples": int(num_generated_samples),
        "num_manifest_entries": int(len(manifest_entries)),
        "manifest_path": str(manifest_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "config": config.to_dict(),
    }
    write_jsonl(manifest_path, manifest_entries)
    write_json(summary_path, summary)
    return summary


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _resolve_declared_constraint_types(entry: CatalogPromptEntry) -> tuple[str, ...]:
    summary_types = entry.constraint_summary.get("constraint_types")
    if isinstance(summary_types, list):
        cleaned = [str(value).strip() for value in summary_types if str(value).strip()]
        if cleaned:
            return tuple(dict.fromkeys(cleaned))
    if entry.constraints_path is None:
        return ()
    payload = json.loads(Path(entry.constraints_path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError(f"Constraint payload at {entry.constraints_path!r} must be a list or dict.")
    types = [
        str(item.get("type", "")).strip()
        for item in payload
        if isinstance(item, dict) and str(item.get("type", "")).strip()
    ]
    return tuple(dict.fromkeys(types))


def _uses_rich_constraints(constraint_types: Sequence[str]) -> bool:
    return any(constraint_type in RICH_CONSTRAINT_TYPES for constraint_type in constraint_types)


def _is_smplx_generation_target(
    *,
    model_display_name: str,
    resolved_model: str,
    target_skeleton: Any,
    runtime_skeleton: Any,
) -> bool:
    if str(target_skeleton or "").upper() == "SMPLX":
        return True
    if _is_smplx_model(model_display_name) or _is_smplx_model(resolved_model):
        return True
    return _is_smplx_model(getattr(runtime_skeleton, "name", ""))


def _describe_loaded_constraints(constraint_lst: Sequence[Any]) -> list[str]:
    types: list[str] = []
    for constraint in constraint_lst:
        if isinstance(constraint, dict):
            name = constraint.get("type")
        else:
            name = getattr(constraint, "name", None) or type(constraint).__name__
        text = str(name).strip()
        if text:
            types.append(text)
    return list(dict.fromkeys(types))


def _resolve_cfg_kwargs(cfg_type: str | None, cfg_weight: Sequence[float] | None) -> dict[str, Any]:
    if cfg_type is None and cfg_weight is None:
        return {}
    if cfg_type == "nocfg":
        if cfg_weight is not None:
            raise ValueError("--cfg-weight cannot be used with --cfg-type nocfg")
        return {"cfg_type": "nocfg"}
    if cfg_type == "regular":
        if cfg_weight is None or len(cfg_weight) != 1:
            raise ValueError("--cfg-type regular requires exactly one --cfg-weight value")
        return {"cfg_type": "regular", "cfg_weight": float(cfg_weight[0])}
    if cfg_type == "separated":
        if cfg_weight is None or len(cfg_weight) != 2:
            raise ValueError("--cfg-type separated requires exactly two --cfg-weight values")
        return {"cfg_type": "separated", "cfg_weight": [float(cfg_weight[0]), float(cfg_weight[1])]}
    if cfg_type is None:
        if cfg_weight is None:
            return {}
        if len(cfg_weight) == 1:
            return {"cfg_type": "regular", "cfg_weight": float(cfg_weight[0])}
        if len(cfg_weight) == 2:
            return {"cfg_type": "separated", "cfg_weight": [float(cfg_weight[0]), float(cfg_weight[1])]}
    raise ValueError(f"Unsupported cfg combination: cfg_type={cfg_type!r}, cfg_weight={cfg_weight!r}")


def _build_sample_manifest_entries(
    *,
    entry: CatalogPromptEntry,
    model_display_name: str,
    resolved_model: str,
    effective_seed: int | None,
    effective_num_samples: int,
    effective_duration_sec: float,
    effective_num_frames: int,
    fps: float,
    diffusion_steps: int,
    post_processing: bool,
    runtime_notes: Sequence[str],
    warnings: Sequence[str],
    constraint_summary: dict[str, Any],
    declared_constraint_types: Sequence[str],
    loaded_constraint_types: Sequence[str],
    base_artifacts: dict[str, Any],
    output_artifacts: dict[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for sample_index in range(int(effective_num_samples)):
        sample_id = f"{entry.prompt_id}__s{sample_index:03d}"
        sample_artifacts = dict(base_artifacts)
        sample_artifacts.update(_select_sample_artifacts(output_artifacts, sample_index=sample_index))
        entries.append(
            {
                "clip_id": entry.clip_id,
                "prompt_id": entry.prompt_id,
                "window_id": entry.window_id,
                "reference_clip_id": entry.reference_clip_id,
                "sample_index": int(sample_index),
                "sample_id": sample_id,
                "status": "warning" if warnings else "ok",
                "error": None,
                "prompt_text": entry.prompt_text,
                "labels": dict(entry.labels),
                "model_name": model_display_name,
                "resolved_model": resolved_model,
                "seed": effective_seed,
                "num_samples_requested": int(effective_num_samples),
                "duration_sec": float(effective_duration_sec),
                "num_frames": int(effective_num_frames),
                "fps": float(fps),
                "diffusion_steps": int(diffusion_steps),
                "post_processing": bool(post_processing),
                "constraints_path": entry.constraints_path,
                "constraint_summary": dict(constraint_summary),
                "declared_constraint_types": list(declared_constraint_types),
                "loaded_constraint_types": list(loaded_constraint_types),
                "num_constraints_loaded": int(len(loaded_constraint_types)),
                "runtime_notes": list(runtime_notes),
                "warnings": list(warnings),
                "artifacts": sample_artifacts,
            }
        )
    return entries


def _select_sample_artifacts(output_artifacts: dict[str, Any], *, sample_index: int) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    if "kimodo_npz_paths" in output_artifacts:
        artifacts["kimodo_npz_path"] = str(output_artifacts["kimodo_npz_paths"][sample_index])
    elif sample_index == 0 and "kimodo_npz_path" in output_artifacts:
        artifacts["kimodo_npz_path"] = str(output_artifacts["kimodo_npz_path"])

    if "amass_npz_paths" in output_artifacts:
        artifacts["amass_npz_path"] = str(output_artifacts["amass_npz_paths"][sample_index])
    elif sample_index == 0 and "amass_npz_path" in output_artifacts:
        artifacts["amass_npz_path"] = str(output_artifacts["amass_npz_path"])

    if "bvh_paths" in output_artifacts:
        if sample_index < len(output_artifacts["bvh_paths"]):
            artifacts["bvh_path"] = str(output_artifacts["bvh_paths"][sample_index])
    elif sample_index == 0 and "bvh_path" in output_artifacts:
        artifacts["bvh_path"] = str(output_artifacts["bvh_path"])

    if "g1_csv_path" in output_artifacts:
        artifacts["g1_csv_path"] = str(output_artifacts["g1_csv_path"])
    return artifacts


class _KimodoRuntime:
    """Minimal lazy bridge to the local Kimodo package."""

    def __init__(self) -> None:
        kimodo_module = _import_kimodo_package()
        self._kimodo = kimodo_module
        from kimodo import DEFAULT_MODEL, load_model
        from kimodo.constraints import load_constraints_lst
        from kimodo.exports.motion_io import save_kimodo_npz
        from kimodo.model.registry import get_model_info
        from kimodo.tools import seed_everything

        self.default_model = DEFAULT_MODEL
        self.load_model = load_model
        self._load_constraints_lst = load_constraints_lst
        self.save_kimodo_npz = save_kimodo_npz
        self.get_model_info = get_model_info
        self.seed_everything = seed_everything

    def resolve_device(self) -> str:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def save_outputs(
        self,
        *,
        output: dict[str, Any],
        output_stem: Path,
        resolved_model: str,
        skeleton: Any,
        fps: float,
        export_bvh: bool,
        device: str,
    ) -> dict[str, Any]:
        n_samples = int(output["posed_joints"].shape[0])
        artifacts: dict[str, Any] = {}

        if n_samples == 1:
            single = _slice_output_sample(output, 0, n_samples)
            npz_path = output_stem.with_suffix(".npz")
            self.save_kimodo_npz(str(npz_path), single)
            artifacts["kimodo_npz_path"] = str(npz_path.resolve())

            if _is_smplx_model(resolved_model):
                artifacts["amass_npz_path"] = _save_smplx_amass_outputs(
                    output=output,
                    output_path=npz_path,
                    skeleton=skeleton,
                    fps=fps,
                )

            if _is_g1_model(resolved_model):
                from kimodo.exports.mujoco import MujocoQposConverter

                csv_path = output_stem.with_suffix(".csv")
                converter = MujocoQposConverter(skeleton)
                qpos = converter.dict_to_qpos(output, device)
                converter.save_csv(qpos, str(csv_path))
                artifacts["g1_csv_path"] = str(csv_path.resolve())

            if export_bvh:
                bvh_path = _export_bvh_if_supported(
                    output=output,
                    output_path=output_stem.with_suffix(".bvh"),
                    skeleton=skeleton,
                    fps=fps,
                    device=device,
                )
                if bvh_path is not None:
                    artifacts["bvh_path"] = str(Path(bvh_path).resolve())
        else:
            motion_dir = output_stem
            motion_dir.mkdir(parents=True, exist_ok=True)
            npz_paths: list[str] = []
            for sample_index in range(n_samples):
                single = _slice_output_sample(output, sample_index, n_samples)
                sample_npz_path = motion_dir / f"{output_stem.name}_{sample_index:02d}.npz"
                self.save_kimodo_npz(str(sample_npz_path), single)
                npz_paths.append(str(sample_npz_path.resolve()))
            artifacts["kimodo_npz_paths"] = npz_paths

            if _is_g1_model(resolved_model):
                from kimodo.exports.mujoco import MujocoQposConverter

                converter = MujocoQposConverter(skeleton)
                qpos = converter.dict_to_qpos(output, device)
                csv_path = motion_dir / f"{output_stem.name}.csv"
                converter.save_csv(qpos, str(csv_path))
                artifacts["g1_csv_path"] = str(csv_path.resolve())

            if _is_smplx_model(resolved_model):
                amass_paths: list[str] = []
                for sample_index, sample_npz_path in enumerate(npz_paths):
                    sample_output = _slice_output_sample(output, sample_index, n_samples)
                    amass_paths.append(
                        _save_smplx_amass_outputs(
                            output=sample_output,
                            output_path=Path(sample_npz_path),
                            skeleton=skeleton,
                            fps=fps,
                        )
                    )
                artifacts["amass_npz_paths"] = amass_paths

            if export_bvh:
                bvh_paths: list[str] = []
                for sample_index in range(n_samples):
                    sample_output = _slice_output_sample(output, sample_index, n_samples)
                    sample_bvh_path = motion_dir / f"{output_stem.name}_{sample_index:02d}.bvh"
                    saved = _export_bvh_if_supported(
                        output=sample_output,
                        output_path=sample_bvh_path,
                        skeleton=skeleton,
                        fps=fps,
                        device=device,
                    )
                    if saved is not None:
                        bvh_paths.append(str(Path(saved).resolve()))
                if bvh_paths:
                    artifacts["bvh_paths"] = bvh_paths

        return artifacts

    def load_constraints(self, path: str | Path, *, skeleton: Any, device: str) -> list[Any]:
        return self._load_constraints_lst(str(path), skeleton, device=device)


def _slice_output_sample(output: dict[str, Any], sample_index: int, n_samples: int) -> dict[str, Any]:
    return {
        key: (
            value[sample_index]
            if hasattr(value, "shape") and len(value.shape) > 0 and int(value.shape[0]) == n_samples
            else value
        )
        for key, value in output.items()
    }


def _is_smplx_model(model_name: str) -> bool:
    return "smplx" in str(model_name).lower()


def _is_g1_model(model_name: str) -> bool:
    return "g1" in str(model_name).lower()


def _save_smplx_amass_outputs(
    *,
    output: dict[str, Any],
    output_path: Path,
    skeleton: Any,
    fps: float,
    converter_factory: Any = None,
) -> str:
    """Save the SMPL-X AMASS export next to the generated Kimodo NPZ."""

    if converter_factory is None:
        from kimodo.exports.smplx import AMASSConverter

        converter_factory = AMASSConverter

    amass_path = output_path.with_name(f"{output_path.stem}_amass.npz")
    converter = converter_factory(skeleton=skeleton, fps=fps)
    converter.convert_save_npz(output, str(amass_path))
    return str(amass_path.resolve())


def _export_bvh_if_supported(
    *,
    output: dict[str, Any],
    output_path: Path,
    skeleton: Any,
    fps: float,
    device: str,
) -> str | None:
    if "somaskel" not in getattr(skeleton, "name", ""):
        return None

    import torch
    from kimodo.exports.bvh import save_motion_bvh
    from kimodo.skeleton import SOMASkeleton30, global_rots_to_local_rots

    bvh_skeleton = skeleton
    if isinstance(bvh_skeleton, SOMASkeleton30):
        bvh_skeleton = bvh_skeleton.somaskel77.to(device)

    joints_pos = torch.from_numpy(output["posed_joints"]).to(device)
    joints_rot = torch.from_numpy(output["global_rot_mats"]).to(device)
    local_rot_mats = global_rots_to_local_rots(joints_rot, bvh_skeleton)
    root_positions = joints_pos[:, bvh_skeleton.root_idx, :]
    save_motion_bvh(str(output_path), local_rot_mats, root_positions, skeleton=bvh_skeleton, fps=fps)
    return str(output_path)


def _load_kimodo_runtime() -> _KimodoRuntime:
    return _KimodoRuntime()


def _import_kimodo_package():
    try:
        import kimodo as kimodo_module

        if hasattr(kimodo_module, "load_model"):
            return kimodo_module
    except Exception:
        kimodo_module = None

    local_package_root = Path(__file__).resolve().parents[1] / "kimodo"
    local_package_root_str = str(local_package_root)
    if local_package_root_str not in sys.path:
        sys.path.insert(0, local_package_root_str)
    stale = sys.modules.get("kimodo")
    if stale is not None and not hasattr(stale, "load_model"):
        del sys.modules["kimodo"]
    import kimodo as kimodo_module

    if not hasattr(kimodo_module, "load_model"):
        raise RuntimeError(
            "Kimodo is not importable as a Python package. "
            "Install it with 'cd kimodo && pip install -e .' or keep the local kimodo submodule available."
        )
    return kimodo_module
