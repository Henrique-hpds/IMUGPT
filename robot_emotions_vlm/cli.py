"""Standalone CLI for RobotEmotions video description with Qwen3-VL."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
import time
from typing import Any, Sequence

from .dataset import RobotEmotionsDataset, read_video_metadata
from .export import (
    build_kimodo_catalog_entry,
    build_manifest_entry,
    write_clip_artifacts,
    write_json,
    write_root_outputs,
)
from .anchor_catalog import build_anchor_catalog
from .kimodo_generation import generate_kimodo_from_catalog
from .kimodo_generation import DEFAULT_KIMODO_GENERATION_MODEL
from .prompts import (
    DEFAULT_SYSTEM_PROMPT_PATH,
    DEFAULT_USER_PROMPT_PATH,
    render_prompts,
)
from .qwen_backend import QwenGenerationConfig, QwenVideoBackend
from .schemas import DescriptionValidationError, VideoDescription, parse_model_response
from .window_descriptions import describe_windows


def export_kimodo_virtual_imu(
    *,
    kimodo_manifest_path: str | Path,
    output_dir: str | Path,
    sensor_layout_path: str | None = None,
    imu_acc_noise_std_m_s2: float | None = None,
    imu_gyro_noise_std_rad_s: float | None = None,
    imu_random_seed: int = 0,
    export_bvh: bool = False,
    skip_existing: bool = True,
    clip_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Batch-process a Kimodo generation manifest → virtual IMU.

    Reads a JSONL manifest produced by generate-kimodo (or batch_kimodo_pose3d),
    runs metric normalisation + root estimation + IK + IMUSim on each ok entry,
    and writes a virtual_imu_manifest.jsonl alongside a summary.
    """
    from pose_module.pipeline import run_virtual_imu_from_kimodo

    manifest_path = Path(kimodo_manifest_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    entries = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ok_entries = [e for e in entries if e.get("status") == "ok"]
    if clip_ids is not None:
        clip_id_set = set(clip_ids)
        ok_entries = [e for e in ok_entries if e.get("clip_id") in clip_id_set or e.get("prompt_id") in clip_id_set or e.get("sample_id") in clip_id_set]

    t_start = time.time()
    result_entries: list[dict[str, Any]] = []
    num_ok = num_skip = num_fail = 0

    for i, entry in enumerate(ok_entries):
        prompt_id = str(entry.get("prompt_id", entry.get("sample_id", f"entry_{i}")))
        sample_id = str(entry.get("sample_id") or prompt_id)
        clip_id = sample_id
        kimodo_npz = str(entry.get("artifacts", {}).get("kimodo_npz_path", ""))
        gen_cfg = entry.get("artifacts", {}).get("generation_config_json_path") or None
        clip_output_dir = output_root / prompt_id

        if skip_existing and (clip_output_dir / "virtual_imu" / "virtual_imu.npz").exists():
            result_entries.append({
                **entry,
                "pose_kind": "synthetic",
                "virtual_imu_artifacts": {
                    "virtual_imu_npz_path": str((clip_output_dir / "virtual_imu" / "virtual_imu.npz").resolve()),
                },
            })
            num_skip += 1
            continue

        try:
            result = run_virtual_imu_from_kimodo(
                clip_id=clip_id,
                kimodo_npz_path=kimodo_npz,
                output_dir=clip_output_dir,
                generation_config_path=gen_cfg,
                export_bvh=bool(export_bvh),
                sensor_layout_path=sensor_layout_path or None,
                imu_acc_noise_std_m_s2=imu_acc_noise_std_m_s2,
                imu_gyro_noise_std_rad_s=imu_gyro_noise_std_rad_s,
                imu_random_seed=int(imu_random_seed),
            )
            virtual_imu_seq = result["virtual_imu_sequence"]
            result_entries.append({
                **entry,
                "pose_kind": "synthetic",
                "virtual_imu": {
                    "fps": None if virtual_imu_seq.fps is None else float(virtual_imu_seq.fps),
                    "num_frames": int(virtual_imu_seq.num_frames),
                    "num_sensors": int(virtual_imu_seq.num_sensors),
                    "sensor_names": list(virtual_imu_seq.sensor_names),
                    "source": str(virtual_imu_seq.source),
                },
                "virtual_imu_artifacts": result["artifacts"],
            })
            num_ok += 1
        except Exception as exc:
            result_entries.append({
                **entry,
                "pose_kind": "synthetic",
                "status": "fail",
                "virtual_imu_error": f"{type(exc).__name__}: {exc}",
            })
            num_fail += 1
            err_path = output_root / prompt_id / "virtual_imu_error.txt"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            err_path.write_text(traceback.format_exc())
            print(f"  FAIL [{i+1}] {prompt_id}: {exc}", flush=True)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            remaining = (len(ok_entries) - i - 1) / max(rate, 1e-9)
            print(
                f"  [{i+1}/{len(ok_entries)}] ok={num_ok} skip={num_skip} fail={num_fail}"
                f" | {rate:.1f} it/s | ~{remaining/60:.1f} min left",
                flush=True,
            )

    manifest_out = output_root / "virtual_imu_manifest.jsonl"
    manifest_out.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in result_entries) + "\n",
        encoding="utf-8",
    )
    summary = {
        "kimodo_manifest_path": str(manifest_path.resolve()),
        "output_dir": str(output_root.resolve()),
        "num_total": len(ok_entries),
        "num_ok": num_ok,
        "num_skip": num_skip,
        "num_fail": num_fail,
        "elapsed_sec": round(time.time() - t_start, 1),
        "virtual_imu_manifest_path": str(manifest_out.resolve()),
    }
    (output_root / "virtual_imu_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def export_mixed_virtual_imu(
    *,
    real_manifest_path: str | Path,
    synthetic_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Merge a real virtual_imu_manifest and a synthetic virtual_imu_manifest into one.

    Reads both JSONL manifests, tags each entry with pose_kind=real/synthetic,
    and writes a combined mixed_virtual_imu_manifest.jsonl. No recomputation is done.
    """
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    def _load_manifest(path: Path) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    real_entries = _load_manifest(Path(real_manifest_path))
    synthetic_entries = _load_manifest(Path(synthetic_manifest_path))

    combined: list[dict[str, Any]] = []
    for entry in real_entries:
        combined.append({**entry, "pose_kind": entry.get("pose_kind", "real")})
    for entry in synthetic_entries:
        combined.append({**entry, "pose_kind": entry.get("pose_kind", "synthetic")})

    num_real = len(real_entries)
    num_synthetic = len(synthetic_entries)
    num_valid_real = sum(1 for e in real_entries if e.get("status") in ("ok", "warning"))
    num_valid_synthetic = sum(1 for e in synthetic_entries if e.get("status") in ("ok", "warning"))
    num_fail_real = sum(1 for e in real_entries if e.get("status") == "fail")
    num_fail_synthetic = sum(1 for e in synthetic_entries if e.get("status") == "fail")

    manifest_out = output_root / "mixed_virtual_imu_manifest.jsonl"
    manifest_out.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in combined) + "\n",
        encoding="utf-8",
    )
    summary = {
        "real_manifest_path": str(Path(real_manifest_path).resolve()),
        "synthetic_manifest_path": str(Path(synthetic_manifest_path).resolve()),
        "output_dir": str(output_root.resolve()),
        "num_real": num_real,
        "num_synthetic": num_synthetic,
        "num_total": num_real + num_synthetic,
        "num_valid_real": num_valid_real,
        "num_valid_synthetic": num_valid_synthetic,
        "num_fail_real": num_fail_real,
        "num_fail_synthetic": num_fail_synthetic,
        "mixed_virtual_imu_manifest_path": str(manifest_out.resolve()),
    }
    (output_root / "mixed_virtual_imu_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def describe_videos(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    domains: Sequence[str] = ("10ms", "30ms"),
    clip_ids: Sequence[str] | None = None,
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    local_files_only: bool = False,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    attn_implementation: str = "sdpa",
    num_video_frames: int = 32,
    max_new_tokens: int = 384,
    temperature: float = 0.2,
    top_p: float = 0.9,
    system_prompt_path: str | Path = DEFAULT_SYSTEM_PROMPT_PATH,
    user_prompt_path: str | Path = DEFAULT_USER_PROMPT_PATH,
    catalog_output_path: str | Path | None = None,
    seed: int = 123,
    num_samples: int = 1,
    backend: QwenVideoBackend | None = None,
) -> dict[str, Any]:
    """Execute the standalone RobotEmotions-to-kimodo prompt export flow."""

    started_at = time.perf_counter()
    dataset = RobotEmotionsDataset(dataset_root, domains=tuple(str(domain) for domain in domains))
    records = dataset.select_records(clip_ids=None if clip_ids is None else list(clip_ids))

    generation_config = QwenGenerationConfig(
        model_id=model_id,
        local_files_only=bool(local_files_only),
        device_map=device_map,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        num_video_frames=int(num_video_frames),
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
    )
    if backend is None:
        backend = QwenVideoBackend(generation_config)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []
    catalog_entries: list[dict[str, Any]] = []
    num_ok = 0
    num_warning = 0
    num_fail = 0

    for record in records:
        raw_response = ""
        description: VideoDescription | None = None
        parsed_warnings: list[str] = []
        errors: list[str] = []

        video_metadata = read_video_metadata(record.video_path)
        prompt_bundle = render_prompts(
            record,
            video_metadata,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
        )
        prompt_context = {
            "clip": record.to_dict(),
            "video_metadata": video_metadata,
            "model_id": model_id,
            "generation_config": generation_config.to_dict(),
            "prompt_bundle": prompt_bundle,
        }

        started_clip = time.perf_counter()
        try:
            raw_response = backend.describe_video(
                record.video_path,
                system_prompt=prompt_bundle["system_prompt"],
                user_prompt=prompt_bundle["user_prompt"],
            )
            parsed = parse_model_response(raw_response)
            description = parsed.description
            parsed_warnings.extend(parsed.warnings)
        except DescriptionValidationError as exc:
            errors.append(str(exc))
        except Exception as exc:  # pragma: no cover - covered in runtime rather than unit tests
            errors.append(f"{type(exc).__name__}: {exc}")

        if description is None:
            status = "fail"
            num_fail += 1
        else:
            if not bool(video_metadata.get("available", False)):
                parsed_warnings.append("video_metadata_unavailable")
            status = "warning" if len(parsed_warnings) > 0 else "ok"
            if status == "ok":
                num_ok += 1
            else:
                num_warning += 1
            catalog_entries.append(
                build_kimodo_catalog_entry(
                    record=record,
                    description=description,
                    model_id=model_id,
                    seed=seed,
                    num_samples=num_samples,
                )
            )

        quality_report = {
            "clip_id": record.clip_id,
            "status": status,
            "warnings": list(dict.fromkeys(parsed_warnings)),
            "errors": errors,
            "response_char_count": int(len(raw_response)),
            "video_metadata_available": bool(video_metadata.get("available", False)),
            "elapsed_sec": float(time.perf_counter() - started_clip),
        }
        description_artifact = {
            "status": status,
            "description": None if description is None else description.to_dict(),
            "errors": errors,
        }
        artifacts = write_clip_artifacts(
            output_root=output_root,
            record=record,
            description_artifact=description_artifact,
            raw_response=raw_response,
            prompt_context=prompt_context,
            quality_report=quality_report,
        )
        manifest_entries.append(
            build_manifest_entry(
                record=record,
                video_metadata=video_metadata,
                status=status,
                model_id=model_id,
                generation_config=generation_config.to_dict(),
                description=description,
                artifacts=artifacts,
            )
        )

    elapsed_sec = float(time.perf_counter() - started_at)
    summary = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "output_dir": str(output_root.resolve()),
        "domains": [str(domain) for domain in domains],
        "num_total": int(len(records)),
        "num_ok": int(num_ok),
        "num_warning": int(num_warning),
        "num_fail": int(num_fail),
        "model_id": model_id,
        "generation_config": generation_config.to_dict(),
        "seed": int(seed),
        "num_samples": int(num_samples),
        "elapsed_sec": elapsed_sec,
    }
    written_paths = write_root_outputs(
        output_dir=output_root,
        manifest_entries=manifest_entries,
        catalog_entries=catalog_entries,
        summary=summary,
        catalog_output_path=catalog_output_path,
    )
    summary.update(
        {
            "video_description_manifest_path": written_paths["manifest_path"],
            "video_description_summary_path": written_paths["summary_path"],
            "kimodo_prompt_catalog_path": written_paths["catalog_path"],
        }
    )
    write_json(written_paths["summary_path"], summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(prog="python -m robot_emotions_vlm")
    subparsers = parser.add_subparsers(dest="command")

    describe_parser = subparsers.add_parser("describe-videos")
    describe_parser.add_argument("--dataset-root", default="data/RobotEmotions")
    describe_parser.add_argument("--domains", nargs="+", default=["10ms", "30ms"])
    describe_parser.add_argument("--clip-id", action="append", dest="clip_ids")
    describe_parser.add_argument("--output-dir", default="output/robot_emotions_qwen")
    describe_parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    describe_parser.add_argument("--local-files-only", action="store_true")
    describe_parser.add_argument("--device-map", default="auto")
    describe_parser.add_argument("--torch-dtype", default="auto")
    describe_parser.add_argument("--attn-implementation", default="sdpa")
    describe_parser.add_argument("--num-video-frames", type=int, default=32)
    describe_parser.add_argument("--max-new-tokens", type=int, default=384)
    describe_parser.add_argument("--temperature", type=float, default=0.2)
    describe_parser.add_argument("--top-p", type=float, default=0.9)
    describe_parser.add_argument(
        "--system-prompt-path",
        default=str(DEFAULT_SYSTEM_PROMPT_PATH),
    )
    describe_parser.add_argument(
        "--user-prompt-path",
        default=str(DEFAULT_USER_PROMPT_PATH),
    )
    describe_parser.add_argument("--catalog-output-path")
    describe_parser.add_argument("--seed", type=int, default=123)
    describe_parser.add_argument("--num-samples", type=int, default=1)

    describe_windows_parser = subparsers.add_parser("describe-windows")
    describe_windows_parser.add_argument("--pose3d-manifest-path", required=True)
    describe_windows_parser.add_argument("--clip-id", action="append", dest="clip_ids")
    describe_windows_parser.add_argument("--output-dir", default="output/robot_emotions_qwen_windows")
    describe_windows_parser.add_argument("--window-sec", type=float, default=5.0)
    describe_windows_parser.add_argument("--window-hop-sec", type=float, default=2.5)
    describe_windows_parser.add_argument("--max-windows-per-clip", type=int)
    describe_windows_parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    describe_windows_parser.add_argument("--local-files-only", action="store_true")
    describe_windows_parser.add_argument("--device-map", default="auto")
    describe_windows_parser.add_argument("--torch-dtype", default="auto")
    describe_windows_parser.add_argument("--attn-implementation", default="sdpa")
    describe_windows_parser.add_argument("--num-video-frames", type=int, default=48)
    describe_windows_parser.add_argument("--max-new-tokens", type=int, default=384)
    describe_windows_parser.add_argument("--temperature", type=float, default=0.2)
    describe_windows_parser.add_argument("--top-p", type=float, default=0.9)
    describe_windows_parser.add_argument(
        "--system-prompt-path",
        default=str(DEFAULT_SYSTEM_PROMPT_PATH),
    )
    describe_windows_parser.add_argument(
        "--user-prompt-path",
        default=str(DEFAULT_USER_PROMPT_PATH),
    )
    describe_windows_parser.add_argument("--catalog-output-path")
    describe_windows_parser.add_argument("--seed", type=int, default=123)
    describe_windows_parser.add_argument("--num-samples", type=int, default=1)

    generate_parser = subparsers.add_parser("generate-kimodo")
    generate_parser.add_argument("--catalog-path", required=True)
    generate_parser.add_argument("--output-dir", default="output/robot_emotions_kimodo")
    generate_parser.add_argument("--clip-id", action="append", dest="clip_ids")
    generate_parser.add_argument("--prompt-id", action="append", dest="prompt_ids")
    generate_parser.add_argument("--window-id", action="append", dest="window_ids")
    generate_parser.add_argument("--model", default=DEFAULT_KIMODO_GENERATION_MODEL)
    generate_parser.add_argument("--duration-sec", type=float, default=5.0)
    generate_parser.add_argument("--diffusion-steps", type=int, default=100)
    generate_parser.add_argument("--seed", type=int)
    generate_parser.add_argument("--num-samples", type=int)
    generate_parser.add_argument("--no-postprocess", action="store_true")
    generate_parser.add_argument("--bvh", action="store_true")
    generate_parser.add_argument("--cfg-type", choices=("nocfg", "regular", "separated"))
    generate_parser.add_argument("--cfg-weight", type=float, nargs="*")

    anchor_parser = subparsers.add_parser("build-anchor-catalog")
    anchor_parser.add_argument("--pose3d-manifest-path", required=True)
    anchor_parser.add_argument("--qwen-window-catalog-path", required=True)
    anchor_parser.add_argument("--output-dir", required=True)
    anchor_parser.add_argument("--model", default=DEFAULT_KIMODO_GENERATION_MODEL)
    anchor_parser.add_argument("--clip-id", action="append", dest="clip_ids")
    anchor_parser.add_argument(
        "--hand-keyframes",
        type=int,
        default=0,
        help=(
            "Number of sparse keyframes for left-hand and right-hand end-effector constraints. "
            "0 (default) = root2d only.  Requires the kimodo conda environment."
        ),
    )

    kimodo_imu_parser = subparsers.add_parser(
        "export-kimodo-virtual-imu",
        help=(
            "Convert a Kimodo generation manifest to synthetic virtual IMU signals. "
            "Applies the same temporal smoothing as the real-video pipeline "
            "(metric normalisation + root trajectory estimation) before IK + IMUSim."
        ),
    )
    kimodo_imu_parser.add_argument(
        "--kimodo-manifest", required=True,
        help="JSONL manifest produced by generate-kimodo (or batch_kimodo_pose3d.py).",
    )
    kimodo_imu_parser.add_argument(
        "--output-dir", required=True,
        help="Output directory. Per-clip artifacts land under <output-dir>/<prompt_id>/.",
    )
    kimodo_imu_parser.add_argument("--clip-id", action="append", dest="clip_ids")
    kimodo_imu_parser.add_argument("--sensor-layout-path", default=None)
    kimodo_imu_parser.add_argument("--imu-acc-noise-std-m-s2", type=float, default=None)
    kimodo_imu_parser.add_argument("--imu-gyro-noise-std-rad-s", type=float, default=None)
    kimodo_imu_parser.add_argument("--imu-random-seed", type=int, default=0)
    kimodo_imu_parser.add_argument(
        "--export-bvh", action="store_true",
        help="Also export a BVH file for each pose3d clip.",
    )
    kimodo_imu_parser.add_argument(
        "--no-skip-existing", action="store_true",
        help="Reprocess clips even if virtual_imu.npz already exists.",
    )

    mixed_parser = subparsers.add_parser(
        "export-mixed-virtual-imu",
        help=(
            "Merge a real virtual_imu_manifest and a synthetic virtual_imu_manifest "
            "into a single mixed_virtual_imu_manifest.jsonl for combined experiments."
        ),
    )
    mixed_parser.add_argument(
        "--real-manifest", required=True,
        help="JSONL manifest from export-virtual-imu (real video pipeline).",
    )
    mixed_parser.add_argument(
        "--synthetic-manifest", required=True,
        help="JSONL manifest from export-kimodo-virtual-imu.",
    )
    mixed_parser.add_argument(
        "--output-dir", required=True,
        help="Directory where mixed_virtual_imu_manifest.jsonl is written.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by ``python -m robot_emotions_vlm``."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "describe-videos":
        if args.command == "describe-windows":
            summary = describe_windows(
                pose3d_manifest_path=args.pose3d_manifest_path,
                output_dir=args.output_dir,
                clip_ids=args.clip_ids,
                window_sec=args.window_sec,
                window_hop_sec=args.window_hop_sec,
                max_windows_per_clip=args.max_windows_per_clip,
                model_id=args.model_id,
                local_files_only=args.local_files_only,
                device_map=args.device_map,
                torch_dtype=args.torch_dtype,
                attn_implementation=args.attn_implementation,
                num_video_frames=args.num_video_frames,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                system_prompt_path=args.system_prompt_path,
                user_prompt_path=args.user_prompt_path,
                catalog_output_path=args.catalog_output_path,
                seed=args.seed,
                num_samples=args.num_samples,
            )
            print(json.dumps(summary, indent=2, ensure_ascii=True))
            return 0
        if args.command == "generate-kimodo":
            summary = generate_kimodo_from_catalog(
                catalog_path=args.catalog_path,
                output_dir=args.output_dir,
                clip_ids=args.clip_ids,
                prompt_ids=args.prompt_ids,
                window_ids=args.window_ids,
                model_name=args.model,
                duration_sec=args.duration_sec,
                diffusion_steps=args.diffusion_steps,
                seed=args.seed,
                num_samples=args.num_samples,
                no_postprocess=args.no_postprocess,
                bvh=args.bvh,
                cfg_type=args.cfg_type,
                cfg_weight=args.cfg_weight,
            )
            print(json.dumps(summary, indent=2, ensure_ascii=True))
            return 0
        if args.command == "build-anchor-catalog":
            summary = build_anchor_catalog(
                pose3d_manifest_path=args.pose3d_manifest_path,
                qwen_window_catalog_path=args.qwen_window_catalog_path,
                output_dir=args.output_dir,
                model_name=args.model,
                clip_ids=args.clip_ids,
                hand_keyframes=args.hand_keyframes,
            )
            print(json.dumps(summary, indent=2, ensure_ascii=True))
            return 0
        if args.command == "export-kimodo-virtual-imu":
            summary = export_kimodo_virtual_imu(
                kimodo_manifest_path=args.kimodo_manifest,
                output_dir=args.output_dir,
                sensor_layout_path=args.sensor_layout_path,
                imu_acc_noise_std_m_s2=args.imu_acc_noise_std_m_s2,
                imu_gyro_noise_std_rad_s=args.imu_gyro_noise_std_rad_s,
                imu_random_seed=args.imu_random_seed,
                export_bvh=bool(args.export_bvh),
                skip_existing=not bool(args.no_skip_existing),
                clip_ids=args.clip_ids,
            )
            print(json.dumps(summary, indent=2, ensure_ascii=True))
            return 0
        if args.command == "export-mixed-virtual-imu":
            summary = export_mixed_virtual_imu(
                real_manifest_path=args.real_manifest,
                synthetic_manifest_path=args.synthetic_manifest,
                output_dir=args.output_dir,
            )
            print(json.dumps(summary, indent=2, ensure_ascii=True))
            return 0
        parser.print_help()
        return 1

    summary = describe_videos(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        domains=args.domains,
        clip_ids=args.clip_ids,
        model_id=args.model_id,
        local_files_only=args.local_files_only,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        num_video_frames=args.num_video_frames,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        system_prompt_path=args.system_prompt_path,
        user_prompt_path=args.user_prompt_path,
        catalog_output_path=args.catalog_output_path,
        seed=args.seed,
        num_samples=args.num_samples,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0
