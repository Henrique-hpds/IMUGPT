"""Window-level Qwen description pipeline for RobotEmotions."""

from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import time
from typing import Any, Sequence

from .dataset import read_video_metadata
from .export import write_json, write_jsonl, write_text
from .metadata import get_protocol_info
from .prompts import (
    DEFAULT_SYSTEM_PROMPT_PATH,
    DEFAULT_USER_PROMPT_PATH,
    build_prompt_placeholders_from_metadata,
    render_prompts_from_placeholders,
)
from .qwen_backend import QwenGenerationConfig, QwenVideoBackend
from .schemas import DescriptionValidationError, VideoDescription, parse_model_response
from .windowing import (
    WindowSpec,
    build_windows,
    load_pose_manifest_entries,
    load_pose_sequence3d,
    resolve_source_times,
    select_pose_entries,
)


def describe_windows(
    *,
    pose3d_manifest_path: str | Path,
    output_dir: str | Path,
    clip_ids: Sequence[str] | None = None,
    window_sec: float = 5.0,
    window_hop_sec: float = 2.5,
    max_windows_per_clip: int | None = None,
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
    """Run Qwen on real video windows and export a window-level Kimodo catalog."""

    if float(window_sec) <= 0.0:
        raise ValueError("window_sec must be positive.")
    if float(window_hop_sec) <= 0.0:
        raise ValueError("window_hop_sec must be positive.")
    if max_windows_per_clip is not None and int(max_windows_per_clip) <= 0:
        raise ValueError("max_windows_per_clip must be positive when provided.")
    if int(num_samples) <= 0:
        raise ValueError("num_samples must be >= 1.")

    started_at = time.perf_counter()
    pose_entries = select_pose_entries(load_pose_manifest_entries(pose3d_manifest_path), clip_ids=clip_ids)
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

    for pose_entry in pose_entries:
        source_video_path = pose_entry.source_video_path
        if source_video_path is None:
            raise ValueError(f"Pose manifest entry {pose_entry.clip_id!r} is missing source.video_path.")
        if pose_entry.domain is None or pose_entry.user_id is None or pose_entry.tag_number is None:
            raise ValueError(
                f"Pose manifest entry {pose_entry.clip_id!r} must expose domain, user_id, and tag_number."
            )

        sequence = load_pose_sequence3d(pose_entry.pose3d_npz_path)
        source_times = resolve_source_times(sequence)
        windows = build_windows(
            clip_id=pose_entry.clip_id,
            source_times=source_times,
            window_sec=float(window_sec),
            window_hop_sec=float(window_hop_sec),
            max_windows_per_clip=max_windows_per_clip,
        )

        for window in windows:
            raw_response = ""
            description: VideoDescription | None = None
            parsed_warnings: list[str] = []
            errors: list[str] = []

            window_dir = output_root / window.prompt_id
            window_dir.mkdir(parents=True, exist_ok=True)
            window_video_path = window_dir / "window.mp4"
            _write_video_window_subclip(
                source_video_path=source_video_path,
                output_path=window_video_path,
                start_sec=float(window.start_sec),
                end_sec=float(window.end_sec),
            )
            video_metadata = read_video_metadata(window_video_path)
            window_metadata = window.to_dict()
            window_metadata["reference_clip_id"] = pose_entry.clip_id
            window_metadata["source_video_path"] = str(Path(source_video_path).resolve())

            placeholders = build_prompt_placeholders_from_metadata(
                clip_id=pose_entry.clip_id,
                domain=str(pose_entry.domain),
                user_id=int(pose_entry.user_id),
                tag_number=int(pose_entry.tag_number),
                take_id=pose_entry.take_id,
                source_rel_dir=pose_entry.source_rel_dir,
                video_path=window_video_path,
                labels=dict(pose_entry.labels),
                protocol=(
                    get_protocol_info(pose_entry.domain, pose_entry.tag_number)
                ),
                video_metadata=video_metadata,
                analysis_scope="window",
                window_metadata=window_metadata,
            )
            prompt_bundle = render_prompts_from_placeholders(
                placeholders,
                system_prompt_path=system_prompt_path,
                user_prompt_path=user_prompt_path,
            )
            prompt_context = {
                "reference_clip_id": pose_entry.clip_id,
                "window": window_metadata,
                "video_metadata": video_metadata,
                "model_id": model_id,
                "generation_config": generation_config.to_dict(),
                "prompt_bundle": prompt_bundle,
            }

            started_window = time.perf_counter()
            try:
                raw_response = backend.describe_video(
                    window_video_path,
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
                    _build_window_catalog_entry(
                        prompt_id=window.prompt_id,
                        reference_clip_id=pose_entry.clip_id,
                        window=window,
                        labels=pose_entry.labels,
                        description=description,
                        model_id=model_id,
                        seed=seed,
                        num_samples=int(num_samples),
                    )
                )

            description_artifact = {
                "status": status,
                "description": None if description is None else description.to_dict(),
                "errors": errors,
            }
            quality_report = {
                "prompt_id": window.prompt_id,
                "reference_clip_id": pose_entry.clip_id,
                "status": status,
                "warnings": list(dict.fromkeys(parsed_warnings)),
                "errors": errors,
                "response_char_count": int(len(raw_response)),
                "video_metadata_available": bool(video_metadata.get("available", False)),
                "elapsed_sec": float(time.perf_counter() - started_window),
            }
            artifacts = _write_window_artifacts(
                window_dir=window_dir,
                description_artifact=description_artifact,
                raw_response=raw_response,
                prompt_context=prompt_context,
                quality_report=quality_report,
                window_video_path=window_video_path,
            )
            manifest_entries.append(
                _build_window_manifest_entry(
                    pose_entry=pose_entry,
                    window=window,
                    video_metadata=video_metadata,
                    status=status,
                    model_id=model_id,
                    generation_config=generation_config.to_dict(),
                    description=description,
                    artifacts=artifacts,
                )
            )

    elapsed_sec = float(time.perf_counter() - started_at)
    manifest_path = output_root / "window_description_manifest.jsonl"
    summary_path = output_root / "window_description_summary.json"
    if catalog_output_path is None:
        catalog_path = output_root / "kimodo_window_prompt_catalog.jsonl"
    else:
        catalog_path = Path(catalog_output_path)

    write_jsonl(manifest_path, manifest_entries)
    write_jsonl(catalog_path, catalog_entries)
    summary = {
        "pose3d_manifest_path": str(Path(pose3d_manifest_path).resolve()),
        "output_dir": str(output_root.resolve()),
        "window_sec": float(window_sec),
        "window_hop_sec": float(window_hop_sec),
        "max_windows_per_clip": None if max_windows_per_clip is None else int(max_windows_per_clip),
        "num_total_windows": int(len(manifest_entries)),
        "num_ok": int(num_ok),
        "num_warning": int(num_warning),
        "num_fail": int(num_fail),
        "model_id": model_id,
        "generation_config": generation_config.to_dict(),
        "seed": int(seed),
        "num_samples": int(num_samples),
        "elapsed_sec": elapsed_sec,
        "window_description_manifest_path": str(manifest_path.resolve()),
        "window_description_summary_path": str(summary_path.resolve()),
        "kimodo_window_prompt_catalog_path": str(catalog_path.resolve()),
    }
    write_json(summary_path, summary)
    return summary


def _build_window_catalog_entry(
    *,
    prompt_id: str,
    reference_clip_id: str,
    window: WindowSpec,
    labels: dict[str, Any],
    description: VideoDescription,
    model_id: str,
    seed: int,
    num_samples: int,
) -> dict[str, Any]:
    return {
        "prompt_id": prompt_id,
        "window_id": prompt_id,
        "prompt_text": description.prompt_text,
        "labels": dict(labels),
        "seed": int(seed),
        "num_samples": int(num_samples),
        "reference_clip_id": reference_clip_id,
        "duration_hint_sec": float(window.duration_sec),
        "window": window.to_dict(),
        "source_metadata": {
            "dataset": "RobotEmotions",
            "model_id": model_id,
            "body_parts": description.body_parts.to_dict(),
        },
    }


def _write_window_artifacts(
    *,
    window_dir: Path,
    description_artifact: dict[str, Any],
    raw_response: str,
    prompt_context: dict[str, Any],
    quality_report: dict[str, Any],
    window_video_path: Path,
) -> dict[str, str]:
    description_path = window_dir / "description.json"
    raw_response_path = window_dir / "raw_response.txt"
    prompt_context_path = window_dir / "prompt_context.json"
    quality_report_path = window_dir / "quality_report.json"

    write_json(description_path, description_artifact)
    write_text(raw_response_path, raw_response)
    write_json(prompt_context_path, prompt_context)
    write_json(quality_report_path, quality_report)

    return {
        "description_json_path": str(description_path.resolve()),
        "raw_response_txt_path": str(raw_response_path.resolve()),
        "prompt_context_json_path": str(prompt_context_path.resolve()),
        "quality_report_json_path": str(quality_report_path.resolve()),
        "window_video_path": str(window_video_path.resolve()),
    }


def _build_window_manifest_entry(
    *,
    pose_entry: Any,
    window: WindowSpec,
    video_metadata: dict[str, Any],
    status: str,
    model_id: str,
    generation_config: dict[str, Any],
    description: VideoDescription | None,
    artifacts: dict[str, str],
) -> dict[str, Any]:
    return {
        "prompt_id": window.prompt_id,
        "window_id": window.prompt_id,
        "reference_clip_id": pose_entry.clip_id,
        "dataset": "RobotEmotions",
        "domain": pose_entry.domain,
        "user_id": pose_entry.user_id,
        "tag_number": pose_entry.tag_number,
        "take_id": pose_entry.take_id,
        "labels": dict(pose_entry.labels),
        "source": {
            "video_path": pose_entry.source_video_path,
            "source_rel_dir": pose_entry.source_rel_dir,
            "pose3d_npz_path": str(Path(pose_entry.pose3d_npz_path).resolve()),
        },
        "window": window.to_dict(),
        "video": dict(video_metadata),
        "status": status,
        "model_id": model_id,
        "generation_config": dict(generation_config),
        "description": None if description is None else description.to_dict(),
        "artifacts": dict(artifacts),
    }


def _write_video_window_subclip(
    *,
    source_video_path: str | Path,
    output_path: str | Path,
    start_sec: float,
    end_sec: float,
) -> None:
    """Materialize a real MP4 subclip for the requested window."""

    if float(end_sec) <= float(start_sec):
        raise ValueError(f"Invalid window range: start_sec={start_sec}, end_sec={end_sec}")

    try:
        import av
    except ImportError as exc:  # pragma: no cover - environment validation covers this
        raise RuntimeError("PyAV is required to materialize window subclips.") from exc

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded_frames = 0

    with av.open(str(Path(source_video_path).resolve())) as input_container:
        input_stream = next((stream for stream in input_container.streams if stream.type == "video"), None)
        if input_stream is None:
            raise ValueError(f"No video stream found in {source_video_path}")
        if input_stream.average_rate is None:
            raise ValueError(f"Video stream {source_video_path} does not expose average_rate")
        average_rate = input_stream.average_rate
        if hasattr(average_rate, "numerator") and hasattr(average_rate, "denominator"):
            output_rate = Fraction(int(average_rate.numerator), int(average_rate.denominator))
        else:
            output_rate = Fraction(str(float(average_rate))).limit_denominator(1000)
        output_time_base = Fraction(output_rate.denominator, output_rate.numerator)
        width = int(input_stream.codec_context.width)
        height = int(input_stream.codec_context.height)

        with av.open(str(target), mode="w") as output_container:
            output_stream = output_container.add_stream("libx264", rate=output_rate)
            output_stream.width = width
            output_stream.height = height
            output_stream.pix_fmt = "yuv420p"

            for frame in input_container.decode(input_stream):
                if frame.time is None:
                    raise ValueError(f"Frame timestamps are required to cut {source_video_path}")
                frame_time = float(frame.time)
                if frame_time < float(start_sec):
                    continue
                if frame_time >= float(end_sec):
                    break
                formatted = frame.reformat(width=width, height=height, format="yuv420p")
                formatted.pts = encoded_frames
                formatted.time_base = output_time_base
                for packet in output_stream.encode(formatted):
                    output_container.mux(packet)
                encoded_frames += 1

            for packet in output_stream.encode(None):
                output_container.mux(packet)

    if encoded_frames <= 0:
        raise ValueError(
            f"Window [{start_sec}, {end_sec}) did not produce frames from {source_video_path}"
        )
