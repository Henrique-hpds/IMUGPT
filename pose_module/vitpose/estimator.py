"""ViTPose estimator backend and cross-environment launcher."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pose_module.export.debug_video import resolve_debug_overlay_path
from pose_module.interfaces import Pose2DJob, Pose2DResult
from pose_module.io.cache import load_json_file, tail_text, write_json_file
from pose_module.io.video_loader import select_frame_indices


def run_backend_job(
    *,
    job: Pose2DJob,
    env_name: str,
    output_dir: str | Path,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    job_json_path = output_dir / "backend_job.json"
    result_json_path = output_dir / "backend_result.json"
    write_json_file(job.to_dict(), job_json_path)

    repo_root = Path(__file__).resolve().parents[2]
    launcher, probe_diagnostics = _select_backend_launcher(str(env_name), cwd=repo_root)
    if launcher is None:
        backend_run = {
            "status": "fail",
            "effective_fps": None,
            "selected_frame_indices": [],
            "artifacts": {
                "raw_prediction_json_path": str(job.raw_prediction_path.resolve()),
                "debug_overlay_path": str(job.debug_overlay_path.resolve()) if job.save_debug else None,
            },
            "quality_report": {},
            "backend": {
                "launcher": None,
                "probe_diagnostics": probe_diagnostics,
            },
            "error": "No Python launcher with mmpose/mmdet/mmpretrain available. "
            "Use --env-name openmmlab or install OpenMMLab packages in the active interpreter.",
            "env_name": str(env_name),
            "returncode": 1,
        }
        write_json_file(backend_run, output_dir / "backend_run.json")
        return backend_run

    command = list(launcher["prefix"]) + [
        "-m",
        "pose_module.vitpose.estimator",
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
        result = Pose2DResult.from_dict(load_json_file(result_json_path))
    else:
        result = Pose2DResult(
            status="fail",
            effective_fps=None,
            selected_frame_indices=[],
            artifacts={},
            quality_report={},
            backend={},
            error="backend_result_json_missing",
        )

    backend_run = result.to_dict()
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
        backend_run["error"] = "backend_process_failed"

    write_json_file(backend_run, output_dir / "backend_run.json")
    return backend_run


def _select_backend_launcher(env_name: str, *, cwd: Path) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    diagnostics: List[Dict[str, Any]] = []
    for candidate in _build_launcher_candidates(env_name):
        probe_command = list(candidate["prefix"]) + [
            "-c",
            "import mmpose, mmdet, mmpretrain; print('ok')",
        ]
        completed = subprocess.run(
            probe_command,
            cwd=str(cwd.resolve()),
            text=True,
            capture_output=True,
            check=False,
        )
        diagnostics.append(
            {
                "name": str(candidate["name"]),
                "python": str(candidate["python"]),
                "command": probe_command,
                "returncode": int(completed.returncode),
                "stdout_tail": tail_text(completed.stdout, max_chars=2000) if completed.stdout else "",
                "stderr_tail": tail_text(completed.stderr, max_chars=2000) if completed.stderr else "",
            }
        )
        if completed.returncode == 0:
            return candidate, diagnostics
    return None, diagnostics


def _build_launcher_candidates(env_name: str) -> List[Dict[str, Any]]:
    normalized = str(env_name).strip()
    lowered = normalized.lower()

    candidates: List[Dict[str, Any]] = []
    if lowered not in {"", "auto", "current"}:
        conda_python = _resolve_conda_env_python(normalized)
        if conda_python is not None:
            candidates.append(
                {
                    "name": "conda_env_python",
                    "python": str(conda_python),
                    "prefix": [str(conda_python)],
                }
            )
        candidates.append(
            {
                "name": "conda_env",
                "python": f"conda:{normalized}",
                "prefix": ["conda", "run", "-n", normalized, "python"],
            }
        )

    candidates.append(
        {
            "name": "current_python",
            "python": str(Path(sys.executable).resolve()),
            "prefix": [str(Path(sys.executable).resolve())],
        }
    )

    if lowered in {"", "auto"}:
        openmmlab_python = _resolve_conda_env_python("openmmlab")
        if openmmlab_python is not None:
            candidates.append(
                {
                    "name": "conda_env_python",
                    "python": str(openmmlab_python),
                    "prefix": [str(openmmlab_python)],
                }
            )
        candidates.append(
            {
                "name": "conda_env",
                "python": "conda:openmmlab",
                "prefix": ["conda", "run", "-n", "openmmlab", "python"],
            }
        )

    return candidates


def _resolve_conda_env_python(env_name: str) -> Optional[Path]:
    try:
        completed = subprocess.run(
            ["conda", "env", "list", "--json"],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return None

    if completed.returncode != 0:
        return None

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None

    env_paths = payload.get("envs", [])
    if not isinstance(env_paths, list):
        return None

    normalized = str(env_name).strip()
    for raw_env_path in env_paths:
        env_path = Path(str(raw_env_path))
        if env_path.name != normalized:
            continue
        python_path = env_path / "bin" / "python"
        if python_path.exists():
            return python_path.resolve()
    return None


def run_pose2d_backend(job: Pose2DJob) -> Pose2DResult:
    from mmpose import __version__ as mmpose_version
    from mmpose.apis import MMPoseInferencer
    import torch

    output_dir = Path(job.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_prediction_dir = output_dir / "_backend_predictions"
    raw_prediction_dir.mkdir(parents=True, exist_ok=True)

    debug_overlay_path = resolve_debug_overlay_path(
        output_dir,
        filename=str(job.debug_overlay_filename),
        enabled=bool(job.save_debug),
    )
    if debug_overlay_path is not None:
        debug_overlay_path.parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(str(job.device_preference))
    inferencer = MMPoseInferencer(
        pose2d=str(job.model_alias),
        det_cat_ids=list(job.detector_category_ids),
        device=device,
        show_progress=False,
    )
    generator = inferencer(
        str(Path(job.video_path).resolve()),
        pred_out_dir=str(raw_prediction_dir.resolve()),
        vis_out_dir="" if debug_overlay_path is None else str(debug_overlay_path.resolve()),
    )
    for _ in generator:
        pass

    generated_raw_prediction_path = raw_prediction_dir / (Path(job.video_path).stem + ".json")
    if not generated_raw_prediction_path.exists():
        raise FileNotFoundError(
            "MMPose backend did not create the raw prediction JSON file at "
            + str(generated_raw_prediction_path)
        )

    final_raw_prediction_path = job.raw_prediction_path
    final_raw_prediction_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(generated_raw_prediction_path), str(final_raw_prediction_path))
    if raw_prediction_dir.exists():
        shutil.rmtree(raw_prediction_dir, ignore_errors=True)

    raw_predictions = json.loads(final_raw_prediction_path.read_text(encoding="utf-8"))
    frames_total = int(len(raw_predictions))
    frames_with_detections = int(
        sum(1 for frame_payload in raw_predictions if len(frame_payload.get("instances", [])) > 0)
    )

    frame_count_for_sampling = job.video_num_frames if job.video_num_frames is not None else frames_total
    selected_frame_indices, effective_fps, _ = select_frame_indices(
        int(frame_count_for_sampling),
        job.video_fps,
        int(job.fps_target),
    )
    warnings = []
    if job.video_num_frames is not None and int(job.video_num_frames) != frames_total:
        warnings.append("video_num_frames_differs_from_backend_predictions")

    return Pose2DResult(
        status="ok",
        effective_fps=effective_fps,
        selected_frame_indices=selected_frame_indices.astype(int).tolist(),
        artifacts={
            "raw_prediction_json_path": str(final_raw_prediction_path.resolve()),
            "debug_overlay_path": (
                None
                if debug_overlay_path is None or not debug_overlay_path.exists()
                else str(debug_overlay_path.resolve())
            ),
        },
        quality_report={
            "fps_original": job.video_fps,
            "effective_fps": effective_fps,
            "frames_total": frames_total,
            "frames_selected": int(len(selected_frame_indices)),
            "frames_with_detections": frames_with_detections,
            "warnings": warnings,
        },
        backend={
            "model_alias": str(job.model_alias),
            "device": str(device),
            "device_preference": str(job.device_preference),
            "detector_category_ids": [int(value) for value in job.detector_category_ids],
            "mmpose_version": str(mmpose_version),
            "torch_cuda_available": bool(torch.cuda.is_available()),
        },
        error=None,
    )


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Run ViTPose pose2d backend in openmmlab.")
    parser.add_argument("--job-json", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    args = parser.parse_args(argv)

    payload = json.loads(args.job_json.read_text(encoding="utf-8"))
    job = Pose2DJob.from_dict(payload)

    try:
        result = run_pose2d_backend(job)
    except Exception as exc:
        result = Pose2DResult(
            status="fail",
            effective_fps=None,
            selected_frame_indices=[],
            artifacts={
                "raw_prediction_json_path": str(job.raw_prediction_path.resolve()),
                "debug_overlay_path": str(job.debug_overlay_path.resolve()) if job.save_debug else None,
            },
            quality_report={
                "fps_original": job.video_fps,
                "effective_fps": None,
                "frames_total": 0,
                "frames_selected": 0,
                "frames_with_detections": 0,
                "warnings": [],
            },
            backend={
                "model_alias": str(job.model_alias),
                "device_preference": str(job.device_preference),
            },
            error=str(exc),
        )

    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return 0 if result.status == "ok" else 1


def _resolve_device(device_preference: str) -> str:
    import torch

    preference = str(device_preference).strip().lower()
    if preference not in {"", "auto"}:
        return str(device_preference)
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


if __name__ == "__main__":
    raise SystemExit(main())
