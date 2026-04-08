"""CLI for RobotEmotions extraction and pose export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .extractor import RobotEmotionsExtractor
from .pose2d import run_robot_emotions_pose2d
from .pose3d import run_robot_emotions_pose3d
from .prompt_exports import build_robot_emotions_prompt_catalog, run_robot_emotions_prompt_pose3d
from .virtual_imu import run_robot_emotions_virtual_imu


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "scan":
        extractor = RobotEmotionsExtractor(args.dataset_root, domains=tuple(args.domains))
        records = extractor.select_records(
            clip_ids=None if args.clip_id is None else list(args.clip_id),
        )
        summary = {
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "domains": list(args.domains),
            "num_records": int(len(records)),
            "sample_clip_ids": [record.clip_id for record in records[:5]],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return 0

    if args.command == "export-imu":
        extractor = RobotEmotionsExtractor(args.dataset_root, domains=tuple(args.domains))
        summary = extractor.export(
            args.output_dir,
            clip_ids=None if args.clip_id is None else list(args.clip_id),
        )
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return 0

    if args.command == "export-pose2d":
        summary = run_robot_emotions_pose2d(
            dataset_root=str(args.dataset_root),
            output_dir=str(args.output_dir),
            fps_target=int(args.fps_target),
            clip_ids=None if args.clip_id is None else list(args.clip_id),
            save_debug=bool(not args.no_debug),
            env_name=str(args.env_name),
            domains=tuple(args.domains),
        )
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return 0

    if args.command == "export-pose3d":
        save_debug_2d = bool(not args.no_debug) if args.debug_2d is None else bool(args.debug_2d)
        save_debug_3d = bool(not args.no_debug) if args.debug_3d is None else bool(args.debug_3d)
        summary = run_robot_emotions_pose3d(
            dataset_root=str(args.dataset_root),
            output_dir=str(args.output_dir),
            fps_target=int(args.fps_target),
            clip_ids=None if args.clip_id is None else list(args.clip_id),
            save_debug=bool(not args.no_debug),
            save_debug_2d=bool(save_debug_2d),
            save_debug_3d=bool(save_debug_3d),
            env_name=str(args.env_name),
            motionbert_env_name=(
                None if args.motionbert_env_name in (None, "") else str(args.motionbert_env_name)
            ),
            motionbert_window_size=int(args.motionbert_window_size),
            motionbert_window_overlap=float(args.motionbert_window_overlap),
            include_motionbert_confidence=bool(not args.no_motionbert_confidence),
            motionbert_device=str(args.motionbert_device),
            allow_motionbert_fallback_backend=bool(args.allow_motionbert_fallback_backend),
            domains=tuple(args.domains),
        )
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return 0

    if args.command == "build-prompt-catalog":
        summary = build_robot_emotions_prompt_catalog(
            dataset_root=str(args.dataset_root),
            output_path=str(args.output_path),
            real_pose3d_manifest_path=(
                None
                if args.real_pose3d_manifest_path in (None, "")
                else str(args.real_pose3d_manifest_path)
            ),
            domains=tuple(args.domains),
            default_num_samples=int(args.num_samples),
            default_seed=int(args.seed),
            default_fps=float(args.fps),
        )
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return 0

    if args.command == "export-prompt-pose3d":
        summary = run_robot_emotions_prompt_pose3d(
            prompt_catalog_path=str(args.prompt_catalog),
            output_dir=str(args.output_dir),
            export_bvh=bool(not args.no_bvh),
        )
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return 0

    if args.command == "export-virtual-imu":
        save_debug_2d = bool(not args.no_debug) if args.debug_2d is None else bool(args.debug_2d)
        save_debug_3d = bool(not args.no_debug) if args.debug_3d is None else bool(args.debug_3d)
        summary = run_robot_emotions_virtual_imu(
            dataset_root=str(args.dataset_root),
            output_dir=str(args.output_dir),
            fps_target=int(args.fps_target),
            clip_ids=None if args.clip_id is None else list(args.clip_id),
            save_debug=bool(not args.no_debug),
            save_debug_2d=bool(save_debug_2d),
            save_debug_3d=bool(save_debug_3d),
            env_name=str(args.env_name),
            motionbert_env_name=(
                None if args.motionbert_env_name in (None, "") else str(args.motionbert_env_name)
            ),
            motionbert_window_size=int(args.motionbert_window_size),
            motionbert_window_overlap=float(args.motionbert_window_overlap),
            include_motionbert_confidence=bool(not args.no_motionbert_confidence),
            motionbert_device=str(args.motionbert_device),
            allow_motionbert_fallback_backend=bool(args.allow_motionbert_fallback_backend),
            sensor_layout_path=(
                None if args.sensor_layout_path in (None, "") else str(args.sensor_layout_path)
            ),
            imu_acc_noise_std_m_s2=args.imu_acc_noise_std_m_s2,
            imu_gyro_noise_std_rad_s=args.imu_gyro_noise_std_rad_s,
            imu_random_seed=int(args.imu_random_seed),
            real_imu_reference_path=(
                None if args.real_imu_reference_path in (None, "") else str(args.real_imu_reference_path)
            ),
            real_imu_label_key=(
                None if args.real_imu_label_key in (None, "") else str(args.real_imu_label_key)
            ),
            real_imu_signal_mode=str(args.real_imu_signal_mode),
            real_imu_percentile_resolution=int(args.real_imu_percentile_resolution),
            real_imu_per_class_calibration=bool(not args.no_real_imu_per_class_calibration),
            estimate_sensor_frame=bool(args.estimate_sensor_frame),
            estimate_sensor_names=(
                None if args.estimate_sensor_names is None else list(args.estimate_sensor_names)
            ),
            domains=tuple(args.domains),
        )
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RobotEmotions extraction and pose export.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Scan RobotEmotions records.")
    _add_shared_dataset_arguments(scan_parser, include_output_dir=False)

    export_imu_parser = subparsers.add_parser("export-imu", help="Export IMU artifacts and manifest.")
    _add_shared_dataset_arguments(export_imu_parser, include_output_dir=True)

    export_pose_parser = subparsers.add_parser(
        "export-pose2d",
        help="Export stage-5.3 pose2d artifacts alongside the extracted IMU artifacts.",
    )
    _add_pose_export_arguments(export_pose_parser, include_motionbert_arguments=False)

    export_pose3d_parser = subparsers.add_parser(
        "export-pose3d",
        help="Export stage-5.8 pose3d artifacts, including BVH, alongside the extracted IMU artifacts.",
    )
    _add_pose_export_arguments(export_pose3d_parser, include_motionbert_arguments=True)

    build_prompt_catalog_parser = subparsers.add_parser(
        "build-prompt-catalog",
        help="Build a RobotEmotions prompt catalog with canonical labels and motion-focused prompt text.",
    )
    _add_shared_dataset_arguments(build_prompt_catalog_parser, include_output_dir=False)
    build_prompt_catalog_parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/prompts/robot_emotions_prompts.jsonl"),
        help="Output JSONL catalog path.",
    )
    build_prompt_catalog_parser.add_argument(
        "--real-pose3d-manifest-path",
        type=Path,
        default=None,
        help=(
            "Optional real-manifest JSONL used to enrich prompts with motion traits. "
            "The file may be a pose3d manifest or a virtual_imu manifest, as long as "
            "each entry exposes artifacts.pose3d_npz_path."
        ),
    )
    build_prompt_catalog_parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Default num_samples value written to each prompt catalog entry.",
    )
    build_prompt_catalog_parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Default seed written to each prompt catalog entry.",
    )
    build_prompt_catalog_parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Default FPS written to each prompt catalog entry.",
    )

    export_prompt_pose3d_parser = subparsers.add_parser(
        "export-prompt-pose3d",
        help="Export synthetic pose3d artifacts from a prompt catalog without using BVH as an intermediate format.",
    )
    export_prompt_pose3d_parser.add_argument(
        "--prompt-catalog",
        type=Path,
        required=True,
        help="JSONL prompt catalog path.",
    )
    export_prompt_pose3d_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for synthetic pose3d exports.",
    )
    export_prompt_pose3d_parser.add_argument(
        "--no-bvh",
        action="store_true",
        help="Disable optional BVH export for synthetic pose3d artifacts.",
    )

    export_virtual_imu_parser = subparsers.add_parser(
        "export-virtual-imu",
        help="Export the full stage-5.10 virtual IMU pipeline alongside the extracted real IMU artifacts.",
    )
    _add_pose_export_arguments(export_virtual_imu_parser, include_motionbert_arguments=True)
    export_virtual_imu_parser.add_argument(
        "--sensor-layout-path",
        type=str,
        default=None,
        help="Optional sensor layout config path for the virtual IMU adapter.",
    )
    export_virtual_imu_parser.add_argument(
        "--imu-acc-noise-std-m-s2",
        type=float,
        default=None,
        help="Optional accelerometer Gaussian noise std in m/s^2.",
    )
    export_virtual_imu_parser.add_argument(
        "--imu-gyro-noise-std-rad-s",
        type=float,
        default=None,
        help="Optional gyroscope Gaussian noise std in rad/s.",
    )
    export_virtual_imu_parser.add_argument(
        "--imu-random-seed",
        type=int,
        default=0,
        help="Random seed used for optional virtual IMU noise.",
    )
    export_virtual_imu_parser.add_argument(
        "--real-imu-reference-path",
        type=str,
        default=None,
        help="Optional NPZ reference with real IMU data used for percentile calibration of the virtual IMU.",
    )
    export_virtual_imu_parser.add_argument(
        "--real-imu-label-key",
        type=str,
        default=None,
        help="Optional label field from the RobotEmotions manifest used for per-class calibration, e.g. action.",
    )
    export_virtual_imu_parser.add_argument(
        "--real-imu-signal-mode",
        type=str,
        default="acc",
        choices=["acc", "gyro", "both"],
        help="Which signal subset to calibrate against the real IMU reference.",
    )
    export_virtual_imu_parser.add_argument(
        "--real-imu-percentile-resolution",
        type=int,
        default=100,
        help="Number of percentile bins used by the article-style rank mapping calibration.",
    )
    export_virtual_imu_parser.add_argument(
        "--no-real-imu-per-class-calibration",
        action="store_true",
        help="Disable per-class calibration and always use the full real IMU reference distribution.",
    )
    export_virtual_imu_parser.add_argument(
        "--estimate-sensor-frame",
        action="store_true",
        help="Estimate a diagnostic per-clip frame alignment for the selected IMU sensors.",
    )
    export_virtual_imu_parser.add_argument(
        "--estimate-sensor-names",
        nargs="+",
        default=None,
        help="Optional list of sensor names targeted by the frame-alignment estimator.",
    )
    return parser


def _add_shared_dataset_arguments(parser: argparse.ArgumentParser, *, include_output_dir: bool) -> None:
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/RobotEmotions"),
        help="Root folder of the RobotEmotions dataset.",
    )
    if include_output_dir:
        parser.add_argument(
            "--output-dir",
            type=Path,
            required=True,
            help="Output directory for exported artifacts.",
        )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["10ms", "30ms"],
        help="Domains to process.",
    )
    parser.add_argument(
        "--clip-id",
        nargs="+",
        default=None,
        help="Optional explicit clip_id selection.",
    )


def _add_pose_export_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_motionbert_arguments: bool,
) -> None:
    _add_shared_dataset_arguments(parser, include_output_dir=True)
    parser.add_argument(
        "--fps-target",
        type=int,
        default=20,
        help="Target fps for the pose sequence output.",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="openmmlab",
        help="Backend environment selector: openmmlab, current, auto, or another Conda env name.",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug overlay video generation.",
    )
    if not include_motionbert_arguments:
        return
    parser.add_argument(
        "--debug-2d",
        dest="debug_2d",
        action="store_true",
        default=None,
        help="Enable the 2D debug overlay videos for the 3D-capable exports.",
    )
    parser.add_argument(
        "--no-debug-2d",
        dest="debug_2d",
        action="store_false",
        default=None,
        help="Disable the 2D debug overlay videos for the 3D-capable exports.",
    )
    parser.add_argument(
        "--debug-3d",
        dest="debug_3d",
        action="store_true",
        default=None,
        help="Enable the 3D debug overlay video for the 3D-capable exports.",
    )
    parser.add_argument(
        "--no-debug-3d",
        dest="debug_3d",
        action="store_false",
        default=None,
        help="Disable the 3D debug overlay video for the 3D-capable exports.",
    )
    parser.add_argument(
        "--motionbert-env-name",
        type=str,
        default=None,
        help="Optional Conda env override for the MotionBERT backend.",
    )
    parser.add_argument(
        "--motionbert-window-size",
        type=int,
        default=81,
        help="Requested temporal window size for MotionBERT.",
    )
    parser.add_argument(
        "--motionbert-window-overlap",
        type=float,
        default=0.5,
        help="Window overlap ratio for MotionBERT.",
    )
    parser.add_argument(
        "--motionbert-device",
        type=str,
        default="auto",
        help="MotionBERT device selector passed to the backend.",
    )
    parser.add_argument(
        "--no-motionbert-confidence",
        action="store_true",
        help="Disable the confidence channel in the MotionBERT input tensor.",
    )
    parser.add_argument(
        "--allow-motionbert-fallback-backend",
        action="store_true",
        help="Allow the heuristic fallback backend if the real MotionBERT backend fails.",
    )
