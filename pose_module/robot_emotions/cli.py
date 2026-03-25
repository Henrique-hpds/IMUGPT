"""CLI for RobotEmotions extraction and pose2d export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .extractor import RobotEmotionsExtractor
from .pose2d import run_robot_emotions_pose2d


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

    parser.error(f"Unsupported command: {args.command}")
    return 2


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RobotEmotions extraction and pose2d export.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Scan RobotEmotions records.")
    _add_shared_dataset_arguments(scan_parser, include_output_dir=False)

    export_imu_parser = subparsers.add_parser("export-imu", help="Export IMU artifacts and manifest.")
    _add_shared_dataset_arguments(export_imu_parser, include_output_dir=True)

    export_pose_parser = subparsers.add_parser(
        "export-pose2d",
        help="Export stage-5.3 pose2d artifacts alongside the extracted IMU artifacts.",
    )
    _add_shared_dataset_arguments(export_pose_parser, include_output_dir=True)
    export_pose_parser.add_argument(
        "--fps-target",
        type=int,
        default=20,
        help="Target fps for the pose sequence output.",
    )
    export_pose_parser.add_argument(
        "--env-name",
        type=str,
        default="auto",
        help="Backend environment selector: auto, current, or a Conda env name (for example: openmmlab).",
    )
    export_pose_parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug overlay video generation.",
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
