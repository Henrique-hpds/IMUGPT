"""RobotEmotions dataset scanner, parser, and exporter."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metadata import (
    CHANNEL_AXIS_ORDER,
    SENSOR_MAPPING_POLICY_NOTE,
    get_protocol_info,
    get_sensor_name,
    get_user_profile,
)

EXPECTED_NUM_SENSORS = 4
CHANNELS_PER_SENSOR = 6
USER_DIR_RE = re.compile(r"(?i)^user(\d+)$")
TAG_DIR_RE = re.compile(r"(?i)^tag(\d+)$")
TIMESTAMP_COL_RE = re.compile(r"(?i)^timestamp\d*$")


@dataclass(frozen=True)
class RobotEmotionsClipRecord:
    clip_id: str
    domain: str
    user_id: int
    tag_number: int
    tag_dir: Path
    imu_csv_path: Path
    video_path: Path
    source_rel_dir: str
    take_id: str | None
    participant: dict[str, Any]
    protocol: dict[str, Any] | None


@dataclass(frozen=True)
class RobotEmotionsExtractedClip:
    record: RobotEmotionsClipRecord
    timestamps_sec: np.ndarray
    timestamp_column_names: tuple[str, ...]
    timestamp_columns_sec: np.ndarray
    imu: np.ndarray
    sensor_ids: tuple[int, ...]
    video_metadata: dict[str, Any]
    quality_report: dict[str, Any]

    @property
    def sensor_names(self) -> list[str]:
        return [get_sensor_name(sensor_id) for sensor_id in self.sensor_ids]

    def to_manifest_entry(
        self,
        *,
        output_dir: Path | None = None,
        imu_npz_path: Path | None = None,
        metadata_json_path: Path | None = None,
    ) -> dict[str, Any]:
        imu_shape = [int(x) for x in self.imu.shape]
        entry = {
            "clip_id": self.record.clip_id,
            "dataset": "RobotEmotions",
            "domain": self.record.domain,
            "user_id": int(self.record.user_id),
            "tag_number": int(self.record.tag_number),
            "take_id": self.record.take_id,
            "labels": {
                "emotion": None if self.record.protocol is None else self.record.protocol.get("emotion"),
                "action": None if self.record.protocol is None else self.record.protocol.get("action"),
                "stimulus": None if self.record.protocol is None else self.record.protocol.get("stimulus"),
            },
            "participant": dict(self.record.participant),
            "protocol": None if self.record.protocol is None else dict(self.record.protocol),
            "source": {
                "tag_dir": str(self.record.tag_dir.resolve()),
                "source_rel_dir": str(self.record.source_rel_dir),
                "imu_csv_path": str(self.record.imu_csv_path.resolve()),
                "video_path": str(self.record.video_path.resolve()),
            },
            "imu": {
                "shape": imu_shape,
                "flattened_shape": [imu_shape[0], imu_shape[1] * imu_shape[2]],
                "sensor_ids": [int(sensor_id) for sensor_id in self.sensor_ids],
                "sensor_names": self.sensor_names,
                "sensor_id_to_name": {
                    str(sensor_id): get_sensor_name(sensor_id) for sensor_id in self.sensor_ids
                },
                "channel_axis_order": list(CHANNEL_AXIS_ORDER),
                "timestamps_sec_range": [
                    float(self.timestamps_sec[0]) if len(self.timestamps_sec) > 0 else 0.0,
                    float(self.timestamps_sec[-1]) if len(self.timestamps_sec) > 0 else 0.0,
                ],
            },
            "video": dict(self.video_metadata),
            "sync": {
                "known_offset_sec": None,
                "status": "unknown",
                "notes": "Extractor does not infer cross-modal synchronization between video and IMU.",
            },
            "quality_report": dict(self.quality_report),
            "notes": {
                "sensor_mapping_policy": SENSOR_MAPPING_POLICY_NOTE,
            },
        }
        if output_dir is not None:
            entry["export_root"] = str(output_dir.resolve())
        if imu_npz_path is not None or metadata_json_path is not None:
            entry["artifacts"] = {
                "imu_npz_path": None if imu_npz_path is None else str(imu_npz_path.resolve()),
                "metadata_json_path": (
                    None if metadata_json_path is None else str(metadata_json_path.resolve())
                ),
            }
        return entry


class RobotEmotionsExtractor:
    """Prepare RobotEmotions clips for future pose-pipeline integration."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        domains: tuple[str, ...] = ("10ms", "30ms"),
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.domains = tuple(str(domain) for domain in domains)

    def scan(self) -> list[RobotEmotionsClipRecord]:
        records: list[RobotEmotionsClipRecord] = []
        clip_id_counts: dict[str, int] = {}
        for domain in self.domains:
            domain_root = self.dataset_root / domain
            if not domain_root.exists():
                continue
            for tag_dir in sorted(domain_root.rglob("*")):
                if not tag_dir.is_dir():
                    continue
                tag_number = _parse_tag_number(tag_dir.name)
                if tag_number is None:
                    continue
                user_id = _find_last_user_id(tag_dir.relative_to(domain_root).parts)
                if user_id is None:
                    continue
                capture_pairs = _pair_capture_files(
                    tag_dir=tag_dir,
                    user_id=user_id,
                    tag_number=tag_number,
                )
                for imu_csv_path, video_path in capture_pairs:
                    take_id = _infer_take_id(
                        imu_csv_path=imu_csv_path,
                        video_path=video_path,
                        user_id=user_id,
                        tag_number=tag_number,
                    )
                    clip_id = _build_clip_id(
                        domain=domain,
                        user_id=user_id,
                        tag_number=tag_number,
                        take_id=take_id,
                        clip_id_counts=clip_id_counts,
                    )
                    record = RobotEmotionsClipRecord(
                        clip_id=clip_id,
                        domain=domain,
                        user_id=user_id,
                        tag_number=tag_number,
                        tag_dir=tag_dir,
                        imu_csv_path=imu_csv_path,
                        video_path=video_path,
                        source_rel_dir=str(tag_dir.relative_to(self.dataset_root)),
                        take_id=take_id,
                        participant=get_user_profile(domain, user_id),
                        protocol=get_protocol_info(domain, tag_number),
                    )
                    records.append(record)
        records.sort(
            key=lambda item: (
                item.domain,
                item.user_id,
                item.tag_number,
                item.source_rel_dir,
                "" if item.take_id is None else item.take_id,
            )
        )
        return records

    def extract_clip(self, record: RobotEmotionsClipRecord) -> RobotEmotionsExtractedClip:
        frame = pd.read_csv(record.imu_csv_path)
        if frame.shape[1] < 2:
            raise ValueError(f"CSV has insufficient columns: {record.imu_csv_path}")

        timestamp_column_names, timestamp_columns_sec, reference_timestamps_sec = _parse_timestamps(frame)
        imu, sensor_ids, parsing_mode = _parse_imu_channels(frame)

        valid_rows = (~np.isnan(reference_timestamps_sec)) & (~np.isnan(imu).any(axis=(1, 2)))
        total_rows = int(len(reference_timestamps_sec))
        dropped_rows = int(total_rows - int(valid_rows.sum()))
        if not np.any(valid_rows):
            raise ValueError(f"No valid IMU rows left after cleaning: {record.imu_csv_path}")

        timestamps_sec = reference_timestamps_sec[valid_rows]
        timestamp_columns_sec = timestamp_columns_sec[:, valid_rows]
        imu = imu[valid_rows]

        sort_idx = np.argsort(timestamps_sec, kind="stable")
        timestamps_sec = timestamps_sec[sort_idx]
        timestamp_columns_sec = timestamp_columns_sec[:, sort_idx]
        imu = imu[sort_idx]

        duplicate_groups = int(np.sum(np.diff(timestamps_sec) == 0.0))
        timestamps_sec, timestamp_columns_sec, imu, duplicates_collapsed = _collapse_duplicate_timestamps(
            timestamps_sec=timestamps_sec,
            timestamp_columns_sec=timestamp_columns_sec,
            imu=imu,
        )

        video_metadata = _read_video_metadata(record.video_path)
        row_ratio = float(len(timestamps_sec) / total_rows) if total_rows > 0 else 0.0

        warnings: list[str] = []
        if len(sensor_ids) != EXPECTED_NUM_SENSORS:
            warnings.append(
                f"expected_{EXPECTED_NUM_SENSORS}_sensors_but_found_{len(sensor_ids)}"
            )
        if dropped_rows > 0:
            warnings.append("invalid_rows_removed")
        if duplicates_collapsed > 0:
            warnings.append("duplicate_timestamps_collapsed")
        if parsing_mode != "header_mapping":
            warnings.append(f"imu_parsed_with_{parsing_mode}")
        if row_ratio < 0.9:
            warnings.append("less_than_90_percent_rows_kept")
        if not bool(video_metadata.get("available", False)):
            warnings.append("video_metadata_unavailable")

        timestamp_spread_ms = 0.0
        if timestamp_columns_sec.shape[0] > 1:
            timestamp_spread_ms = float(np.nanmedian(np.nanstd(timestamp_columns_sec, axis=0)) * 1000.0)

        quality_report = {
            "status": "ok" if len(warnings) == 0 else "warning",
            "warnings": warnings,
            "rows_total": total_rows,
            "rows_kept": int(len(timestamps_sec)),
            "rows_removed": dropped_rows,
            "duplicate_timestamps_before_collapse": duplicate_groups,
            "duplicate_groups_collapsed": duplicates_collapsed,
            "sensor_count": int(len(sensor_ids)),
            "sensor_ids": [int(sensor_id) for sensor_id in sensor_ids],
            "parsing_mode": parsing_mode,
            "timestamps_monotonic_non_decreasing": bool(np.all(np.diff(timestamps_sec) >= 0.0)),
            "timestamp_columns": list(timestamp_column_names),
            "timestamp_column_count": int(len(timestamp_column_names)),
            "timestamp_cross_column_spread_ms": timestamp_spread_ms,
            "video_metadata_available": bool(video_metadata.get("available", False)),
        }

        return RobotEmotionsExtractedClip(
            record=record,
            timestamps_sec=timestamps_sec.astype(np.float32, copy=False),
            timestamp_column_names=tuple(timestamp_column_names),
            timestamp_columns_sec=timestamp_columns_sec.astype(np.float32, copy=False),
            imu=imu.astype(np.float32, copy=False),
            sensor_ids=tuple(int(sensor_id) for sensor_id in sensor_ids),
            video_metadata=video_metadata,
            quality_report=quality_report,
        )

    def export(
        self,
        output_dir: str | Path,
        *,
        limit: int | None = None,
    ) -> dict[str, Any]:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        records = self.scan()
        if limit is not None:
            records = records[: int(limit)]

        manifest_entries: list[dict[str, Any]] = []
        for record in records:
            clip = self.extract_clip(record)
            clip_dir = output_root / record.domain / f"user_{record.user_id:02d}" / record.clip_id
            clip_dir.mkdir(parents=True, exist_ok=True)

            imu_npz_path = clip_dir / "imu.npz"
            np.savez_compressed(
                imu_npz_path,
                timestamps_sec=clip.timestamps_sec,
                timestamp_columns_sec=clip.timestamp_columns_sec,
                timestamp_column_names=np.asarray(clip.timestamp_column_names),
                imu=clip.imu,
                imu_flat=clip.imu.reshape(int(clip.imu.shape[0]), -1),
                sensor_ids=np.asarray(clip.sensor_ids, dtype=np.int32),
                channel_axis_order=np.asarray(CHANNEL_AXIS_ORDER),
            )

            metadata_json_path = clip_dir / "metadata.json"
            manifest_entry = clip.to_manifest_entry(
                output_dir=output_root,
                imu_npz_path=imu_npz_path,
                metadata_json_path=metadata_json_path,
            )
            metadata_json_path.write_text(
                json.dumps(manifest_entry, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
            manifest_entries.append(manifest_entry)

        manifest_path = output_root / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for entry in manifest_entries:
                handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

        summary = {
            "dataset_root": str(self.dataset_root.resolve()),
            "output_dir": str(output_root.resolve()),
            "domains": list(self.domains),
            "num_clips": int(len(manifest_entries)),
            "manifest_path": str(manifest_path.resolve()),
        }
        summary_path = output_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        return summary


def _parse_tag_number(name: str) -> int | None:
    match = TAG_DIR_RE.match(str(name).strip())
    if match is None:
        return None
    return int(match.group(1))


def _find_last_user_id(parts: tuple[str, ...] | list[str]) -> int | None:
    last_user_id: int | None = None
    for part in parts:
        match = USER_DIR_RE.match(str(part).strip())
        if match is not None:
            last_user_id = int(match.group(1))
    return last_user_id


def _pair_capture_files(
    *,
    tag_dir: Path,
    user_id: int,
    tag_number: int,
) -> list[tuple[Path, Path]]:
    csv_candidates = sorted(tag_dir.glob("*.csv"))
    video_candidates = sorted(tag_dir.glob("*.mp4"))
    if len(csv_candidates) == 0 or len(video_candidates) == 0:
        return []

    csv_by_key = _group_candidates_by_capture_key(
        candidates=csv_candidates,
        prefixes=[f"esp_{user_id}_{tag_number}"],
    )
    video_by_key = _group_candidates_by_capture_key(
        candidates=video_candidates,
        prefixes=[
            f"tag_{user_id}_{tag_number}",
            f"tag_{tag_number}",
            f"tag{tag_number}",
        ],
    )

    pairs: list[tuple[Path, Path]] = []
    for capture_key in sorted(set(csv_by_key).intersection(video_by_key), key=_capture_sort_key):
        csv_paths = sorted(csv_by_key[capture_key])
        video_paths = sorted(video_by_key[capture_key])
        pair_count = min(len(csv_paths), len(video_paths))
        for idx in range(pair_count):
            pairs.append((csv_paths[idx], video_paths[idx]))
    return pairs


def _group_candidates_by_capture_key(
    *,
    candidates: list[Path],
    prefixes: list[str],
) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for candidate in candidates:
        capture_key = _extract_capture_key(candidate.stem, prefixes)
        grouped.setdefault(capture_key, []).append(candidate)
    return grouped


def _extract_capture_key(stem: str, prefixes: list[str]) -> str:
    normalized_stem = stem.strip().lower()
    normalized_prefixes = sorted((prefix.strip().lower() for prefix in prefixes), key=len, reverse=True)
    for prefix in normalized_prefixes:
        if normalized_stem.startswith(prefix):
            suffix = normalized_stem[len(prefix) :].strip("_- ")
            return suffix
    return normalized_stem


def _capture_sort_key(capture_key: str) -> tuple[int, int | str]:
    if capture_key == "":
        return (0, 0)
    if capture_key.isdigit():
        return (1, int(capture_key))
    return (2, capture_key)


def _infer_take_id(
    *,
    imu_csv_path: Path,
    video_path: Path,
    user_id: int,
    tag_number: int,
) -> str | None:
    sanitized_prefixes = [
        f"esp_{user_id}_{tag_number}",
        f"tag_{user_id}_{tag_number}",
        f"tag_{tag_number}",
    ]
    extras: list[str] = []
    for stem in (imu_csv_path.stem.lower(), video_path.stem.lower()):
        candidate = stem
        for prefix in sanitized_prefixes:
            if candidate.startswith(prefix):
                candidate = candidate[len(prefix) :]
                break
        candidate = candidate.strip("_- ")
        if len(candidate) > 0:
            extras.append(candidate)
    if len(extras) == 0:
        return None
    merged = "_".join(dict.fromkeys(extras))
    return re.sub(r"[^a-z0-9]+", "_", merged).strip("_") or None


def _build_clip_id(
    *,
    domain: str,
    user_id: int,
    tag_number: int,
    take_id: str | None,
    clip_id_counts: dict[str, int],
) -> str:
    base = f"robot_emotions_{domain}_u{user_id:02d}_tag{tag_number:02d}"
    if take_id is not None:
        base = f"{base}_{take_id}"
    count = clip_id_counts.get(base, 0) + 1
    clip_id_counts[base] = count
    if count == 1:
        return base
    return f"{base}_dup{count}"


def _parse_timestamps(frame: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    timestamp_column_names = [
        str(column)
        for column in frame.columns
        if TIMESTAMP_COL_RE.match(str(column).strip().lower()) is not None
    ]
    if len(timestamp_column_names) == 0:
        timestamp_column_names = [str(frame.columns[0])]

    columns_sec: list[np.ndarray] = []
    for column_name in timestamp_column_names:
        values = pd.to_numeric(frame[column_name], errors="coerce").to_numpy(dtype=np.float64) / 1000.0
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            values = values - values[valid_mask][0]
        columns_sec.append(values)

    timestamp_columns_sec = np.stack(columns_sec, axis=0)
    reference_timestamps_sec = np.nanmedian(timestamp_columns_sec, axis=0)
    valid_reference = ~np.isnan(reference_timestamps_sec)
    if np.any(valid_reference):
        reference_timestamps_sec = reference_timestamps_sec - reference_timestamps_sec[valid_reference][0]
    return timestamp_column_names, timestamp_columns_sec, reference_timestamps_sec


def _parse_imu_channels(frame: pd.DataFrame) -> tuple[np.ndarray, tuple[int, ...], str]:
    acc_cols = _collect_axis_columns(frame, prefix="acc")
    gyro_cols = _collect_axis_columns(frame, prefix="gyro")
    sensor_ids = sorted({sid for sid, _ in acc_cols}.intersection({sid for sid, _ in gyro_cols}))

    sensor_blocks: list[np.ndarray] = []
    resolved_sensor_ids: list[int] = []
    for sensor_id in sensor_ids:
        channel_values: list[np.ndarray] = []
        complete = True
        for axis in ("x", "y", "z"):
            key = (sensor_id, axis)
            if key not in acc_cols:
                complete = False
                break
            channel_values.append(
                pd.to_numeric(frame[acc_cols[key]], errors="coerce").to_numpy(dtype=np.float64)
            )
        if not complete:
            continue
        for axis in ("x", "y", "z"):
            key = (sensor_id, axis)
            if key not in gyro_cols:
                complete = False
                break
            channel_values.append(
                pd.to_numeric(frame[gyro_cols[key]], errors="coerce").to_numpy(dtype=np.float64)
            )
        if not complete:
            continue
        sensor_blocks.append(np.stack(channel_values, axis=1))
        resolved_sensor_ids.append(int(sensor_id))

    if len(sensor_blocks) > 0:
        imu = np.stack(sensor_blocks, axis=1)
        return imu, tuple(resolved_sensor_ids), "header_mapping"

    ignored_terms = ("timestamp", "pitch", "roll", "yaw")
    numeric_candidates = [
        column
        for column in frame.columns
        if not any(term in str(column).strip().lower() for term in ignored_terms)
    ]
    numeric_values = frame[numeric_candidates].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if numeric_values.shape[1] < CHANNELS_PER_SENSOR:
        raise ValueError("Could not recover IMU channels from CSV.")
    sensor_count = numeric_values.shape[1] // CHANNELS_PER_SENSOR
    clipped = numeric_values[:, : sensor_count * CHANNELS_PER_SENSOR]
    imu = clipped.reshape(int(clipped.shape[0]), sensor_count, CHANNELS_PER_SENSOR)
    sensor_ids = tuple(range(1, sensor_count + 1))
    return imu, sensor_ids, "numeric_fallback"


def _collect_axis_columns(frame: pd.DataFrame, *, prefix: str) -> dict[tuple[int, str], str]:
    collected: dict[tuple[int, str], str] = {}
    for column in frame.columns:
        lower = str(column).strip().lower()
        if prefix not in lower:
            continue
        sensor_match = re.search(r"(\d+)", lower)
        if sensor_match is None:
            continue
        sensor_id = int(sensor_match.group(1))
        axis: str | None = None
        if re.search(r"(?:^|[_\-\s])x(?:$|[_\-\s])", lower) or lower.endswith("x"):
            axis = "x"
        elif re.search(r"(?:^|[_\-\s])y(?:$|[_\-\s])", lower) or lower.endswith("y"):
            axis = "y"
        elif re.search(r"(?:^|[_\-\s])z(?:$|[_\-\s])", lower) or lower.endswith("z"):
            axis = "z"
        if axis is not None:
            collected[(sensor_id, axis)] = str(column)
    return collected


def _collapse_duplicate_timestamps(
    *,
    timestamps_sec: np.ndarray,
    timestamp_columns_sec: np.ndarray,
    imu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if len(timestamps_sec) <= 1:
        return timestamps_sec, timestamp_columns_sec, imu, 0

    grouped_timestamps: list[float] = []
    grouped_columns: list[np.ndarray] = []
    grouped_imu: list[np.ndarray] = []
    groups_collapsed = 0
    start = 0
    while start < len(timestamps_sec):
        end = start + 1
        while end < len(timestamps_sec) and timestamps_sec[end] == timestamps_sec[start]:
            end += 1
        if end - start > 1:
            groups_collapsed += 1
        grouped_timestamps.append(float(np.mean(timestamps_sec[start:end])))
        grouped_columns.append(np.mean(timestamp_columns_sec[:, start:end], axis=1))
        grouped_imu.append(np.mean(imu[start:end], axis=0))
        start = end

    return (
        np.asarray(grouped_timestamps, dtype=np.float64),
        np.stack(grouped_columns, axis=1),
        np.stack(grouped_imu, axis=0),
        groups_collapsed,
    )


def _read_video_metadata(video_path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {
            "available": False,
            "video_path": str(video_path.resolve()),
            "reason": "ffprobe_not_found",
        }
    except subprocess.CalledProcessError as exc:
        return {
            "available": False,
            "video_path": str(video_path.resolve()),
            "reason": "ffprobe_failed",
            "stderr": exc.stderr.strip(),
        }

    payload = json.loads(completed.stdout)
    stream = payload.get("streams", [{}])[0]
    format_info = payload.get("format", {})
    fps = _parse_ffprobe_fps(stream.get("avg_frame_rate"))
    num_frames = _parse_optional_int(stream.get("nb_frames"))
    duration_sec = _parse_optional_float(stream.get("duration"))
    if duration_sec is None:
        duration_sec = _parse_optional_float(format_info.get("duration"))
    if num_frames is None and fps is not None and duration_sec is not None:
        num_frames = int(round(fps * duration_sec))

    return {
        "available": True,
        "video_path": str(video_path.resolve()),
        "fps": fps,
        "num_frames": num_frames,
        "duration_sec": duration_sec,
    }


def _parse_ffprobe_fps(raw_value: str | None) -> float | None:
    if raw_value is None or raw_value in {"", "0/0"}:
        return None
    try:
        return float(Fraction(str(raw_value)))
    except (ValueError, ZeroDivisionError):
        return None


def _parse_optional_int(raw_value: Any) -> int | None:
    if raw_value in (None, ""):
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def _parse_optional_float(raw_value: Any) -> float | None:
    if raw_value in (None, ""):
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract RobotEmotions into manifest + npz artifacts.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/RobotEmotions"),
        help="Root folder of the RobotEmotions dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. If omitted, only a scan summary is printed.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["10ms", "30ms"],
        help="Domains to scan.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional clip limit for quick validation runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    extractor = RobotEmotionsExtractor(args.dataset_root, domains=tuple(args.domains))
    records = extractor.scan()
    if args.limit is not None:
        records = records[: int(args.limit)]

    if args.output_dir is None:
        summary = {
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "domains": list(args.domains),
            "num_records": int(len(records)),
            "sample_clip_ids": [record.clip_id for record in records[:5]],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return 0

    summary = extractor.export(args.output_dir, limit=args.limit)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0
