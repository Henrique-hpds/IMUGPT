from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pose_module.processing.frequency_alignment import estimate_sampling_frequency_hz, prepare_real_signal_for_virtual_comparison
from pose_module.robot_emotions.metadata import get_protocol_info, get_sensor_name

CHANNEL_GROUP_TO_LABELS = {"acc": ("ax", "ay", "az"), "gyro": ("gx", "gy", "gz")}
VIRTUAL_GROUP_TO_KEY = {"acc": "acc", "gyro": "gyro"}
GROUP_TO_UNIT = {"acc": "m/s^2", "gyro": "rad/s"}
AXIS_COLORS = {"x": "tab:red", "y": "tab:green", "z": "tab:blue",}
SENSOR_LINESTYLES = ("-", "--", ":", "-.")
CAPTURE_TABLE_MAX_COLUMNS = 24
CAPTURE_TABLE_DISPLAY_WIDTH = 240


def normalize_take_id(value: Any, clip_id: str | None = None) -> str:
	if value in (None, ""):
		if clip_id is not None:
			match = re.search(r"_(\d+)$", str(clip_id))
			if match is not None:
				return str(int(match.group(1)))
		return "1"
	return str(value)

def find_project_root() -> Path:
	"""Finds IMUGPT repository root by searching for `pose_module` and `output`."""
	candidate_roots = []
	cwd = Path.cwd().resolve()
	candidate_roots.extend([cwd, *cwd.parents])

	seen = set()
	for candidate in candidate_roots:
		candidate = candidate.resolve()
		if candidate in seen:
			continue
		seen.add(candidate)

		if (candidate / "pose_module").exists() and (candidate / "output").exists():
			output_dir = candidate / "output"
			if any(output_dir.iterdir()):
				return candidate

	raise RuntimeError("Didn't find project root")


def configure_capture_table_display() -> None:
	pd.set_option("display.max_columns", max(int(pd.get_option("display.max_columns") or 0), CAPTURE_TABLE_MAX_COLUMNS))
	pd.set_option("display.width", max(int(pd.get_option("display.width") or 0), CAPTURE_TABLE_DISPLAY_WIDTH))


def resolve_capture_metadata(domain: str, user_id: int, tag_number: int) -> dict[str, Any]:
	protocol_info = get_protocol_info(str(domain), int(tag_number))
	return {
		"emotion": None if protocol_info is None else protocol_info.get("emotion"),
		"modality": None if protocol_info is None else protocol_info.get("modality"),
		"stimulus": None if protocol_info is None else protocol_info.get("stimulus"),
		"stimulus_details": None if protocol_info is None else protocol_info.get("stimulus_details"),
		"protocol_tag_10ms": None if protocol_info is None else protocol_info.get("tag_10ms"),
		"protocol_tag_30ms": None if protocol_info is None else protocol_info.get("tag_30ms")
	}

def _extract_virtual_npz_path(entry: dict[str, Any]) -> str | None:
	artifacts = dict(entry.get("artifacts", {}))
	if artifacts.get("virtual_imu_npz_path") not in (None, ""):
		return str(artifacts.get("virtual_imu_npz_path"))

	virtual_imu_artifacts = dict(entry.get("virtual_imu_artifacts", {}))
	if virtual_imu_artifacts.get("virtual_imu_npz_path") not in (None, ""):
		return str(virtual_imu_artifacts.get("virtual_imu_npz_path"))

	return None

def _extract_real_npz_path(entry: dict[str, Any]) -> str | None:
	input_artifacts = dict(entry.get("input_artifacts", {}))
	if input_artifacts.get("imu_npz_path") not in (None, ""):
		return str(input_artifacts.get("imu_npz_path"))

	artifacts = dict(entry.get("artifacts", {}))
	if artifacts.get("imu_npz_path") not in (None, ""):
		return str(artifacts.get("imu_npz_path"))

	return None

def _parse_clip_identity(*values: Any) -> tuple[str, int, int] | None:
	for value in values:
		if value in (None, ""):
			continue
		match = re.search(r"robot_emotions_(\d+ms)_u(\d+)_tag(\d+)", str(value))
		if match is not None:
			domain = str(match.group(1))
			user_id = int(match.group(2))
			tag_number = int(match.group(3))
			return domain, user_id, tag_number
	return None

def _find_real_capture_dir_from_outputs(output_root: Path, domain: str, user_id: int, clip_id: str) -> Path | None:
	output_dir = None
	resolved_root = Path(output_root).resolve()
	for candidate in [resolved_root, *resolved_root.parents]:
		if candidate.is_dir() and candidate.name == "output":
			output_dir = candidate
			break
		candidate_output = candidate / "output"
		if candidate_output.exists() and candidate_output.is_dir():
			output_dir = candidate_output
			break

	if output_dir is None:
		return None

	if not output_dir.exists():
		return None

	user_folder = f"user_{int(user_id):02d}"
	for child in sorted(output_dir.iterdir()):
		if not child.is_dir() or child.name.startswith("exp_"):
			continue
		candidate_dir = child / str(domain) / user_folder / str(clip_id)
		candidate_imu = candidate_dir / "imu.npz"
		candidate_metadata = candidate_dir / "metadata.json"
		if candidate_imu.exists() and candidate_metadata.exists():
			return candidate_dir

	return None

def _resolve_manifest_context(path: Path) -> tuple[Path | None, Path | None]:
	resolved_path = Path(path).resolve()
	manifest_local = resolved_path / "virtual_imu_manifest.jsonl"
	if manifest_local.exists():
		return resolved_path, None

	selected_virtual_npz: Path | None = None
	if resolved_path.is_dir():
		candidate_virtual = resolved_path / "virtual_imu" / "virtual_imu.npz"
		if candidate_virtual.exists():
			selected_virtual_npz = candidate_virtual.resolve()

	for parent in [resolved_path, *resolved_path.parents]:
		manifest_path = parent / "virtual_imu_manifest.jsonl"
		if manifest_path.exists():
			return parent.resolve(), selected_virtual_npz

	return None, selected_virtual_npz

def build_exported_capture_table(output_root: Path | str) -> pd.DataFrame:
	"""Builds capture catalog from `virtual_imu_manifest.jsonl` entries."""
	configure_capture_table_display()
	requested_output_root = Path(output_root).resolve()
	manifest_root, selected_virtual_npz = _resolve_manifest_context(requested_output_root)
	rows = []
	seen_clip_ids: set[str] = set()

	if manifest_root is not None:
		manifest_path = manifest_root / "virtual_imu_manifest.jsonl"
		with manifest_path.open("r", encoding="utf-8") as handle:
			for raw_line in handle:
				line = raw_line.strip()
				if not line:
					continue
				entry = json.loads(line)

				virtual_npz_path = _extract_virtual_npz_path(entry)
				if virtual_npz_path in (None, ""):
					continue
				virtual_npz = Path(str(virtual_npz_path)).resolve()
				if not virtual_npz.exists():
					continue
				if selected_virtual_npz is not None and virtual_npz != selected_virtual_npz:
					continue

				reference_clip_id = str(entry.get("reference_clip_id") or entry.get("clip_id") or "")
				clip_id = str(entry.get("window_id") or entry.get("prompt_id") or entry.get("clip_id") or "")
				if not clip_id:
					continue

				identity = _parse_clip_identity(
					entry.get("domain"),
					entry.get("clip_id"),
					entry.get("reference_clip_id"),
					entry.get("window_id"),
					entry.get("prompt_id")
				)
				if identity is None:
					continue
				domain, user_id, tag_number = identity

				real_npz_path = _extract_real_npz_path(entry)
				real_npz: Path | None = None
				if real_npz_path not in (None, ""):
					real_candidate = Path(str(real_npz_path)).resolve()
					if real_candidate.exists():
						real_npz = real_candidate

				if real_npz is None and reference_clip_id:
					real_capture_dir = _find_real_capture_dir_from_outputs(
						output_root=requested_output_root,
						domain=domain,
						user_id=user_id,
						clip_id=reference_clip_id,
					)
					if real_capture_dir is not None:
						real_npz = (real_capture_dir / "imu.npz").resolve()

				if real_npz is None or not real_npz.exists():
					continue

				capture_metadata = resolve_capture_metadata(domain=domain, user_id=user_id, tag_number=tag_number)
				rows.append(
					{
						"clip_id": clip_id,
						"domain": domain,
						"user_id": user_id,
						"tag_number": tag_number,
						"take_id": normalize_take_id(entry.get("take_id"), clip_id=clip_id),
						"clip_dir": str(real_npz.parent.resolve()),
						"pose_dir": str(virtual_npz.parent.resolve()),
						**capture_metadata
					}
				)
				seen_clip_ids.add(clip_id)

	for metadata_path in sorted(requested_output_root.rglob("metadata.json")):
		try:
			metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
		except Exception:
			continue

		clip_id = str(metadata.get("clip_id") or "")
		if not clip_id or clip_id in seen_clip_ids:
			continue

		artifacts = dict(metadata.get("artifacts", {}))
		real_npz_path = artifacts.get("imu_npz_path")
		virtual_npz_path = artifacts.get("virtual_imu_npz_path")
		if real_npz_path in (None, "") or virtual_npz_path in (None, ""):
			continue

		real_npz = Path(str(real_npz_path))
		virtual_npz = Path(str(virtual_npz_path))
		if not real_npz.exists() or not virtual_npz.exists():
			continue

		domain = str(metadata.get("domain"))
		user_id = int(metadata.get("user_id"))
		tag_number = int(metadata.get("tag_number"))
		capture_metadata = resolve_capture_metadata(domain=domain, user_id=user_id, tag_number=tag_number)
		rows.append(
			{
				"clip_id": clip_id,
				"domain": domain,
				"user_id": user_id,
				"tag_number": tag_number,
				"take_id": normalize_take_id(metadata.get("take_id"), clip_id=clip_id),
				"clip_dir": str(real_npz.parent.resolve()),
				"pose_dir": str(virtual_npz.parent.resolve()),
				**capture_metadata
			}
		)
		seen_clip_ids.add(clip_id)

	frame = pd.DataFrame(rows)
	if frame.empty:
		raise RuntimeError(f"No capture ready to plot found at {requested_output_root}.")

	return frame.sort_values(["domain", "user_id", "tag_number", "take_id", "clip_id"], kind="stable").reset_index(drop=True)

def select_capture_row(captures_df: pd.DataFrame, domain: str, user_id: int, tag_number: int, take_id: Optional[str] = None) -> pd.Series:
	frame = captures_df.copy()

	frame = frame[(frame["domain"] == str(domain)) & (frame["user_id"] == int(user_id)) & (frame["tag_number"] == int(tag_number))]
 
	if take_id not in (None, ""):
		take_id_value = str(take_id)
		frame = frame[frame["take_id"].fillna("").astype(str) == take_id_value]
	if frame.empty:
		raise ValueError("No capture found for the choosen parameters.\nConsult CAPTURES_DF for the available options.")
	if len(frame) > 1:
		raise ValueError("2 or more captures have been found. Specify TAKE_ID.")

	return frame.iloc[0]

def resolve_real_sensor_names(metadata: dict[str, Any], payload: Any, sensor_count: int) -> list[str]:
	if "sensor_ids" in payload:
		sensor_ids = [int(value) for value in np.asarray(payload["sensor_ids"]).tolist()]
		if len(sensor_ids) == int(sensor_count):
			return [get_sensor_name(sensor_id) for sensor_id in sensor_ids]

	metadata_sensor_ids = metadata.get("imu", {}).get("sensor_ids", [])
	if len(metadata_sensor_ids) == int(sensor_count):
		return [get_sensor_name(sensor_id) for sensor_id in metadata_sensor_ids]

	metadata_sensor_names = metadata.get("imu", {}).get("sensor_names", [])
	if len(metadata_sensor_names) == int(sensor_count):
		return [str(value) for value in metadata_sensor_names]

	return [f"sensor_{sensor_index}" for sensor_index in range(int(sensor_count))]

def load_real_capture(clip_dir: Path | str) -> dict[str, Any]:
	clip_dir = Path(clip_dir)
	metadata = json.loads((clip_dir / "metadata.json").read_text(encoding="utf-8"))
	with np.load(clip_dir / "imu.npz", allow_pickle=True) as payload:
		imu = np.asarray(payload["imu"], dtype=np.float32)
		return {
			"timestamps_sec": np.asarray(payload["timestamps_sec"], dtype=np.float32),
			"imu": imu,
			"sensor_names": resolve_real_sensor_names(metadata, payload, sensor_count=imu.shape[1]),
			"channel_axis_order": list(metadata["imu"]["channel_axis_order"]),
			"metadata": metadata
		}

def load_virtual_capture(pose_dir: Path | str, filename: str = "virtual_imu.npz") -> dict[str, Any]:
	pose_dir = Path(pose_dir)
	payload_path = pose_dir / str(filename)
	with np.load(payload_path, allow_pickle=True) as payload:
		return {
			"timestamps_sec": np.asarray(payload["timestamps_sec"], dtype=np.float32),
			"acc": np.asarray(payload["acc"], dtype=np.float32),
			"gyro": np.asarray(payload["gyro"], dtype=np.float32),
			"sensor_names": [
				str(value) for value in np.asarray(payload["sensor_names"]).tolist()
			],
			"fps": float(np.asarray(payload["fps"]).item()),
			"clip_id": str(np.asarray(payload["clip_id"]).item()),
			"source": str(np.asarray(payload["source"]).item()),
			"path": str(payload_path)
		}

def resolve_selected_sensors(requested_sensors: Optional[Sequence[str]], available_sensors: Sequence[str]) -> list[str]:
	if requested_sensors is None:
		return list(available_sensors)

	selected_sensors = [str(sensor_name) for sensor_name in requested_sensors]
	missing_sensors = [sensor_name for sensor_name in selected_sensors if sensor_name not in set(available_sensors)]
 
	if len(missing_sensors) > 0:
		raise ValueError(f"Sensors not found: {missing_sensors}")

	return selected_sensors

def apply_time_range(timestamps_sec: np.ndarray, values: np.ndarray, time_range_sec: Optional[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
	if time_range_sec is None:
		return timestamps_sec, values
	start_sec, end_sec = time_range_sec
	mask = (timestamps_sec >= float(start_sec)) & (timestamps_sec <= float(end_sec))
	return timestamps_sec[mask], values[mask]

def plot_signal_block(
	ax: plt.Axes,
	timestamps_sec: np.ndarray,
	values: np.ndarray,
	sensor_names: Sequence[str],
	selected_sensors: Sequence[str],
	component_labels: Sequence[str],
	title: str,
	unit: str,
	time_range_sec: Optional[tuple[float, float]] = None,
	line_width: float = 1.5
) -> None:
    
	timestamps_plot, values_plot = apply_time_range(timestamps_sec, values, time_range_sec)
	if values_plot.shape[0] == 0:
		raise ValueError("TIME_RANGE_SEC didn't return any sample.")

	for sensor_order, sensor_name in enumerate(selected_sensors):
		sensor_index = list(sensor_names).index(sensor_name)
		line_style = SENSOR_LINESTYLES[sensor_order % len(SENSOR_LINESTYLES)]
		for component_index, component_label in enumerate(component_labels):
			axis_name = component_label[-1]
			ax.plot(
				timestamps_plot,
				values_plot[:, sensor_index, component_index],
				label=f"{sensor_name} / {component_label}",
				color=AXIS_COLORS[axis_name],
				linestyle=line_style,
				linewidth=line_width,
			)

	ax.set_title(title)
	ax.set_ylabel(unit)
	ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))

def plot_real_virtual_capture(
	captures_df: pd.DataFrame,
	domain: Optional[str] = None,
	user_id: Optional[int] = None,
	tag_number: Optional[int] = None,
	take_id: Optional[str] = None,
	signal_group: str = "acc",
	sensor_names: Optional[Sequence[str]] = None,
	time_range_sec: Optional[tuple[float, float]] = None,
	figsize: tuple[float, float] = (16, 10),
	line_width: float = 1.5,
	undersample_real_to_virtual: bool = True,
	show: bool = True
) -> tuple[pd.DataFrame, plt.Figure]:
	"""Plots real and virtual IMU signals and returns summary table + figure."""
	configure_capture_table_display()
 
	if signal_group not in CHANNEL_GROUP_TO_LABELS:
		raise ValueError("SIGNAL_GROUP must be 'acc' or 'gyro'.")

	capture_row = select_capture_row(captures_df, domain=domain, user_id=user_id, tag_number=tag_number, take_id=take_id)

	clip_dir = Path(capture_row["clip_dir"])
	pose_dir = Path(capture_row["pose_dir"])

	real_data = load_real_capture(clip_dir)
	virtual_data = load_virtual_capture(pose_dir)
	aligned_virtual_path = pose_dir / "virtual_imu_frame_aligned.npz"
	aligned_virtual_data = (load_virtual_capture(pose_dir, filename="virtual_imu_frame_aligned.npz") if aligned_virtual_path.exists() else None)

	selected_sensors = resolve_selected_sensors(sensor_names, real_data["sensor_names"])
	selected_sensors = resolve_selected_sensors(selected_sensors, virtual_data["sensor_names"])
	if aligned_virtual_data is not None:
		selected_sensors = resolve_selected_sensors(selected_sensors, aligned_virtual_data["sensor_names"])

	component_labels = CHANNEL_GROUP_TO_LABELS[signal_group]
	real_slice = slice(0, 3) if signal_group == "acc" else slice(3, 6)
	real_values = real_data["imu"][:, :, real_slice]
	virtual_values = virtual_data[VIRTUAL_GROUP_TO_KEY[signal_group]]
	aligned_virtual_values = (None if aligned_virtual_data is None else aligned_virtual_data[VIRTUAL_GROUP_TO_KEY[signal_group]])

	real_plot_timestamps = real_data["timestamps_sec"]
	real_plot_values = real_values
	real_plot_title = "Real IMU"
	frequency_summary = {
		"real_original_frequency_hz": estimate_sampling_frequency_hz(real_data["timestamps_sec"]),
		"real_plot_frequency_hz": estimate_sampling_frequency_hz(real_data["timestamps_sec"]),
		"virtual_frequency_hz": estimate_sampling_frequency_hz(virtual_data["timestamps_sec"]),
		"mean_time_error_ms": 0.0,
		"max_time_error_ms": 0.0
	}
 
	if undersample_real_to_virtual:
		real_plot_bundle = prepare_real_signal_for_virtual_comparison(real_timestamps_sec=real_data["timestamps_sec"], real_values=real_values, virtual_timestamps_sec=virtual_data["timestamps_sec"])
		real_plot_timestamps = real_plot_bundle["timestamps_sec"]
		real_plot_values = real_plot_bundle["values"]
		frequency_summary = real_plot_bundle["summary"]
		real_plot_title = "Real IMU undersampled"

	capture_metadata = resolve_capture_metadata(domain=str(capture_row["domain"]), user_id=int(capture_row["user_id"]), tag_number=int(capture_row["tag_number"]))
	summary_row = {
		"clip_id": capture_row["clip_id"],
		"domain": capture_row["domain"],
		"user_id": int(capture_row["user_id"]),
		"tag_number": int(capture_row["tag_number"]),
		"take_id": capture_row["take_id"],
		**capture_metadata,
		"signal_group": signal_group,
		"undersampled_real_to_virtual": bool(undersample_real_to_virtual),
		"selected_sensors": ", ".join(selected_sensors),
		"real_frames": int(real_values.shape[0]),
		"real_plot_frames": int(real_plot_values.shape[0]),
		"virtual_frames": int(virtual_values.shape[0]),
		"real_original_frequency_hz": float(frequency_summary["real_original_frequency_hz"]),
		"real_plot_frequency_hz": float(frequency_summary["real_plot_frequency_hz"]),
		"virtual_frequency_hz": float(frequency_summary["virtual_frequency_hz"]),
		"real_vs_virtual_mean_time_error_ms": float(frequency_summary["mean_time_error_ms"]),
		"real_vs_virtual_max_time_error_ms": float(frequency_summary["max_time_error_ms"]),
		"frame_aligned_available": bool(aligned_virtual_data is not None),
		"frame_aligned_frames": None if aligned_virtual_values is None else int(aligned_virtual_values.shape[0])
	}
 
	summary_df = pd.DataFrame([summary_row])

	_, real_values_for_scale = apply_time_range(real_plot_timestamps, real_plot_values, time_range_sec)
	_, virtual_values_for_scale = apply_time_range(virtual_data["timestamps_sec"], virtual_values, time_range_sec)
	values_for_scale = [real_values_for_scale, virtual_values_for_scale]
 
	if aligned_virtual_values is not None:
		_, aligned_virtual_values_for_scale = apply_time_range(aligned_virtual_data["timestamps_sec"], aligned_virtual_values, time_range_sec)
		values_for_scale.append(aligned_virtual_values_for_scale)

	if any(block.shape[0] == 0 for block in values_for_scale):
		raise ValueError("TIME_RANGE_SEC didn't return any sample for comparision with all the signals.")

	y_min = float(min(block.min() for block in values_for_scale))
	y_max = float(max(block.max() for block in values_for_scale))
	if y_min == y_max:
		y_min -= 1e-6
		y_max += 1e-6

	num_plot_rows = 3 if aligned_virtual_data is not None else 2
	fig, axes = plt.subplots(num_plot_rows, 3, figsize=figsize, sharex=False, sharey=True, constrained_layout=True)

	for component_index, component_label in enumerate(component_labels):
		axis_name = component_label[-1]
		real_component_values = real_plot_values[:, :, component_index : component_index + 1]
		virtual_component_values = virtual_values[:, :, component_index : component_index + 1]

		plot_signal_block(
			axes[0, component_index],
			timestamps_sec=real_plot_timestamps,
			values=real_component_values,
			sensor_names=real_data["sensor_names"],
			selected_sensors=selected_sensors,
			component_labels=(component_label,),
			title=f"{real_plot_title} | {axis_name} axis",
			unit=GROUP_TO_UNIT[signal_group],
			time_range_sec=time_range_sec,
			line_width=line_width
		)
  
		axes[0, component_index].set_ylim(y_min, y_max)

		plot_signal_block(
			axes[1, component_index],
			timestamps_sec=virtual_data["timestamps_sec"],
			values=virtual_component_values,
			sensor_names=virtual_data["sensor_names"],
			selected_sensors=selected_sensors,
			component_labels=(component_label,),
			title=f"Virtual IMU | {axis_name} axis",
			unit=GROUP_TO_UNIT[signal_group],
			time_range_sec=time_range_sec,
			line_width=line_width
		)
		axes[1, component_index].set_ylim(y_min, y_max)

		if aligned_virtual_data is not None:
			aligned_component_values = aligned_virtual_values[:, :, component_index : component_index + 1]
			plot_signal_block(
				axes[2, component_index],
				timestamps_sec=aligned_virtual_data["timestamps_sec"],
				values=aligned_component_values,
				sensor_names=aligned_virtual_data["sensor_names"],
				selected_sensors=selected_sensors,
				component_labels=(component_label,),
				title=f"Virtual IMU aligned | {axis_name} axis",
				unit=GROUP_TO_UNIT[signal_group],
				time_range_sec=time_range_sec,
				line_width=line_width
			)
   
			axes[2, component_index].set_ylim(y_min, y_max)
			axes[2, component_index].set_xlabel("Time (s)")
		else:
			axes[1, component_index].set_xlabel("Time (s)")

	if show:
		plt.show()

	return summary_df, fig


def list_capture_options(dataset_root: Path | str) -> dict[str, dict[int, list[int]]]:
	dataset_root = Path(dataset_root)
	domains = sorted([path.name for path in dataset_root.iterdir() if path.is_dir()])
	options: dict[str, dict[int, list[int]]] = {}
	for domain in domains:
		domain_dir = dataset_root / domain
		users = sorted(int(match.group(1)) for path in domain_dir.iterdir() if path.is_dir() and (match := re.match(r"^User(\d+)$", path.name, flags=re.IGNORECASE)))
		options[domain] = {}
		for user in users:
			user_dir = domain_dir / f"User{user}"
			tags = sorted(int(match.group(1)) for path in user_dir.iterdir() if path.is_dir() and (match := re.match(r"^Tag(\d+)$", path.name, flags=re.IGNORECASE)))
			options[domain][user] = tags
	return options


def resolve_csv_path(dataset_root: Path | str, domain: str, user: int, tag: int) -> Path:
	dataset_root = Path(dataset_root)
	tag_dir = dataset_root / domain / f"User{user}" / f"Tag{tag}"
	expected_path = tag_dir / f"ESP_{user}_{tag}.csv"
	if expected_path.exists():
		return expected_path

	candidate_paths = sorted(tag_dir.glob("ESP*.csv"))
	if not candidate_paths:
		raise FileNotFoundError(f"No CSV found in: {tag_dir}")
	return candidate_paths[0]


def prepare_robotemotions_capture(csv_path: Path | str, sensor_order: Sequence[str], selected_sensors: Sequence[str], start_idx: int = 0, end_idx: Optional[int] = None) -> dict[str, Any]:
	df = pd.read_csv(csv_path)

	timestamp_cols = [c for c in df.columns if c.lower().startswith("timestamp")]
	if not timestamp_cols:
		raise ValueError("No timestamp column found in the CSV.")

	timestamps_sec = pd.to_numeric(df[timestamp_cols[0]], errors="coerce")
	timestamps_sec = (timestamps_sec - timestamps_sec.iloc[0]) / 1000.0

	sensor_to_index = {sensor_name: index + 1 for index, sensor_name in enumerate(sensor_order)}
	invalid_sensors = [sensor_name for sensor_name in selected_sensors if sensor_name not in sensor_to_index]
	if invalid_sensors:
		raise ValueError(f"Invalid sensors: {invalid_sensors}. Valid options: {list(sensor_to_index.keys())}")

	if end_idx is None:
		sample_slice = slice(start_idx, None)
	else:
		sample_slice = slice(start_idx, end_idx)

	return {
		"df": df,
		"timestamps_sec": timestamps_sec,
		"sample_slice": sample_slice,
		"sensor_to_index": sensor_to_index
	}


def plot_robotemotions_imus_csv(
	*,
	dataset_root: Path | str,
	domain: str,
	user: int,
	tag: int,
	sensor_order: Sequence[str],
	selected_sensors: Sequence[str],
	modalities: Sequence[str] = ("acc", "gyro"),
	start_idx: int = 0,
	end_idx: Optional[int] = None,
	show: bool = True,
) -> tuple[Path, pd.DataFrame, plt.Figure]:
	csv_path = resolve_csv_path(dataset_root, domain, user, tag)
	prepared = prepare_robotemotions_capture(
		csv_path,
		sensor_order=sensor_order,
		selected_sensors=selected_sensors,
		start_idx=start_idx,
		end_idx=end_idx
	)

	df = prepared["df"]
	timestamps_sec = prepared["timestamps_sec"]
	sample_slice = prepared["sample_slice"]
	sensor_to_index = prepared["sensor_to_index"]

	valid_modalities = {"acc": ("X", "Y", "Z"), "gyro": ("X", "Y", "Z")}

	n_rows = len(selected_sensors)
	n_cols = len(modalities)
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.2 * n_rows), squeeze=False)

	for row_index, sensor_name in enumerate(selected_sensors):
		sensor_idx = sensor_to_index[sensor_name]

		for column_index, modality in enumerate(modalities):
			ax = axes[row_index, column_index]
			if modality not in valid_modalities:
				ax.set_title(f"{sensor_name} | invalid modality: {modality}")
				ax.axis("off")
				continue

			axis_labels = valid_modalities[modality]
			plotted = False
			for axis_label in axis_labels:
				column_name = f"{modality}_{axis_label}_{sensor_idx}"
				if column_name in df.columns:
					signal_values = pd.to_numeric(df[column_name], errors="coerce")
					ax.plot(
						timestamps_sec.iloc[sample_slice],
						signal_values.iloc[sample_slice],
						label=f"{modality}_{axis_label}",
						linewidth=1.0
					)
					plotted = True

			ax.set_title(f"{sensor_name} (idx={sensor_idx}) | {modality}")
			ax.set_xlabel("time (s)")
			ax.set_ylabel("raw value")
			ax.grid(alpha=0.25)

			if plotted:
				ax.legend(loc="upper right", ncol=3, fontsize=8)
			else:
				ax.text(0.5, 0.5, "No columns found for this sensor/modality", ha="center", va="center")

	plt.tight_layout()
	if show:
		plt.show()
	return csv_path, df, fig
