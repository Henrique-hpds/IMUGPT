import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple, Mapping, Sequence
import re

from .common import build_event_uid, create_sliding_windows, resample_imu_uniform
from .video_utils import read_video_info
from pipeline.utils.imu_unit_standardization import (
    repeat_sensor_channel_slices,
    standardize_imu_units,
)
from pipeline.sensors.sensor_meta import (
    ROBOT_EMOTIONS_SENSOR_LAYOUT,
    create_sensor_metas_robot_emotions,
)


ROBOT_EMOTIONS_SENSOR_ORDER: Tuple[str, ...] = tuple(
    str(item["name"]) for item in ROBOT_EMOTIONS_SENSOR_LAYOUT
)
ROBOT_EMOTIONS_CHANNELS_PER_SENSOR = 6
ROBOT_EMOTIONS_TOTAL_CHANNELS = int(len(ROBOT_EMOTIONS_SENSOR_ORDER) * ROBOT_EMOTIONS_CHANNELS_PER_SENSOR)
ROBOT_EMOTIONS_DEFAULT_SENSOR_CHANNEL_SLICES: Dict[str, List[int]] = {
    "waist": [0, 6],
    "head_left": [6, 12],
    "right_forearm": [12, 18],
    "left_forearm": [18, 24],
}
ROBOT_EMOTIONS_SINGLE_SENSOR_ALIASES: Dict[str, str] = {
    "forearm_right": "right_forearm",
    "right_arm": "right_forearm",
    "r_forearm": "right_forearm",
    "forearm_left": "left_forearm",
    "left_arm": "left_forearm",
    "l_forearm": "left_forearm",
    "head": "head_left",
    "trunk": "waist",
    "center_waist": "waist",
}
ROBOT_EMOTIONS_CHANNEL_AXIS_ORDER: Tuple[str, ...] = ("ax", "ay", "az", "gx", "gy", "gz")


def _active_channel_axis_order(
    *,
    use_acc: bool,
    use_gyro: bool,
) -> List[str]:
    order: List[str] = []
    if bool(use_acc):
        order.extend(["ax", "ay", "az"])
    if bool(use_gyro):
        order.extend(["gx", "gy", "gz"])
    if len(order) == 0:
        raise ValueError("At least one IMU channel group must be enabled (acc and/or gyro).")
    return order


def _select_imu_channel_groups(
    imu_data: np.ndarray,
    *,
    use_acc: bool,
    use_gyro: bool,
) -> np.ndarray:
    if imu_data.ndim != 2:
        raise ValueError(f"imu_data must be [T,C], got shape {imu_data.shape}")
    if bool(use_acc) and bool(use_gyro):
        return imu_data
    if (not bool(use_acc)) and (not bool(use_gyro)):
        raise ValueError("At least one IMU channel group must be enabled (acc and/or gyro).")

    channels = int(imu_data.shape[1])
    if channels % ROBOT_EMOTIONS_CHANNELS_PER_SENSOR != 0:
        raise ValueError(
            "RobotEmotions channel-group ablation expects channels multiple of 6, "
            f"got C={channels}"
        )

    per_sensor_indices: List[int] = []
    for start in range(0, channels, ROBOT_EMOTIONS_CHANNELS_PER_SENSOR):
        if bool(use_acc):
            per_sensor_indices.extend([start + 0, start + 1, start + 2])
        if bool(use_gyro):
            per_sensor_indices.extend([start + 3, start + 4, start + 5])
    return imu_data[:, per_sensor_indices]


def _canonical_robot_emotions_sensor_name(sensor_name: str) -> str:
    candidate = str(sensor_name).strip().lower()
    if candidate in ROBOT_EMOTIONS_SINGLE_SENSOR_ALIASES:
        candidate = ROBOT_EMOTIONS_SINGLE_SENSOR_ALIASES[candidate]
    return candidate


def _normalize_sensor_channel_slices(
    channel_slices: Optional[Mapping[str, Sequence[int]]],
) -> Dict[str, List[int]]:
    normalized: Dict[str, List[int]] = {
        str(name): [int(x) for x in values]
        for name, values in ROBOT_EMOTIONS_DEFAULT_SENSOR_CHANNEL_SLICES.items()
    }
    if channel_slices is None:
        return normalized
    for raw_name, raw_values in channel_slices.items():
        canonical_name = _canonical_robot_emotions_sensor_name(str(raw_name))
        values = [int(x) for x in list(raw_values)]
        if len(values) == 0:
            raise ValueError(
                f"Empty channel slice for RobotEmotions sensor '{raw_name}'."
            )
        normalized[canonical_name] = values
    return normalized


def resolve_robot_emotions_sensor_indices(
    *,
    sensor_name: str,
    channel_slices: Optional[Mapping[str, Sequence[int]]] = None,
    total_channels: int = ROBOT_EMOTIONS_TOTAL_CHANNELS,
    channels_per_sensor: int = ROBOT_EMOTIONS_CHANNELS_PER_SENSOR,
) -> Tuple[str, List[int]]:
    canonical_name = _canonical_robot_emotions_sensor_name(sensor_name)
    normalized_slices = _normalize_sensor_channel_slices(channel_slices)
    if canonical_name not in normalized_slices:
        raise ValueError(
            "Unknown RobotEmotions single sensor "
            f"'{sensor_name}'. Available names: {sorted(normalized_slices.keys())}"
        )

    raw_values = [int(x) for x in normalized_slices[canonical_name]]
    if len(raw_values) == 2 and raw_values[1] > raw_values[0]:
        indices = list(range(int(raw_values[0]), int(raw_values[1])))
    else:
        indices = [int(x) for x in raw_values]
    if len(indices) != int(channels_per_sensor):
        raise ValueError(
            "RobotEmotions single-sensor mapping must produce exactly "
            f"{channels_per_sensor} channels. Got {len(indices)} for '{canonical_name}'."
        )
    if len(set(indices)) != len(indices):
        raise ValueError(
            f"RobotEmotions single-sensor mapping for '{canonical_name}' has duplicate indices: {indices}"
        )
    min_idx = min(indices)
    max_idx = max(indices)
    if min_idx < 0 or max_idx >= int(total_channels):
        raise ValueError(
            f"RobotEmotions single-sensor indices out of bounds [0,{int(total_channels)-1}]: {indices}"
        )
    return canonical_name, [int(x) for x in indices]


def slice_robot_emotions_single_sensor(
    imu_data: np.ndarray,
    *,
    sensor_name: str,
    channel_slices: Optional[Mapping[str, Sequence[int]]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if imu_data.ndim != 2:
        raise ValueError(f"imu_data must be [T,C], got shape {imu_data.shape}")
    canonical_name, indices = resolve_robot_emotions_sensor_indices(
        sensor_name=sensor_name,
        channel_slices=channel_slices,
        total_channels=int(imu_data.shape[1]),
        channels_per_sensor=ROBOT_EMOTIONS_CHANNELS_PER_SENSOR,
    )
    selected = imu_data[:, indices]
    details = {
        "selected_sensor": str(canonical_name),
        "selected_channel_indices": [int(x) for x in indices],
        "channel_axis_order": list(ROBOT_EMOTIONS_CHANNEL_AXIS_ORDER),
    }
    return selected, details


def robot_emotions_to_sensor_set(imu_data: np.ndarray) -> np.ndarray:
    """
    Convert IMU [T,C] into [S,T,6] when C is multiple of 6.
    """
    if imu_data.ndim != 2:
        raise ValueError(f"imu_data must be [T,C], got shape {imu_data.shape}")
    channels = int(imu_data.shape[1])
    if channels % ROBOT_EMOTIONS_CHANNELS_PER_SENSOR != 0:
        raise ValueError(
            "RobotEmotions set-mode scaffold expects C multiple of 6, "
            f"got C={channels}"
        )
    sensors = channels // ROBOT_EMOTIONS_CHANNELS_PER_SENSOR
    return np.transpose(
        imu_data.reshape(int(imu_data.shape[0]), sensors, ROBOT_EMOTIONS_CHANNELS_PER_SENSOR),
        (1, 0, 2),
    )


def robot_emotions_block_energy(imu_data: np.ndarray) -> Dict[str, float]:
    if imu_data.ndim != 2:
        raise ValueError(f"imu_data must be [T,C], got shape {imu_data.shape}")
    if int(imu_data.shape[1]) != ROBOT_EMOTIONS_TOTAL_CHANNELS:
        raise ValueError(
            f"Expected RobotEmotions full layout with {ROBOT_EMOTIONS_TOTAL_CHANNELS} channels, got {imu_data.shape[1]}"
        )
    out: Dict[str, float] = {}
    for sensor_name in ROBOT_EMOTIONS_SENSOR_ORDER:
        start, end = ROBOT_EMOTIONS_DEFAULT_SENSOR_CHANNEL_SLICES[sensor_name]
        sensor_block = np.asarray(imu_data[:, int(start):int(end)], dtype=np.float64)
        out[str(sensor_name)] = float(np.sqrt(np.mean(np.square(sensor_block)))) if sensor_block.size > 0 else 0.0
    return out


class RobotEmotionsDataset(Dataset):
    """
    RobotEmotions dataset parser.
    
    Structure:
    - RobotEmotions/{10ms,30ms}/UserX/TagY/
      - ESP_X_Y.csv (IMU data for 4 sensors)
      - TAG_Y.mp4 (video)
    
    IMU CSV: timestamp, acc_X_1, acc_Y_1, acc_Z_1, gyro_X_1, gyro_Y_1, gyro_Z_1, ...,
             acc_X_4, acc_Y_4, acc_Z_4, gyro_X_4, gyro_Y_4, gyro_Z_4, ...
    
    4 sensors: Tag1=left_arm, Tag2=right_arm, Tag3=head, Tag4=trunk (example mapping)
    
    NOTE:
    This class stores complete participant/protocol metadata for both 10ms and 30ms
    domains. Phase 1 training still consumes only the fields already used previously
    (IMU, video path/indices, sensor_metas, core metadata).
    """

    # Full participant table provided for both domains.
    # `collection_marker` keeps the original marker from the protocol sheet (e.g., "x").
    PARTICIPANTS_BY_DOMAIN: Dict[str, Dict[int, Dict[str, Any]]] = {
        "30ms": {
            1: {"name": "Pasquale", "age": 24, "gender": "M"},
            2: {"name": "Luigi", "age": 26, "gender": "M"},
            3: {"name": "Alessandro", "age": 24, "gender": "M"},
            4: {"name": "Carmine", "age": 24, "gender": "M"},
            5: {"name": "Andrea", "age": 24, "gender": "M"},
            6: {"name": "Rosaria", "age": 24, "gender": "F"},
            8: {"name": "Sara", "age": 26, "gender": "F"}
        },
        "10ms": {
            2: {"name": "Fulvio", "age": 24, "gender": "M"},
            3: {"name": "Manuel", "age": 23, "gender": "M"},
            4: {"name": "Andrea", "age": 24, "gender": "M"},
            5: {"name": "Gennaro", "age": 24, "gender": "M"}
        }
    }

    # Full protocol table.
    # In practice, action lookup is domain-specific by tag:
    # e.g., 10ms/Tag1 -> "Standing", 30ms/Tag1 -> "Standing".
    PROTOCOL_ROWS: List[Dict[str, Any]] = [
        {
            "emotion": "Neutrality",
            "action": "Standing",
            "stimulus": "N.A.",
            "stimulus_details": "N.A.",
            "tag_30ms": 1,
            "tag_10ms": 1,
            "notes": "",
        },
        {
            "emotion": "Neutrality",
            "action": "Sitting",
            "stimulus": "N.A.",
            "stimulus_details": "N.A.",
            "tag_30ms": 2,
            "tag_10ms": 2,
            "notes": "",
        },
        {
            "emotion": "Neutrality",
            "action": "Walking + arms outstretched",
            "stimulus": "N.A.",
            "stimulus_details": "N.A.",
            "tag_30ms": 14,
            "tag_10ms": 3,
            "notes": "",
        },
        {
            "emotion": "Sadness",
            "action": "Slow walking + free arm movement",
            "stimulus": "AUDIO",
            "stimulus_details": "Have the participant choose a sad song",
            "tag_30ms": 15,
            "tag_10ms": 4,
            "notes": "",
        },
        {
            "emotion": "Sadness",
            "action": "Sitting + free arm movement",
            "stimulus": "VISUAL",
            "stimulus_details": "Show/have the participant choose sad video clips",
            "tag_30ms": 7,
            "tag_10ms": 5,
            "notes": "",
        },
        {
            "emotion": "Sadness",
            "action": "Free arm movement + sitting",
            "stimulus": "AUTOBIOGRAPHICAL RECALL",
            "stimulus_details": "Ask the participant to remember a sad episode",
            "tag_30ms": 8,
            "tag_10ms": 6,
            "notes": "",
        },
        {
            "emotion": "Sadness",
            "action": "Free arm movement + standing",
            "stimulus": "AUTOBIOGRAPHICAL RECALL",
            "stimulus_details": "Ask the participant to remember a sad episode",
            "tag_30ms": 17,
            "tag_10ms": 7,
            "notes": "",
        },
        {
            "emotion": "Happiness",
            "action": "Fast walking + free arm movement",
            "stimulus": "AUDIO",
            "stimulus_details": "Have the participant choose a song they like",
            "tag_30ms": 16,
            "tag_10ms": 8,
            "notes": "",
        },
        {
            "emotion": "Happiness",
            "action": "Sitting + free arm movement",
            "stimulus": "VISUAL",
            "stimulus_details": "Show/have the participant choose happy video clips",
            "tag_30ms": 5,
            "tag_10ms": 9,
            "notes": "",
        },
        {
            "emotion": "Happiness",
            "action": "Gesturing with arms raised + sitting",
            "stimulus": "AUTOBIOGRAPHICAL RECALL / SIMULATING A CONVERSATION",
            "stimulus_details": "Ask the participant to recall a happy episode/funny anecdote",
            "tag_30ms": 3,
            "tag_10ms": 10,
            "notes": "",
        },
        {
            "emotion": "Happiness",
            "action": "Gesturing with arms raised + raised",
            "stimulus": "AUTOBIOGRAPHICAL RECALL / SIMULATING A CONVERSATION",
            "stimulus_details": "Ask the participant to imagine a happy episode/funny anecdote",
            "tag_30ms": 4,
            "tag_10ms": 11,
            "notes": "",
        },
        {
            "emotion": "Anger",
            "action": "Standing + free arm movement",
            "stimulus": "AUTOBIOGRAPHICAL RECALL / SIMULATING AN UNPLEASANT CONVERSATION",
            "stimulus_details": (
                "Ask the participant to imagine an unfair episode towards him/her/"
                "a moment of argument"
            ),
            "tag_30ms": 18,
            "tag_10ms": 12,
            "notes": "",
        },
        {
            "emotion": "Fear",
            "action": "Bounce + free arm movement",
            "stimulus": "AUDIOVISUAL/ SIMULATION",
            "stimulus_details": (
                "Show the participant an unexpected, scary video/"
                "Ask the participant to simulate the action"
            ),
            "tag_30ms": 9,
            "tag_10ms": 14,
            "notes": "30ms table indicates '9 - sitting'",
        },
        {
            "emotion": "Fear",
            "action": "Escape",
            "stimulus": "SIMULATION",
            "stimulus_details": (
                "Ask the participant to simulate a dangerous situation in which he turns "
                "and runs away"
            ),
            "tag_30ms": 13,
            "tag_10ms": 15,
            "notes": "",
        },
        {
            "emotion": "Boredom",
            "action": "Sitting + free arm movement",
            "stimulus": "AUDIOVISUAL/ SIMULATION",
            "stimulus_details": (
                "Show the participant a video that is not interesting to him/her/"
                "Ask the participant to simulate a moment of boredom"
            ),
            "tag_30ms": 6,
            "tag_10ms": 16,
            "notes": "",
        },
        {
            "emotion": "Boredom",
            "action": "Standing + free arm movement",
            "stimulus": "AUDIOVISUAL/ SIMULATION",
            "stimulus_details": (
                "Show the participant a video that is not interesting to him/her/"
                "Ask the participant to simulate a moment of boredom"
            ),
            "tag_30ms": 12,
            "tag_10ms": 17,
            "notes": "",
        },
        {
            "emotion": "Curiosity",
            "action": "Free movement of the still body",
            "stimulus": "SIMULATION",
            "stimulus_details": "Ask the participant to find a hidden object",
            "tag_30ms": 10,
            "tag_10ms": 18,
            "notes": "",
        },
        {
            "emotion": "Curiosity",
            "action": "Free body movement + walking",
            "stimulus": "SIMULATION",
            "stimulus_details": "Ask the participant to find a hidden object",
            "tag_30ms": 11,
            "tag_10ms": 19,
            "notes": "",
        },
    ]

    PROTOCOL_BY_TAG: Dict[str, Dict[int, Dict[str, Any]]] = {"10ms": {}, "30ms": {}}
    for _entry in PROTOCOL_ROWS:
        PROTOCOL_BY_TAG["10ms"][int(_entry["tag_10ms"])] = _entry
        PROTOCOL_BY_TAG["30ms"][int(_entry["tag_30ms"])] = _entry
    del _entry
    
    def __init__(
        self,
        root_dir: Path,
        sampling_domain: str = "10ms",  # "10ms" or "30ms"
        users: Optional[List[int]] = None,
        window_len_sec: float = 2.0,
        stride_sec: float = 0.2,
        video_fps: float = 30.0,
        target_imu_len: int = 200,
        normalize: bool = True,
        cache_dir: Optional[Path] = None,
        long_streaming_window_policy: str = "sliding",
        max_windows_per_sequence: int = 0,
        robot_emotions_single_sensor_enable: bool = False,
        robot_emotions_single_sensor_name: str = "right_forearm",
        robot_emotions_single_sensor_channel_slices: Optional[Mapping[str, Sequence[int]]] = None,
        robot_emotions_set_mode_enable: bool = False,
        strict_imu_len: bool = False,
        imu_use_acc: bool = True,
        imu_use_gyro: bool = True,
        event_bin_sec: float = 1.0,
    ):
        self.root_dir = Path(root_dir)
        if sampling_domain not in ("10ms", "30ms"):
            raise ValueError(f"Invalid sampling_domain={sampling_domain}; expected '10ms' or '30ms'.")
        self.sampling_domain = sampling_domain
        self.window_len_sec = window_len_sec
        self.stride_sec = stride_sec
        self.video_fps = video_fps
        self.target_imu_len = target_imu_len
        self.normalize = normalize
        self.cache_dir = cache_dir
        self.long_streaming_window_policy = str(long_streaming_window_policy).strip().lower()
        self.max_windows_per_sequence = int(max_windows_per_sequence)
        self.robot_emotions_single_sensor_enable = bool(robot_emotions_single_sensor_enable)
        self.robot_emotions_single_sensor_name = _canonical_robot_emotions_sensor_name(
            str(robot_emotions_single_sensor_name)
        )
        self.robot_emotions_single_sensor_channel_slices = _normalize_sensor_channel_slices(
            robot_emotions_single_sensor_channel_slices
        )
        self.robot_emotions_set_mode_enable = bool(robot_emotions_set_mode_enable)
        self.strict_imu_len = bool(strict_imu_len)
        self.imu_use_acc = bool(imu_use_acc)
        self.imu_use_gyro = bool(imu_use_gyro)
        self.event_bin_sec = float(event_bin_sec)
        if self.event_bin_sec <= 0.0:
            raise ValueError(f"event_bin_sec must be > 0, got {self.event_bin_sec}")
        self.active_channel_axis_order = _active_channel_axis_order(
            use_acc=self.imu_use_acc,
            use_gyro=self.imu_use_gyro,
        )
        self.channels_per_sensor_after_ablation = len(self.active_channel_axis_order)
        if self.robot_emotions_set_mode_enable and self.channels_per_sensor_after_ablation != 6:
            raise ValueError(
                "robot_emotions_set_mode_enable requires 6 channels per sensor. "
                "Disable set_mode or enable both acc and gyro."
            )
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.selected_single_sensor_indices: List[int] = []
        self.selected_single_sensor_details: Optional[Dict[str, Any]] = None
        self.available_sensor_names: List[str] = list(ROBOT_EMOTIONS_SENSOR_ORDER)
        self.selected_sensor_names: List[str] = list(ROBOT_EMOTIONS_SENSOR_ORDER)
        if self.robot_emotions_single_sensor_enable:
            canonical_name, selected_indices = resolve_robot_emotions_sensor_indices(
                sensor_name=self.robot_emotions_single_sensor_name,
                channel_slices=self.robot_emotions_single_sensor_channel_slices,
                total_channels=ROBOT_EMOTIONS_TOTAL_CHANNELS,
                channels_per_sensor=ROBOT_EMOTIONS_CHANNELS_PER_SENSOR,
            )
            self.robot_emotions_single_sensor_name = str(canonical_name)
            self.selected_single_sensor_indices = [int(x) for x in selected_indices]
            self.selected_single_sensor_details = {
                "selected_sensor": str(canonical_name),
                "selected_channel_indices": [int(x) for x in selected_indices],
                "channel_axis_order": list(self.active_channel_axis_order),
            }
            self.selected_sensor_names = [str(canonical_name)]
        self.output_imu_channels = (
            self.channels_per_sensor_after_ablation
            if self.robot_emotions_single_sensor_enable
            else len(self.selected_sensor_names) * self.channels_per_sensor_after_ablation
        )

        # Keep action mapping ready at initialization (UTD-MHAD style):
        # domain-specific tag number -> action label.
        self.action_by_tag: Dict[int, str] = {
            int(tag_number): str(protocol_row["action"])
            for tag_number, protocol_row in self.PROTOCOL_BY_TAG[self.sampling_domain].items()
            if protocol_row.get("action") is not None
        }
        
        if users is None:
            users = sorted(self.PARTICIPANTS_BY_DOMAIN.get(self.sampling_domain, {}).keys())
            if not users:
                users = [1, 2, 3, 4, 5, 6, 8]
        self.users = users
        
        self.samples = []
        
        self._scan_dataset()
        if self.normalize:
            self.fit_normalization_from_sample_indices(list(range(len(self.samples))))

    def get_sensor_selection_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "single_sensor_enable": bool(self.robot_emotions_single_sensor_enable),
            "single_sensor_name": (
                str(self.robot_emotions_single_sensor_name)
                if self.robot_emotions_single_sensor_enable
                else None
            ),
            "selected_channel_indices": (
                [int(x) for x in self.selected_single_sensor_indices]
                if self.robot_emotions_single_sensor_enable
                else None
            ),
            "selected_channel_axis_order": list(self.active_channel_axis_order),
            "available_sensor_names": list(self.available_sensor_names),
            "active_sensor_names": list(self.selected_sensor_names),
            "imu_use_acc": bool(self.imu_use_acc),
            "imu_use_gyro": bool(self.imu_use_gyro),
            "set_mode_enable": bool(self.robot_emotions_set_mode_enable),
            "set_mode_shape_hint": (
                [len(self.selected_sensor_names), int(self.target_imu_len), self.channels_per_sensor_after_ablation]
                if self.robot_emotions_set_mode_enable
                else None
            ),
        }
        if len(self.samples) > 0:
            first_sample = self.samples[0]
            summary["imu_shape_before_selection"] = first_sample.get("imu_shape_before_selection")
            summary["imu_shape_after_selection"] = first_sample.get("imu_shape_after_selection")
        return summary

    @staticmethod
    def _extract_sensor_id(column_name: str) -> Optional[int]:
        match = re.search(r"(\d+)", str(column_name))
        if match is None:
            return None
        return int(match.group(1))

    @staticmethod
    def _collect_axis_columns(df: pd.DataFrame, prefix: str) -> Dict[Tuple[int, str], str]:
        """
        Collect columns by (sensor_id, axis) from arbitrary CSV headers.
        """
        out: Dict[Tuple[int, str], str] = {}
        for column in df.columns:
            lower = str(column).strip().lower()
            if prefix not in lower:
                continue
            sensor_id = RobotEmotionsDataset._extract_sensor_id(lower)
            if sensor_id is None:
                continue

            axis = None
            if re.search(r"(?:^|[_\-\s])x(?:$|[_\-\s])", lower):
                axis = "x"
            elif re.search(r"(?:^|[_\-\s])y(?:$|[_\-\s])", lower):
                axis = "y"
            elif re.search(r"(?:^|[_\-\s])z(?:$|[_\-\s])", lower):
                axis = "z"
            else:
                # Fallback for names like "acc1x"/"gyro2z"
                if lower.endswith("x"):
                    axis = "x"
                elif lower.endswith("y"):
                    axis = "y"
                elif lower.endswith("z"):
                    axis = "z"

            if axis is None:
                continue
            out[(sensor_id, axis)] = column
        return out

    @staticmethod
    def _parse_tag_number(tag_id: str) -> int:
        """Parse tag name like 'Tag7' or 'TAG19' into integer ID."""
        match = re.match(r"(?i)^tag(\d+)$", tag_id.strip())
        if not match:
            raise ValueError(f"Invalid tag directory name: {tag_id}")
        return int(match.group(1))

    @classmethod
    def get_user_profile(cls, domain: str, user_id: int) -> Dict[str, Any]:
        """Return participant metadata for (domain, user_id)."""
        profile = cls.PARTICIPANTS_BY_DOMAIN.get(domain, {}).get(user_id)
        if profile is None:
            return {
                "name": None,
                "age": None,
                "gender": None,
                "collection_marker": "",
            }
        return dict(profile)

    @classmethod
    def get_protocol_info_by_tag(cls, domain: str, tag_number: int) -> Optional[Dict[str, Any]]:
        """Return protocol metadata for a domain-specific tag number."""
        info = cls.PROTOCOL_BY_TAG.get(domain, {}).get(tag_number)
        return dict(info) if info is not None else None

    @staticmethod
    def _resolve_existing_file(candidates: List[Path]) -> Optional[Path]:
        for file_path in candidates:
            if file_path.exists():
                return file_path
        return None
    
    def _scan_dataset(self):
        """Scan dataset and create window samples."""
        domain_path = self.root_dir / self.sampling_domain
        
        if not domain_path.exists():
            print(f"Warning: RobotEmotions path not found: {domain_path}")
            return
        
        for user_id in self.users:
            user_dir = domain_path / f"User{user_id}"
            if not user_dir.exists():
                continue
            
            # Find all tag directories
            tag_dirs = sorted([d for d in user_dir.iterdir() if d.is_dir() and d.name.startswith("Tag")])
            
            for tag_dir in tag_dirs:
                tag_id = tag_dir.name  # e.g., "Tag1"
                try:
                    tag_number = self._parse_tag_number(tag_id)
                except ValueError:
                    continue
                
                # Find IMU and video files
                imu_file = self._resolve_existing_file([
                    tag_dir / f"ESP_{user_id}_{tag_number}.csv",
                    tag_dir / f"ESP_{user_id}_{tag_id[-1]}.csv",
                ])
                video_file = self._resolve_existing_file([
                    tag_dir / f"{tag_id.upper()}.mp4",
                    tag_dir / f"TAG_{tag_number}.mp4",
                    tag_dir / f"{tag_id}.mp4",
                ])
                
                if imu_file is None or video_file is None:
                    continue
                
                # Load IMU data
                try:
                    imu_data, imu_timestamps = self._load_imu(imu_file)
                except Exception as e:
                    print(f"Error loading {imu_file}: {e}")
                    continue
                acc_idx, gyro_idx = repeat_sensor_channel_slices(
                    total_channels=int(imu_data.shape[1]),
                    channels_per_sensor=ROBOT_EMOTIONS_CHANNELS_PER_SENSOR,
                )
                imu_data = standardize_imu_units(
                    torch.from_numpy(imu_data),
                    dataset_name=f"robot_emotions_{self.sampling_domain}",
                    acc_slice=acc_idx,
                    gyro_slice=gyro_idx,
                ).detach().cpu().numpy().astype(np.float32, copy=False)
                imu_data_before_selection = imu_data
                if self.robot_emotions_single_sensor_enable:
                    imu_data, _ = slice_robot_emotions_single_sensor(
                        imu_data_before_selection,
                        sensor_name=self.robot_emotions_single_sensor_name,
                        channel_slices=self.robot_emotions_single_sensor_channel_slices,
                    )
                imu_data = _select_imu_channel_groups(
                    imu_data,
                    use_acc=self.imu_use_acc,
                    use_gyro=self.imu_use_gyro,
                )
                selected_sensor_names = list(self.selected_sensor_names)
                
                # Get video info
                video_info = self._get_video_info(video_file)
                if video_info is None:
                    continue
                
                num_frames, fps, duration = video_info
                video_timestamps = np.arange(num_frames) / fps
                
                # Create windows
                windows = create_sliding_windows(
                    imu_data, video_timestamps, imu_timestamps,
                    self.window_len_sec, self.stride_sec, fps,
                    return_imu_indices=True,
                    long_streaming_window_policy=self.long_streaming_window_policy,
                    max_windows_per_sequence=self.max_windows_per_sequence,
                )

                user_profile = self.get_user_profile(self.sampling_domain, user_id)
                protocol_info = self.get_protocol_info_by_tag(self.sampling_domain, tag_number)
                action = self.action_by_tag.get(tag_number, "UNK_ACTION")
                
                for imu_window, video_indices, start_t, end_t, imu_indices in windows:
                    # Sensor metas follow selected channel mode (4x6 default, or 1x6 compat mode).
                    sensor_metas = create_sensor_metas_robot_emotions(
                        domain=self.sampling_domain,
                        present_sensors=[True] * len(selected_sensor_names),
                        sensor_names=selected_sensor_names,
                        num_channels_per_sensor=int(self.channels_per_sensor_after_ablation),
                    )
                    imu_i0 = int(imu_indices[0]) if len(imu_indices) > 0 else -1
                    imu_i1 = int(imu_indices[-1]) if len(imu_indices) > 0 else -1
                    vid_i0 = int(video_indices[0]) if len(video_indices) > 0 else -1
                    vid_i1 = int(video_indices[-1]) if len(video_indices) > 0 else -1
                    sample_id = (
                        f"robot_emotions_{self.sampling_domain}:u{user_id}:tag{tag_number}:"
                        f"{float(start_t):.4f}-{float(end_t):.4f}:{vid_i0}-{vid_i1}"
                    )
                    source_sequence_id = f"robot_emotions_{self.sampling_domain}:u{user_id}:tag{tag_number}"
                    event_uid = build_event_uid(
                        sequence_uid=source_sequence_id,
                        window_start_time_sec=float(start_t),
                        event_bin_sec=float(self.event_bin_sec),
                    )
                    
                    self.samples.append({
                        "dataset": f"robot_emotions_{self.sampling_domain}",
                        "user": user_id,
                        "tag": tag_id,
                        "tag_number": tag_number,
                        "video_path": str(video_file),
                        "imu_data": imu_window,
                        "video_indices": video_indices,
                        "start_time": start_t,
                        "end_time": end_t,
                        "imu_index_start": imu_i0,
                        "imu_index_end": imu_i1,
                        "video_index_start": vid_i0,
                        "video_index_end": vid_i1,
                        "imu_num_samples_raw": int(len(imu_window)),
                        "video_num_frames_total": int(num_frames),
                        "video_fps": float(fps),
                        "video_duration_sec": float(duration),
                        "offset_sec": 0.0,
                        "offset_source": "none",
                        "sample_id": sample_id,
                        "source_sequence_id": source_sequence_id,
                        "event_uid": event_uid,
                        "window_idx": len(self.samples),
                        "sensor_metas": sensor_metas,
                        "imu_shape_before_selection": list(imu_data_before_selection.shape),
                        "imu_shape_after_selection": list(imu_window.shape),
                        "robot_emotions_single_sensor_enable": bool(self.robot_emotions_single_sensor_enable),
                        "robot_emotions_selected_sensor": (
                            str(self.robot_emotions_single_sensor_name)
                            if self.robot_emotions_single_sensor_enable
                            else None
                        ),
                        "robot_emotions_selected_channel_indices": (
                            [int(x) for x in self.selected_single_sensor_indices]
                            if self.robot_emotions_single_sensor_enable
                            else None
                        ),
                        "robot_emotions_set_mode_enable": bool(self.robot_emotions_set_mode_enable),
                        "imu_use_acc": bool(self.imu_use_acc),
                        "imu_use_gyro": bool(self.imu_use_gyro),
                        "imu_channel_axis_order": list(self.active_channel_axis_order),
                        "action": action,
                        "user_profile": dict(user_profile),
                        "protocol_info": dict(protocol_info) if protocol_info is not None else None,
                    })
    
    def _load_imu(self, imu_file: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load IMU CSV file for RobotEmotions.
        
        Format: timestamp, acc_X_1, acc_Y_1, acc_Z_1, gyro_X_1, ..., acc_X_4, ..., gyro_Z_4
        Total columns: 1 (timestamp) + 4 sensors * 6 channels = 25+ columns
        """
        df = pd.read_csv(imu_file)
        if df.shape[1] < 2:
            raise ValueError(f"Invalid RobotEmotions CSV with insufficient columns: {imu_file}")

        # Extract timestamps (convert ms to seconds)
        timestamps = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float64) / 1000.0
        if np.isnan(timestamps).all():
            raise ValueError(f"Could not parse timestamp column: {imu_file}")
        valid_t = ~np.isnan(timestamps)
        timestamps = timestamps[valid_t]
        timestamps = timestamps - timestamps[0]  # Start from 0

        df_valid = df.loc[valid_t].copy()

        acc_cols = self._collect_axis_columns(df_valid, prefix="acc")
        gyro_cols = self._collect_axis_columns(df_valid, prefix="gyro")

        sensor_ids = sorted({sid for sid, _ in acc_cols.keys()}.intersection({sid for sid, _ in gyro_cols.keys()}))
        imu_parts: List[np.ndarray] = []

        for sensor_id in sensor_ids:
            sensor_axes = []
            for prefix_map in (acc_cols, gyro_cols):
                for axis in ("x", "y", "z"):
                    key = (sensor_id, axis)
                    if key not in prefix_map:
                        sensor_axes = []
                        break
                    values = pd.to_numeric(df_valid[prefix_map[key]], errors="coerce").to_numpy(dtype=np.float64)
                    sensor_axes.append(values)
                if len(sensor_axes) not in (3, 6):
                    break
            if len(sensor_axes) == 6:
                imu_parts.append(np.stack(sensor_axes, axis=1))

        if len(imu_parts) > 0:
            imu_data = np.concatenate(imu_parts, axis=1)
        else:
            # Fallback for unnamed/irregular CSVs: keep first 24 numeric channels after timestamp.
            numeric = df_valid.apply(pd.to_numeric, errors="coerce")
            numeric_values = numeric.iloc[:, 1:].to_numpy(dtype=np.float64)
            if numeric_values.shape[1] < ROBOT_EMOTIONS_TOTAL_CHANNELS:
                raise ValueError(
                    "RobotEmotions CSV has only "
                    f"{numeric_values.shape[1]} numeric channels after timestamp, "
                    f"expected >= {ROBOT_EMOTIONS_TOTAL_CHANNELS}: {imu_file}"
                )
            imu_data = numeric_values[:, :ROBOT_EMOTIONS_TOTAL_CHANNELS]

        if imu_data.shape[1] != ROBOT_EMOTIONS_TOTAL_CHANNELS:
            raise ValueError(
                "Expected RobotEmotions full IMU channels "
                f"({ROBOT_EMOTIONS_TOTAL_CHANNELS}=4x6), got {imu_data.shape[1]} in {imu_file}"
            )

        # Filter out invalid rows
        valid_mask = ~np.isnan(imu_data).any(axis=1)
        imu_data = imu_data[valid_mask]
        timestamps = timestamps[valid_mask]

        return imu_data.astype(np.float32), timestamps.astype(np.float32)
    
    def _get_video_info(self, video_file: Path) -> Optional[Tuple[int, float, float]]:
        """Get video metadata."""
        try:
            return read_video_info(video_file=video_file, default_fps=self.video_fps)
        except Exception as e:
            print(f"Error reading video {video_file}: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Process IMU data
        imu_data = sample["imu_data"].copy()  # [T, C]
        
        # Resample to target length
        timestamps = np.linspace(0, self.window_len_sec, len(imu_data))
        if len(imu_data) != self.target_imu_len:
            imu_data = resample_imu_uniform(
                imu_data, timestamps, target_length=self.target_imu_len, causal=True
            )
        if self.strict_imu_len and int(imu_data.shape[0]) != int(self.target_imu_len):
            raise RuntimeError(
                "RobotEmotions strict_imu_len expected "
                f"T={self.target_imu_len}, got {imu_data.shape[0]}"
            )

        if self.normalize and self.mean is not None and self.std is not None:
            imu_data = (imu_data - self.mean) / self.std
        
        imu_tensor = torch.from_numpy(imu_data).float()  # [T, C]
        imu_set_tensor: Optional[torch.Tensor] = None
        if self.robot_emotions_set_mode_enable:
            imu_set = robot_emotions_to_sensor_set(imu_data)
            imu_set_tensor = torch.from_numpy(imu_set).float()  # [S, T, 6]
        
        # Video indices for teacher embedding cache
        video_indices = torch.from_numpy(sample["video_indices"]).long()
        
        # Sensor metadata
        sensor_metas = sample["sensor_metas"]
        
        out = {
            "imu": imu_tensor,
            "video_path": sample["video_path"],
            "video_indices": video_indices,
            "sensor_metas": sensor_metas,
            "metadata": {
                "dataset": sample["dataset"],
                "sampling_domain": self.sampling_domain,
                "user": sample["user"],
                "tag": sample["tag"],
                "tag_number": sample["tag_number"],
                "window_idx": sample["window_idx"],
                "start_time": sample["start_time"],
                "end_time": sample["end_time"],
                "imu_index_start": sample.get("imu_index_start", -1),
                "imu_index_end": sample.get("imu_index_end", -1),
                "video_index_start": sample.get("video_index_start", -1),
                "video_index_end": sample.get("video_index_end", -1),
                "imu_num_samples_raw": sample.get("imu_num_samples_raw", int(sample["imu_data"].shape[0])),
                "video_num_frames_total": sample.get("video_num_frames_total", -1),
                "video_fps": sample.get("video_fps", float(self.video_fps)),
                "video_duration_sec": sample.get("video_duration_sec", -1.0),
                "offset_sec": sample.get("offset_sec", 0.0),
                "offset_source": sample.get("offset_source", "none"),
                "sample_id": sample.get("sample_id", f"robot_emotions_{self.sampling_domain}:{sample['window_idx']}"),
                "source_sequence_id": sample.get(
                    "source_sequence_id",
                    f"robot_emotions_{self.sampling_domain}:{sample['user']}:{sample['tag_number']}",
                ),
                "event_uid": sample.get("event_uid"),
                "imu_shape_before_selection": sample.get("imu_shape_before_selection"),
                "imu_shape_after_selection": sample.get("imu_shape_after_selection"),
                "robot_emotions_single_sensor_enable": sample.get("robot_emotions_single_sensor_enable", False),
                "robot_emotions_selected_sensor": sample.get("robot_emotions_selected_sensor"),
                "robot_emotions_selected_channel_indices": sample.get("robot_emotions_selected_channel_indices"),
                "robot_emotions_set_mode_enable": sample.get("robot_emotions_set_mode_enable", False),
                "imu_use_acc": sample.get("imu_use_acc", bool(self.imu_use_acc)),
                "imu_use_gyro": sample.get("imu_use_gyro", bool(self.imu_use_gyro)),
                "imu_channel_axis_order": sample.get("imu_channel_axis_order", list(self.active_channel_axis_order)),
                "action": sample["action"],
                "participant": dict(sample["user_profile"]),
                "protocol": (
                    dict(sample["protocol_info"]) if sample["protocol_info"] is not None else None
                ),
            }
        }
        if imu_set_tensor is not None:
            out["imu_set"] = imu_set_tensor
        return out

    def fit_normalization_from_sample_indices(self, sample_indices: List[int]) -> None:
        if len(self.samples) == 0:
            self.mean = np.zeros(int(self.output_imu_channels), dtype=np.float32)
            self.std = np.ones(int(self.output_imu_channels), dtype=np.float32)
            self.normalize = True
            return

        if len(sample_indices) == 0:
            channels = int(self.samples[0]["imu_data"].shape[1])
            self.mean = np.zeros(channels, dtype=np.float32)
            self.std = np.ones(channels, dtype=np.float32)
            self.normalize = True
            return

        channel_sum = None
        channel_sq_sum = None
        sample_count = 0

        for idx in sample_indices:
            if idx < 0 or idx >= len(self.samples):
                raise IndexError(f"sample index out of range: {idx}")
            values = self.samples[idx]["imu_data"]
            if channel_sum is None:
                channels = int(values.shape[1])
                channel_sum = np.zeros(channels, dtype=np.float64)
                channel_sq_sum = np.zeros(channels, dtype=np.float64)
            channel_sum += values.sum(axis=0)
            channel_sq_sum += np.square(values).sum(axis=0)
            sample_count += int(values.shape[0])

        if sample_count <= 0 or channel_sum is None or channel_sq_sum is None:
            channels = int(self.samples[0]["imu_data"].shape[1])
            self.mean = np.zeros(channels, dtype=np.float32)
            self.std = np.ones(channels, dtype=np.float32)
            self.normalize = True
            return

        mean = channel_sum / sample_count
        var = channel_sq_sum / sample_count - np.square(mean)
        std = np.sqrt(np.maximum(var, 1e-8))
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.normalize = True


def get_robot_emotions_stats(root_dir: Path, sampling_domain: str, users: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalization statistics for RobotEmotions training set."""
    dataset = RobotEmotionsDataset(root_dir, sampling_domain=sampling_domain, users=users, normalize=False)
    
    all_imu = []
    for i in range(len(dataset)):
        all_imu.append(dataset[i]["imu"].numpy())
    
    all_imu = np.concatenate(all_imu, axis=0)  # [N*T, C]
    mean = np.mean(all_imu, axis=0)
    std = np.std(all_imu, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    
    return mean, std
