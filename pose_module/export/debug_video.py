"""Debug video artifact helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from pose_module.interfaces import MOTIONBERT_17_PARENT_INDICES


def resolve_debug_overlay_path(
    output_dir: str | Path,
    *,
    filename: str = "debug_overlay.mp4",
    enabled: bool = True,
) -> Optional[Path]:
    if not bool(enabled):
        return None
    return Path(output_dir) / str(filename)


def resolve_debug_overlay_variant_path(
    output_dir: str | Path,
    *,
    variant: str,
    enabled: bool = True,
) -> Optional[Path]:
    if not bool(enabled):
        return None
    safe_variant = str(variant).strip().replace(" ", "_")
    return Path(output_dir) / f"debug_overlay_{safe_variant}.mp4"


def render_pose_overlay_video(
    *,
    video_path: str | Path,
    output_path: str | Path,
    frame_indices: Sequence[int],
    keypoints_xy: np.ndarray,
    confidence: np.ndarray,
    joint_names: Sequence[str],
    bbox_xywh: np.ndarray | None = None,
    fps: float | None = None,
    overlay_variant: str = "raw",
) -> Optional[Path]:
    selected_frame_indices = np.asarray(frame_indices, dtype=np.int32)
    points_xy = np.asarray(keypoints_xy, dtype=np.float32)
    joint_confidence = np.asarray(confidence, dtype=np.float32)
    if points_xy.ndim != 3 or points_xy.shape[-1] != 2:
        raise ValueError("keypoints_xy must have shape [T, J, 2]")
    if joint_confidence.shape != points_xy.shape[:2]:
        raise ValueError("confidence must have shape [T, J]")
    if selected_frame_indices.shape != (points_xy.shape[0],):
        raise ValueError("frame_indices must align with keypoints first dimension")
    if points_xy.shape[0] == 0:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skeleton_edges = _resolve_skeleton_edges(joint_names)
    line_color, joint_color, bbox_color = _overlay_palette(str(overlay_variant))
    target_fps = 20.0 if fps is None or float(fps) <= 0.0 else float(fps)

    decoder = _open_ffmpeg_decoder(video_path)
    encoder = None
    try:
        selected_position = 0
        frame_index = 0
        while selected_position < len(selected_frame_indices):
            frame_rgb = _read_ppm_frame(decoder.stdout)
            if frame_rgb is None:
                break

            target_frame_index = int(selected_frame_indices[selected_position])
            if frame_index == target_frame_index:
                overlay_frame = frame_rgb.copy()
                current_bbox = None
                if bbox_xywh is not None:
                    current_bbox = np.asarray(bbox_xywh[selected_position], dtype=np.float32)
                _draw_pose_overlay(
                    overlay_frame,
                    points_xy[selected_position],
                    joint_confidence[selected_position],
                    skeleton_edges,
                    line_color=line_color,
                    joint_color=joint_color,
                    bbox_xywh=current_bbox,
                    bbox_color=bbox_color,
                )
                if encoder is None:
                    encoder = _open_ffmpeg_encoder(
                        output_path=output_path,
                        width=int(overlay_frame.shape[1]),
                        height=int(overlay_frame.shape[0]),
                        fps=float(target_fps),
                    )
                assert encoder.stdin is not None
                encoder.stdin.write(overlay_frame.astype(np.uint8, copy=False).tobytes())
                selected_position += 1
            frame_index += 1

        if selected_position != len(selected_frame_indices):
            raise RuntimeError(
                "Could not decode every selected frame for debug overlay rendering. "
                f"Rendered {selected_position} of {len(selected_frame_indices)} frames."
            )
    finally:
        if decoder.stdout is not None:
            decoder.stdout.close()
        decoder.wait()
        if encoder is not None:
            if encoder.stdin is not None:
                encoder.stdin.close()
            encoder.wait()

    if not output_path.exists():
        raise RuntimeError(f"Debug overlay video was not created at {output_path}")
    return output_path.resolve()


def _resolve_skeleton_edges(joint_names: Sequence[str]) -> list[tuple[int, int]]:
    normalized_joint_names = [str(name) for name in joint_names]
    if normalized_joint_names == [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]:
        return [
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (5, 11),
            (6, 12),
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
        ]

    joint_name_to_index = {name: index for index, name in enumerate(normalized_joint_names)}
    motionbert_expected = [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine",
        "left_knee",
        "right_knee",
        "thorax",
        "left_ankle",
        "right_ankle",
        "neck",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ]
    if all(name in joint_name_to_index for name in motionbert_expected):
        edges = []
        ordered_indices = [joint_name_to_index[name] for name in motionbert_expected]
        for child_position, parent_position in enumerate(MOTIONBERT_17_PARENT_INDICES):
            if parent_position < 0:
                continue
            edges.append((ordered_indices[parent_position], ordered_indices[child_position]))
        return edges
    return []


def _overlay_palette(overlay_variant: str) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    normalized = str(overlay_variant).strip().lower()
    if normalized == "clean":
        return (60, 200, 255), (255, 245, 140), (255, 170, 60)
    return (80, 255, 120), (255, 220, 80), (255, 120, 120)


def _open_ffmpeg_decoder(video_path: str | Path) -> subprocess.Popen:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(Path(video_path).resolve()),
        "-f",
        "image2pipe",
        "-vcodec",
        "ppm",
        "-",
    ]
    try:
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg_not_found_for_debug_render") from exc


def _open_ffmpeg_encoder(
    *,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
) -> subprocess.Popen:
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{int(width)}x{int(height)}",
        "-r",
        f"{float(fps):.6f}",
        "-i",
        "-",
        "-an",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path.resolve()),
    ]
    try:
        return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg_not_found_for_debug_render") from exc


def _read_ppm_frame(stream) -> np.ndarray | None:
    magic = _read_ppm_token(stream)
    if magic is None:
        return None
    if magic != b"P6":
        raise RuntimeError(f"Unexpected PPM magic token: {magic!r}")

    width_token = _read_ppm_token(stream)
    height_token = _read_ppm_token(stream)
    maxval_token = _read_ppm_token(stream)
    if width_token is None or height_token is None or maxval_token is None:
        raise RuntimeError("Unexpected EOF while reading PPM header")

    width = int(width_token)
    height = int(height_token)
    max_value = int(maxval_token)
    if max_value != 255:
        raise RuntimeError(f"Unsupported PPM max value: {max_value}")

    frame_size = int(width) * int(height) * 3
    frame_bytes = _read_exact(stream, frame_size)
    if frame_bytes is None:
        raise RuntimeError("Unexpected EOF while reading PPM frame payload")
    return np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3)).copy()


def _read_ppm_token(stream) -> bytes | None:
    token = bytearray()
    while True:
        raw_byte = stream.read(1)
        if raw_byte == b"":
            return None if len(token) == 0 else bytes(token)
        if raw_byte == b"#":
            stream.readline()
            continue
        if raw_byte in b" \t\r\n":
            if len(token) == 0:
                continue
            return bytes(token)
        token.extend(raw_byte)


def _read_exact(stream, num_bytes: int) -> bytes | None:
    chunks = []
    remaining = int(num_bytes)
    while remaining > 0:
        chunk = stream.read(remaining)
        if chunk == b"":
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _draw_pose_overlay(
    frame_rgb: np.ndarray,
    joints_xy: np.ndarray,
    joint_confidence: np.ndarray,
    skeleton_edges: Sequence[tuple[int, int]],
    *,
    line_color: tuple[int, int, int],
    joint_color: tuple[int, int, int],
    bbox_xywh: np.ndarray | None,
    bbox_color: tuple[int, int, int],
) -> None:
    if bbox_xywh is not None and np.isfinite(bbox_xywh).all():
        x, y, width, height = [float(value) for value in bbox_xywh[:4]]
        _draw_rectangle(
            frame_rgb,
            int(round(x)),
            int(round(y)),
            int(round(x + width)),
            int(round(y + height)),
            color=bbox_color,
            thickness=2,
        )

    valid_joint_mask = np.isfinite(joints_xy).all(axis=1) & (joint_confidence > 0.0)
    for parent_index, child_index in skeleton_edges:
        if not (valid_joint_mask[parent_index] and valid_joint_mask[child_index]):
            continue
        start_point = joints_xy[parent_index]
        end_point = joints_xy[child_index]
        _draw_line(
            frame_rgb,
            int(round(start_point[0])),
            int(round(start_point[1])),
            int(round(end_point[0])),
            int(round(end_point[1])),
            color=line_color,
            thickness=2,
        )

    for joint_index, joint_xy in enumerate(joints_xy):
        if not valid_joint_mask[joint_index]:
            continue
        _draw_circle(
            frame_rgb,
            int(round(joint_xy[0])),
            int(round(joint_xy[1])),
            radius=3,
            color=joint_color,
        )


def _draw_rectangle(
    frame_rgb: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    _draw_line(frame_rgb, x1, y1, x2, y1, color=color, thickness=thickness)
    _draw_line(frame_rgb, x2, y1, x2, y2, color=color, thickness=thickness)
    _draw_line(frame_rgb, x2, y2, x1, y2, color=color, thickness=thickness)
    _draw_line(frame_rgb, x1, y2, x1, y1, color=color, thickness=thickness)


def _draw_line(
    frame_rgb: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    num_steps = int(max(abs(x2 - x1), abs(y2 - y1), 1))
    x_values = np.linspace(x1, x2, num_steps + 1)
    y_values = np.linspace(y1, y2, num_steps + 1)
    for x_value, y_value in zip(x_values, y_values):
        _draw_circle(
            frame_rgb,
            int(round(x_value)),
            int(round(y_value)),
            radius=max(int(thickness) - 1, 1),
            color=color,
        )


def _draw_circle(
    frame_rgb: np.ndarray,
    center_x: int,
    center_y: int,
    *,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    height, width = frame_rgb.shape[:2]
    x_min = max(0, int(center_x) - int(radius))
    x_max = min(width - 1, int(center_x) + int(radius))
    y_min = max(0, int(center_y) - int(radius))
    y_max = min(height - 1, int(center_y) + int(radius))
    if x_min > x_max or y_min > y_max:
        return

    yy, xx = np.ogrid[y_min : y_max + 1, x_min : x_max + 1]
    mask = ((xx - int(center_x)) ** 2) + ((yy - int(center_y)) ** 2) <= int(radius) ** 2
    frame_rgb[y_min : y_max + 1, x_min : x_max + 1][mask] = np.asarray(color, dtype=np.uint8)
