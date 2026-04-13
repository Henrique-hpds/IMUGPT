"""Standalone RobotEmotions video-description package backed by Qwen3-VL."""

from .anchor_catalog import build_anchor_catalog
from .cli import describe_videos
from .window_descriptions import describe_windows

__all__ = ["build_anchor_catalog", "describe_videos", "describe_windows"]
