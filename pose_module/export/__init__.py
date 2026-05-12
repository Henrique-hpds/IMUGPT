"""Export helpers for debug artifacts."""

from typing import Any

from .debug_video import (
    render_pose_overlay_video,
    render_pose3d_side_by_side_video,
    resolve_debug_overlay_path,
    resolve_debug_overlay_variant_path,
)


def export_pose_sequence3d_to_bvh(*args: Any, **kwargs: Any) -> Any:
    from .bvh import export_pose_sequence3d_to_bvh as _export_pose_sequence3d_to_bvh

    return _export_pose_sequence3d_to_bvh(*args, **kwargs)


def run_ik(*args: Any, **kwargs: Any) -> Any:
    from .ik_adapter import run_ik as _run_ik

    return _run_ik(*args, **kwargs)


def run_imusim(*args: Any, **kwargs: Any) -> Any:
    from .imusim_adapter import run_imusim as _run_imusim

    return _run_imusim(*args, **kwargs)

__all__ = [
    "export_pose_sequence3d_to_bvh",
    "run_ik",
    "run_imusim",
    "render_pose_overlay_video",
    "render_pose3d_side_by_side_video",
    "resolve_debug_overlay_path",
    "resolve_debug_overlay_variant_path"
]
