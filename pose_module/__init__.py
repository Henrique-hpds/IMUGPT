"""Pose pipeline package."""

from __future__ import annotations

__all__ = [
    "run_pose2d_pipeline",
    "run_pose3d_from_prompt",
    "run_pose3d_from_video",
    "run_pose3d_pipeline",
    "run_virtual_imu_pipeline",
    "generate_pose_from_prompt",
    "run_geometric_alignment",
    "generate_pose_from_video",
    "generate_virtual_imu_from_video",
    "download_required_models",
]


def __getattr__(name: str):
    if name == "run_pose2d_pipeline":
        from .pipeline import run_pose2d_pipeline

        return run_pose2d_pipeline
    if name == "run_pose3d_pipeline":
        from .pipeline import run_pose3d_pipeline

        return run_pose3d_pipeline
    if name == "run_pose3d_from_prompt":
        from .pipeline import run_pose3d_from_prompt

        return run_pose3d_from_prompt
    if name == "run_pose3d_from_video":
        from .pipeline import run_pose3d_from_video

        return run_pose3d_from_video
    if name == "run_virtual_imu_pipeline":
        from .pipeline import run_virtual_imu_pipeline

        return run_virtual_imu_pipeline
    if name == "run_geometric_alignment":
        from .imu_alignment import run_geometric_alignment

        return run_geometric_alignment
    if name == "generate_pose_from_video":
        from .pipeline import generate_pose_from_video

        return generate_pose_from_video
    if name == "generate_pose_from_prompt":
        from .pipeline import generate_pose_from_prompt

        return generate_pose_from_prompt
    if name == "generate_virtual_imu_from_video":
        from .pipeline import generate_virtual_imu_from_video

        return generate_virtual_imu_from_video
    if name == "download_required_models":
        from .download_models import download_required_models

        return download_required_models
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
