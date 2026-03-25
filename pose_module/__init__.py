"""Pose pipeline package."""

from __future__ import annotations

__all__ = ["run_pose2d_pipeline"]


def __getattr__(name: str):
    if name == "run_pose2d_pipeline":
        from .pipeline import run_pose2d_pipeline

        return run_pose2d_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
