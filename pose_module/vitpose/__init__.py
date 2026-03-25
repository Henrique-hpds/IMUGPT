"""ViTPose stage helpers and backend entrypoints."""

from __future__ import annotations

__all__ = [
    "canonicalize_pose_sequence2d",
    "load_raw_prediction_frames",
    "run_backend_job",
    "run_pose2d_backend",
    "write_pose_sequence_npz",
]


def __getattr__(name: str):
    if name in {
        "canonicalize_pose_sequence2d",
        "load_raw_prediction_frames",
        "write_pose_sequence_npz",
    }:
        from .adapter import canonicalize_pose_sequence2d, load_raw_prediction_frames, write_pose_sequence_npz

        exports = {
            "canonicalize_pose_sequence2d": canonicalize_pose_sequence2d,
            "load_raw_prediction_frames": load_raw_prediction_frames,
            "write_pose_sequence_npz": write_pose_sequence_npz,
        }
        return exports[name]

    if name in {"run_backend_job", "run_pose2d_backend"}:
        from .estimator import run_backend_job, run_pose2d_backend

        exports = {
            "run_backend_job": run_backend_job,
            "run_pose2d_backend": run_pose2d_backend,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
