"""Prompt-source helpers for generating canonical 3D pose sequences."""

from __future__ import annotations

from .adapter import (
    PROMPT_RAW_COORDINATE_SPACE,
    build_prompt_adapter_quality_report,
    build_prompt_metadata,
    build_prompt_pose_sequence3d,
    canonicalize_prompt_joint_positions_xyz,
)
from .backend import LegacyT2MGPTBackend, LegacyT2MGPTBackendConfig, PromptMotionBackend
from .catalog import (
    PromptCatalogEntry,
    PromptSample,
    canonicalize_label_value,
    canonicalize_labels,
    iter_prompt_samples,
    load_prompt_catalog,
    write_prompt_catalog,
)
from .quality import build_prompt_pose_quality_report

__all__ = [
    "PROMPT_RAW_COORDINATE_SPACE",
    "LegacyT2MGPTBackend",
    "LegacyT2MGPTBackendConfig",
    "PromptCatalogEntry",
    "PromptMotionBackend",
    "PromptSample",
    "build_prompt_adapter_quality_report",
    "build_prompt_metadata",
    "build_prompt_pose_quality_report",
    "build_prompt_pose_sequence3d",
    "canonicalize_label_value",
    "canonicalize_labels",
    "canonicalize_prompt_joint_positions_xyz",
    "iter_prompt_samples",
    "load_prompt_catalog",
    "write_prompt_catalog",
]
