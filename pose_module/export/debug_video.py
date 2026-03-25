"""Debug video artifact helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_debug_overlay_path(
    output_dir: str | Path,
    *,
    filename: str = "debug_overlay.mp4",
    enabled: bool = True,
) -> Optional[Path]:
    if not bool(enabled):
        return None
    return Path(output_dir) / str(filename)
