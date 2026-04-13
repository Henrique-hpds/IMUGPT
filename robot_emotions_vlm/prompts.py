"""Prompt template loading and rendering for RobotEmotions VLM inference."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from .dataset import RobotEmotionsVideoClip

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_SYSTEM_PROMPT_PATH = PACKAGE_ROOT / "prompt_templates" / "system.txt"
DEFAULT_USER_PROMPT_PATH = PACKAGE_ROOT / "prompt_templates" / "user.txt"

PLACEHOLDER_RE = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")


def load_prompt_template(path: str | Path) -> str:
    """Load an editable text prompt from disk."""

    return Path(path).read_text(encoding="utf-8")


def render_template(template_text: str, values: dict[str, Any]) -> str:
    """Render ``{{placeholders}}`` without conflicting with JSON braces."""

    missing: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            missing.add(key)
            return match.group(0)
        return str(values[key])

    rendered = PLACEHOLDER_RE.sub(replace, template_text)
    if missing:
        names = ", ".join(sorted(missing))
        raise KeyError(f"Missing prompt placeholders: {names}")
    return rendered


def build_prompt_placeholders(
    record: RobotEmotionsVideoClip,
    video_metadata: dict[str, Any],
    *,
    analysis_scope: str = "full clip",
    window_metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Build the placeholder map used by both prompt files."""

    protocol = dict(record.protocol or {})
    placeholders = {
        "clip_id": record.clip_id,
        "domain": record.domain,
        "user_id": f"{record.user_id:02d}",
        "tag_number": f"{record.tag_number:02d}",
        "take_id": "" if record.take_id is None else record.take_id,
        "source_rel_dir": record.source_rel_dir,
        "video_path": str(record.video_path.resolve()),
        "labels_json": json.dumps(record.labels, ensure_ascii=True, sort_keys=True),
        "protocol_json": json.dumps(protocol, ensure_ascii=True, sort_keys=True),
        "video_metadata_json": json.dumps(video_metadata, ensure_ascii=True, sort_keys=True),
        "analysis_scope": str(analysis_scope),
        "window_metadata_json": json.dumps(window_metadata, ensure_ascii=True, sort_keys=True),
    }
    return placeholders


def build_prompt_placeholders_from_metadata(
    *,
    clip_id: str,
    domain: str,
    user_id: int,
    tag_number: int,
    take_id: str | None,
    source_rel_dir: str,
    video_path: str | Path,
    labels: dict[str, Any],
    protocol: dict[str, Any] | None,
    video_metadata: dict[str, Any],
    analysis_scope: str,
    window_metadata: dict[str, Any] | None,
) -> dict[str, str]:
    """Build the prompt placeholder map from raw metadata instead of a dataset record."""

    return {
        "clip_id": str(clip_id),
        "domain": str(domain),
        "user_id": f"{int(user_id):02d}",
        "tag_number": f"{int(tag_number):02d}",
        "take_id": "" if take_id is None else str(take_id),
        "source_rel_dir": str(source_rel_dir),
        "video_path": str(Path(video_path).resolve()),
        "labels_json": json.dumps(labels, ensure_ascii=True, sort_keys=True),
        "protocol_json": json.dumps(dict(protocol or {}), ensure_ascii=True, sort_keys=True),
        "video_metadata_json": json.dumps(video_metadata, ensure_ascii=True, sort_keys=True),
        "analysis_scope": str(analysis_scope),
        "window_metadata_json": json.dumps(window_metadata, ensure_ascii=True, sort_keys=True),
    }


def render_prompts(
    record: RobotEmotionsVideoClip,
    video_metadata: dict[str, Any],
    *,
    system_prompt_path: str | Path = DEFAULT_SYSTEM_PROMPT_PATH,
    user_prompt_path: str | Path = DEFAULT_USER_PROMPT_PATH,
    analysis_scope: str = "full clip",
    window_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and render the editable system and user prompts."""

    placeholders = build_prompt_placeholders(
        record,
        video_metadata,
        analysis_scope=analysis_scope,
        window_metadata=window_metadata,
    )
    return render_prompts_from_placeholders(
        placeholders,
        system_prompt_path=system_prompt_path,
        user_prompt_path=user_prompt_path,
    )


def render_prompts_from_placeholders(
    placeholders: dict[str, str],
    *,
    system_prompt_path: str | Path = DEFAULT_SYSTEM_PROMPT_PATH,
    user_prompt_path: str | Path = DEFAULT_USER_PROMPT_PATH,
) -> dict[str, Any]:
    """Load and render the editable system and user prompts from explicit placeholders."""

    system_template = load_prompt_template(system_prompt_path)
    user_template = load_prompt_template(user_prompt_path)
    return {
        "system_prompt_path": str(Path(system_prompt_path).resolve()),
        "user_prompt_path": str(Path(user_prompt_path).resolve()),
        "system_template": system_template,
        "user_template": user_template,
        "system_prompt": render_template(system_template, placeholders),
        "user_prompt": render_template(user_template, placeholders),
        "placeholders": placeholders,
    }
