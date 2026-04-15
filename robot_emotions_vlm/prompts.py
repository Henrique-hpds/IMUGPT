"""Prompt template loading and rendering for RobotEmotions VLM inference."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .schemas import DescriptionValidationError

from .dataset import RobotEmotionsVideoClip

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_SYSTEM_PROMPT_PATH = PACKAGE_ROOT / "prompt_templates" / "system.txt"
DEFAULT_USER_PROMPT_PATH = PACKAGE_ROOT / "prompt_templates" / "user.txt"

PLACEHOLDER_RE = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")
KIMODO_PROMPT_BLACKLIST = (
    "chair",
    "stool",
    "couch",
    "bench",
    "wire",
    "wires",
    "cable",
    "cord",
    "camera",
    "phone",
    "earbuds",
    "headset",
    "slippers",
)
NON_DEGRADING_PROMPT_WARNINGS = frozenset(
    {
        "prompt_text_wrapping_quotes_removed",
        "prompt_text_trailing_period_removed",
        "prompt_text_prefix_normalized",
        "prompt_text_temporal_tail_removed",
        "prompt_text_artifact_phrase_removed",
        "prompt_text_artifact_clause_removed",
        "prompt_text_duplicate_clause_removed",
        "prompt_text_redundant_token_removed",
        "prompt_text_body_parts_fallback_added",
        "prompt_text_clause_limit_applied",
        "prompt_text_word_limit_applied",
    }
)

_TEMPORAL_TAIL_RE = re.compile(r"\b(?:then|while|after|before)\b.*$", flags=re.IGNORECASE)
_ARTIFACT_KEYWORD_RE = re.compile(
    r"\b(?:chair|stool|couch|bench|wire|wires|cable|cord|camera|phone|earbuds|headset|slippers)\b",
    flags=re.IGNORECASE,
)
_ARTIFACT_PHRASE_RE = re.compile(
    r"\b(?:"
    r"on|in|with|around|using|wearing|holding|resting on|resting in|fiddling with|"
    r"connected to|restrained by|draped over|clasped around|supporting|covering"
    r")\s+(?:a|an|the|their|his|her)?\s*(?:[a-z-]+\s+){0,3}"
    r"(?:chair|stool|couch|bench|wire|wires|cable|cord|camera|phone|earbuds|headset|slippers)\b"
    r"(?:\s+[a-z-]+){0,4}",
    flags=re.IGNORECASE,
)
_CLAUSE_SPLIT_RE = re.compile(r"\s*,\s*")
_LEADING_FILLER_RE = re.compile(r"^(?:and|with|while|on|in|around)\s+", flags=re.IGNORECASE)
_TRAILING_FILLER_RE = re.compile(r"\s+(?:and|with|while|on|in|around)$", flags=re.IGNORECASE)
_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", flags=re.IGNORECASE)
_MULTISPACE_RE = re.compile(r"\s+")


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


def sanitize_kimodo_prompt_text(
    prompt_text: str,
    *,
    body_parts: Mapping[str, str] | None = None,
) -> tuple[str, list[str]]:
    """Normalize a VLM prompt into a short Kimodo-friendly motion description."""

    warnings: list[str] = []
    text = _normalize_whitespace(prompt_text)
    if not text:
        raise DescriptionValidationError("prompt_text cannot be empty after sanitization")

    trimmed = _TEMPORAL_TAIL_RE.sub("", text).strip(" ,")
    if trimmed != text:
        warnings.append("prompt_text_temporal_tail_removed")
        text = trimmed

    clauses = []
    for raw_clause in _CLAUSE_SPLIT_RE.split(text):
        clause, clause_warnings = _sanitize_clause(raw_clause)
        warnings.extend(clause_warnings)
        if clause:
            clauses.append(clause)

    clauses = _dedupe_clauses(clauses, warnings=warnings)
    if not clauses:
        raise DescriptionValidationError("prompt_text became empty after removing artifacts")

    if not clauses[0].startswith("A person"):
        clauses[0] = "A person " + clauses[0].lstrip(", ").lower()

    if body_parts is not None:
        clauses = _extend_with_body_parts(clauses, body_parts, warnings=warnings)

    if len(clauses) > 3:
        clauses = clauses[:3]
        warnings.append("prompt_text_clause_limit_applied")

    sanitized = ", ".join(clauses).strip(" ,")
    sanitized = _normalize_whitespace(sanitized)
    if not sanitized.startswith("A person"):
        raise DescriptionValidationError("sanitized prompt_text must start with 'A person'")
    if any(mark in sanitized for mark in ".!?"):
        raise DescriptionValidationError("sanitized prompt_text must be a single sentence without internal punctuation")

    words = sanitized.split()
    if len(words) < 12:
        raise DescriptionValidationError("sanitized prompt_text must contain at least 12 words")
    if len(words) > 22:
        sanitized = " ".join(words[:22]).rstrip(",")
        sanitized = _normalize_whitespace(sanitized)
        warnings.append("prompt_text_word_limit_applied")

    return sanitized, list(dict.fromkeys(warnings))


def _sanitize_clause(clause_text: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    clause = _normalize_whitespace(clause_text).strip(" ,")
    if not clause:
        return "", warnings

    updated = _ARTIFACT_PHRASE_RE.sub("", clause)
    updated = _normalize_whitespace(updated).strip(" ,")
    if updated != clause:
        warnings.append("prompt_text_artifact_phrase_removed")
        clause = updated

    if _ARTIFACT_KEYWORD_RE.search(clause):
        warnings.append("prompt_text_artifact_clause_removed")
        return "", warnings

    clause = _LEADING_FILLER_RE.sub("", clause)
    clause = _TRAILING_FILLER_RE.sub("", clause)
    clause = _normalize_whitespace(clause).strip(" ,")
    clause = _collapse_redundant_tokens(clause, warnings=warnings)
    return clause, warnings


def _extend_with_body_parts(
    clauses: list[str],
    body_parts: Mapping[str, str],
    *,
    warnings: list[str],
) -> list[str]:
    if len(" ".join(clauses).split()) >= 12 or len(clauses) >= 3:
        return clauses

    for field_name in ("arms", "trunk", "head", "legs"):
        raw_text = _normalize_whitespace(body_parts.get(field_name, ""))
        if not raw_text:
            continue
        raw_text = raw_text.rstrip(".")
        raw_text = _LEADING_ARTICLE_RE.sub("", raw_text)
        clause, clause_warnings = _sanitize_clause(raw_text)
        warnings.extend(clause_warnings)
        if not clause:
            continue
        clause_key = _clause_key(clause)
        if any(_clause_key(existing) == clause_key for existing in clauses):
            continue
        clauses.append(clause)
        warnings.append("prompt_text_body_parts_fallback_added")
        if len(" ".join(clauses).split()) >= 12 or len(clauses) >= 3:
            break
    return clauses


def _dedupe_clauses(clauses: Sequence[str], *, warnings: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for clause in clauses:
        key = _clause_key(clause)
        if not key:
            continue
        if key in seen:
            warnings.append("prompt_text_duplicate_clause_removed")
            continue
        deduped.append(clause)
        seen.add(key)
    return deduped


def _clause_key(clause: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clause.lower()).strip()


def _collapse_redundant_tokens(clause: str, *, warnings: list[str]) -> str:
    tokens = clause.split()
    if not tokens:
        return ""

    compact: list[str] = []
    for token in tokens:
        normalized = token.lower().strip(",")
        if compact and compact[-1].lower().strip(",") == normalized:
            warnings.append("prompt_text_redundant_token_removed")
            continue
        compact.append(token)
    return " ".join(compact)


def _normalize_whitespace(text: str) -> str:
    return _MULTISPACE_RE.sub(" ", str(text)).strip()
