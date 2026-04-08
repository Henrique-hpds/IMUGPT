"""Catalog contracts and canonical label handling for prompt-driven pose exports."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence


_LABEL_KEYS = ("emotion", "modality", "stimulus")
_NON_WORD_RE = re.compile(r"[^a-z0-9]+")


def canonicalize_label_value(value: Any) -> str:
    if value in (None, "", "None"):
        return "none"
    normalized = str(value).strip().lower()
    normalized = _NON_WORD_RE.sub("_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "none"


def canonicalize_optional_text(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def canonicalize_labels(labels: Mapping[str, Any]) -> Dict[str, str]:
    canonical = {
        key: canonicalize_label_value(labels.get(key))
        for key in _LABEL_KEYS
    }
    missing = [key for key, value in canonical.items() if value == "none" and labels.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Prompt catalog labels missing required keys: {missing}")
    return canonical


def make_prompt_sample_id(prompt_id: str, *, seed: int, sample_index: int, num_samples: int) -> str:
    prompt_id = str(prompt_id).strip()
    if num_samples <= 1 and sample_index == 0:
        return prompt_id
    return f"{prompt_id}_sample{sample_index + 1:02d}_seed{int(seed):04d}"


@dataclass(frozen=True)
class PromptCatalogEntry:
    prompt_id: str
    prompt_text: str
    labels: Dict[str, str]
    seed: int = 0
    num_samples: int = 1
    fps: float = 20.0
    duration_hint_sec: Optional[float] = None
    action_detail: Optional[str] = None
    stimulus_type: Optional[str] = None
    reference_clip_id: Optional[str] = None
    group_id: Optional[str] = None
    notes: Optional[str] = None
    source_metadata: Dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromptCatalogEntry":
        prompt_id = canonicalize_optional_text(payload.get("prompt_id"))
        prompt_text = canonicalize_optional_text(payload.get("prompt_text"))
        if prompt_id is None:
            raise ValueError("Prompt catalog entry requires a non-empty prompt_id.")
        if prompt_text is None:
            raise ValueError(f"Prompt catalog entry {prompt_id!r} requires non-empty prompt_text.")
        labels = canonicalize_labels(dict(payload.get("labels", {})))
        num_samples = int(payload.get("num_samples", 1))
        if num_samples <= 0:
            raise ValueError(f"Prompt catalog entry {prompt_id!r} must have num_samples >= 1.")
        fps = float(payload.get("fps", 20.0))
        if fps <= 0.0:
            raise ValueError(f"Prompt catalog entry {prompt_id!r} must have fps > 0.")
        return cls(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            labels=labels,
            seed=int(payload.get("seed", 0)),
            num_samples=num_samples,
            fps=fps,
            duration_hint_sec=(
                None if payload.get("duration_hint_sec") in (None, "") else float(payload["duration_hint_sec"])
            ),
            action_detail=canonicalize_optional_text(payload.get("action_detail")),
            stimulus_type=canonicalize_optional_text(payload.get("stimulus_type")),
            reference_clip_id=canonicalize_optional_text(payload.get("reference_clip_id")),
            group_id=canonicalize_optional_text(payload.get("group_id")),
            notes=canonicalize_optional_text(payload.get("notes")),
            source_metadata=(
                None
                if payload.get("source_metadata") in (None, "")
                else dict(payload.get("source_metadata", {}))
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "prompt_id": str(self.prompt_id),
            "prompt_text": str(self.prompt_text),
            "labels": dict(self.labels),
            "seed": int(self.seed),
            "num_samples": int(self.num_samples),
            "fps": float(self.fps),
        }
        if self.duration_hint_sec is not None:
            payload["duration_hint_sec"] = float(self.duration_hint_sec)
        if self.action_detail is not None:
            payload["action_detail"] = str(self.action_detail)
        if self.stimulus_type is not None:
            payload["stimulus_type"] = str(self.stimulus_type)
        if self.reference_clip_id is not None:
            payload["reference_clip_id"] = str(self.reference_clip_id)
        if self.group_id is not None:
            payload["group_id"] = str(self.group_id)
        if self.notes is not None:
            payload["notes"] = str(self.notes)
        if self.source_metadata:
            payload["source_metadata"] = dict(self.source_metadata)
        return payload


@dataclass(frozen=True)
class PromptSample:
    sample_id: str
    sample_index: int
    prompt_id: str
    prompt_text: str
    labels: Dict[str, str]
    seed: int
    fps: float
    duration_hint_sec: Optional[float]
    action_detail: Optional[str]
    stimulus_type: Optional[str]
    reference_clip_id: Optional[str]
    group_id: Optional[str]
    notes: Optional[str]
    source_metadata: Dict[str, Any]

    @classmethod
    def from_catalog_entry(cls, entry: PromptCatalogEntry, *, sample_index: int) -> "PromptSample":
        sample_seed = int(entry.seed) + int(sample_index)
        return cls(
            sample_id=make_prompt_sample_id(
                entry.prompt_id,
                seed=sample_seed,
                sample_index=int(sample_index),
                num_samples=int(entry.num_samples),
            ),
            sample_index=int(sample_index),
            prompt_id=str(entry.prompt_id),
            prompt_text=str(entry.prompt_text),
            labels=dict(entry.labels),
            seed=sample_seed,
            fps=float(entry.fps),
            duration_hint_sec=None if entry.duration_hint_sec is None else float(entry.duration_hint_sec),
            action_detail=None if entry.action_detail is None else str(entry.action_detail),
            stimulus_type=None if entry.stimulus_type is None else str(entry.stimulus_type),
            reference_clip_id=(
                None if entry.reference_clip_id is None else str(entry.reference_clip_id)
            ),
            group_id=None if entry.group_id is None else str(entry.group_id),
            notes=None if entry.notes is None else str(entry.notes),
            source_metadata={} if entry.source_metadata is None else dict(entry.source_metadata),
        )

    def to_prompt_metadata(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "sample_id": str(self.sample_id),
            "prompt_id": str(self.prompt_id),
            "prompt_text": str(self.prompt_text),
            "labels": dict(self.labels),
            "seed": int(self.seed),
            "sample_index": int(self.sample_index),
            "fps": float(self.fps),
        }
        if self.duration_hint_sec is not None:
            payload["duration_hint_sec"] = float(self.duration_hint_sec)
        if self.action_detail is not None:
            payload["action_detail"] = str(self.action_detail)
        if self.stimulus_type is not None:
            payload["stimulus_type"] = str(self.stimulus_type)
        if self.reference_clip_id is not None:
            payload["reference_clip_id"] = str(self.reference_clip_id)
        if self.group_id is not None:
            payload["group_id"] = str(self.group_id)
        if self.notes is not None:
            payload["notes"] = str(self.notes)
        if self.source_metadata:
            payload["source_metadata"] = dict(self.source_metadata)
        return payload


def load_prompt_catalog(catalog_path: str | Path) -> list[PromptCatalogEntry]:
    catalog_path = Path(catalog_path)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Prompt catalog not found: {catalog_path}")

    entries: list[PromptCatalogEntry] = []
    seen_prompt_ids: set[str] = set()
    with catalog_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            payload = json.loads(line)
            entry = PromptCatalogEntry.from_dict(payload)
            if entry.prompt_id in seen_prompt_ids:
                raise ValueError(
                    f"Prompt catalog contains duplicate prompt_id {entry.prompt_id!r} on line {line_number}."
                )
            seen_prompt_ids.add(entry.prompt_id)
            entries.append(entry)
    return entries


def write_prompt_catalog(entries: Sequence[PromptCatalogEntry], catalog_path: str | Path) -> None:
    catalog_path = Path(catalog_path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with catalog_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=True) + "\n")


def iter_prompt_samples(entries: Sequence[PromptCatalogEntry]) -> Iterator[PromptSample]:
    for entry in entries:
        for sample_index in range(int(entry.num_samples)):
            yield PromptSample.from_catalog_entry(entry, sample_index=sample_index)
