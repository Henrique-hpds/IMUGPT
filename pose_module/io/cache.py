"""Small cache and JSON helpers for pipeline artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def load_json_file(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_file(payload: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def tail_text(text: str, *, max_chars: int = 4000) -> str:
    value = str(text)
    if len(value) <= int(max_chars):
        return value
    return value[-int(max_chars) :]
