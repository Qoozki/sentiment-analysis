"""Descriptive data saved with every analysis in history."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunMeta:
    name: str
    kind: str                        # "text" | "table" | "bulk"
    text_col: str | None = None
    group_col: str | None = None
    date_col: str | None = None
    model: str | None = None
    columns: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
