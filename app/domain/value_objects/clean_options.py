"""How a table is cleaned before scoring."""
from __future__ import annotations

from dataclasses import dataclass

NULL_MODES = ("none", "text", "used", "any")
DUPLICATE_MODES = ("none", "rows", "text")


@dataclass(frozen=True)
class CleanOptions:
    nulls: str = "text"
    duplicates: str = "rows"
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.nulls not in NULL_MODES:
            raise ValueError(f"Unknown missing-values option '{self.nulls}'. Choose one of: {', '.join(NULL_MODES)}.")
        if self.duplicates not in DUPLICATE_MODES:
            raise ValueError(f"Unknown duplicates option '{self.duplicates}'. Choose one of: {', '.join(DUPLICATE_MODES)}.")

    @property
    def active(self) -> bool:
        return self.nulls != "none" or self.duplicates != "none" or self.normalize
