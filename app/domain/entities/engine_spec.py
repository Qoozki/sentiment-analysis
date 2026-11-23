"""Describes an engine that can be loaded: its id, display text and how to build it."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class EngineSpec:
    id: str
    name: str
    description: str
    factory: Callable[[], object]
