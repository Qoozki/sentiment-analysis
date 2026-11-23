"""Request bodies for the JSON endpoints."""
from __future__ import annotations

from pydantic import BaseModel, Field


class TextsIn(BaseModel):
    texts: list[str] = Field(min_length=1)
    name: str | None = None
    engine: str | None = None
    ollama_model: str | None = None
