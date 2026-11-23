"""Port: the list of models installed in the translation service."""
from __future__ import annotations

from typing import Protocol


class ModelCatalog(Protocol):
    def list_models(self) -> list[str]: ...
