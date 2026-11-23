"""Port: where analyses are stored."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from app.domain.entities.run_meta import RunMeta


class RunRepository(ABC):
    @abstractmethod
    def init(self) -> None: ...

    @abstractmethod
    def save_run(self, meta: RunMeta, summary: dict, rows: list[dict]) -> int:
        """Store one analysis (metadata, summary and every scored row). Returns its id."""

    @abstractmethod
    def list_runs(self, limit: int = 200) -> list[dict]: ...

    @abstractmethod
    def get_run(self, run_id: int) -> dict | None: ...

    @abstractmethod
    def get_rows(self, run_id: int, offset: int = 0, limit: int = 50, label: str | None = None,
                 q: str | None = None, sort: str = "original") -> tuple[int, list[dict]]: ...

    @abstractmethod
    def iter_rows(self, run_id: int, batch: int = 1000) -> Iterator[dict]: ...

    @abstractmethod
    def delete_run(self, run_id: int) -> bool: ...
