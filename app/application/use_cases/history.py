"""Use cases for saved analyses: list, open, page through rows, export, delete."""
from __future__ import annotations

from typing import Iterator

from app.domain.errors import NotFoundError
from app.domain.repositories.run_repository import RunRepository

DETAIL_KEYS = ("id", "name", "created_at", "kind", "text_col", "group_col", "date_col",
               "model", "columns", "files", "summary")


def load_run(runs: RunRepository, run_id: int) -> dict:
    run = runs.get_run(run_id)
    if run is None:
        raise NotFoundError("That analysis was not found. It may have been deleted.")
    return run


class ListRuns:
    def __init__(self, runs: RunRepository) -> None:
        self.runs = runs

    def execute(self) -> dict:
        return {"runs": self.runs.list_runs()}


class GetRun:
    def __init__(self, runs: RunRepository) -> None:
        self.runs = runs

    def execute(self, run_id: int) -> dict:
        r = load_run(self.runs, run_id)
        return {k: r[k] for k in DETAIL_KEYS}


class GetRunRows:
    def __init__(self, runs: RunRepository) -> None:
        self.runs = runs

    def execute(self, run_id: int, offset: int = 0, limit: int = 50, label: str | None = None,
                q: str | None = None, sort: str = "original") -> dict:
        load_run(self.runs, run_id)
        total, rows = self.runs.get_rows(run_id, offset, limit, label, q, sort)
        return {"total": total, "rows": rows}


class ExportRun:
    def __init__(self, runs: RunRepository) -> None:
        self.runs = runs

    def execute(self, run_id: int) -> Iterator[dict]:
        load_run(self.runs, run_id)
        return self.runs.iter_rows(run_id)


class DeleteRun:
    def __init__(self, runs: RunRepository) -> None:
        self.runs = runs

    def execute(self, run_id: int) -> dict:
        if not self.runs.delete_run(run_id):
            raise NotFoundError("That analysis was not found.")
        return {"deleted": run_id}
