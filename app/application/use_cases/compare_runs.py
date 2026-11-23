"""Use case: compare two saved analyses. The older one (lower id) is the baseline."""
from __future__ import annotations

from app.application.services.comparison_service import compare_summaries
from app.application.use_cases.history import load_run
from app.domain.errors import InvalidRequestError
from app.domain.repositories.run_repository import RunRepository

META_KEYS = ("id", "name", "created_at", "rows", "model")


class CompareRuns:
    def __init__(self, runs: RunRepository) -> None:
        self.runs = runs

    def execute(self, base: int, latest: int) -> dict:
        if base == latest:
            raise InvalidRequestError("Pick two different analyses.")
        a, b = sorted((load_run(self.runs, base), load_run(self.runs, latest)), key=lambda r: r["id"])
        meta = lambda r: {k: r[k] for k in META_KEYS}
        out = compare_summaries(a["summary"], b["summary"])
        if (a.get("model") or "") != (b.get("model") or ""):
            out["insights"].insert(0, {"level": "warn", "text": (
                "These two analyses used different models. Part of any change below may come from the models disagreeing, "
                "not from the data. For a fair comparison, run both on the same model.")})
        return {"base": meta(a), "latest": meta(b), **out}
