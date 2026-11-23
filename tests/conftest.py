"""Shared fixtures: a tiny deterministic engine, so no model is ever downloaded in tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.entities.prediction import Prediction
from app.domain.errors import EngineError
from app.domain.repositories.engine_repository import EngineRepository
from app.infrastructure.repositories.sqlite_run_repository import SqliteRunRepository

FIXTURES = Path(__file__).parent / "fixtures"
POSITIVE = ("love", "great", "excellent", "amazing", "ممتاز", "رائع", "excelente")
NEGATIVE = ("terrible", "bad", "broke", "awful", "disappointed", "سيئة", "سيء")


class FakeEngine:
    """Scores by keyword: positive +0.8, negative -0.8, otherwise 0."""

    label = "fake-engine"
    lock = None

    def predict(self, texts, progress=None):
        out = []
        for t in texts:
            low = t.lower()
            score = 0.8 if any(w in low for w in POSITIVE) else -0.8 if any(w in low for w in NEGATIVE) else 0.0
            level = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
            out.append(Prediction(level, score, 0.9))
        if progress:
            progress(len(texts), len(texts))
        return out


class FakeEngines(EngineRepository):
    default_id = "fake"

    def start(self) -> None:
        pass

    def describe(self) -> list[dict]:
        return [{"id": "fake", "name": "Fake", "description": "test engine", "state": "ready", "error": None}]

    def get(self, engine_id, ollama_model=None):
        if engine_id not in (None, "fake"):
            raise EngineError(400, f"Unknown model '{engine_id}'.")
        return FakeEngine()


class FakeCatalog:
    def list_models(self) -> list[str]:
        return ["qwen2.5:7b"]


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


@pytest.fixture
def runs(tmp_path):
    repo = SqliteRunRepository(str(tmp_path / "history.db"))
    repo.init()
    return repo


@pytest.fixture
def engines():
    return FakeEngines()
