"""Use cases that describe the engines: which are ready, and which translation models are installed."""
from __future__ import annotations

from app.domain.errors import TranslationError
from app.domain.repositories.engine_repository import EngineRepository
from app.domain.repositories.model_catalog import ModelCatalog


class GetEngineStatus:
    def __init__(self, engines: EngineRepository, bulk_rows: int, max_files: int) -> None:
        self.engines, self.bulk_rows, self.max_files = engines, bulk_rows, max_files

    def execute(self) -> dict:
        engines = self.engines.describe()
        return {"ready": any(e["state"] == "ready" for e in engines), "engines": engines,
                "default_engine": self.engines.default_id,
                "limits": {"bulk_rows": self.bulk_rows, "files": self.max_files}}


class ListTranslationModels:
    def __init__(self, catalog: ModelCatalog, default_model: str) -> None:
        self.catalog, self.default_model = catalog, default_model

    def execute(self) -> dict:
        try:
            models = self.catalog.list_models()
        except TranslationError as exc:
            return {"reachable": False, "models": [], "default": self.default_model, "error": str(exc)}
        ok = self.default_model in models or f"{self.default_model}:latest" in models
        return {"reachable": True, "models": models, "default": self.default_model if ok or not models else models[0]}
