"""Use case: classify a list of typed-in texts and save the analysis to history."""
from __future__ import annotations

import pandas as pd

from app.application.services.analysis_service import AnalysisOutcome, AnalysisPipeline
from app.domain.errors import InvalidRequestError, LimitExceededError
from app.domain.repositories.engine_repository import EngineRepository
from app.domain.value_objects.clean_options import CleanOptions


class AnalyzeTexts:
    def __init__(self, pipeline: AnalysisPipeline, engines: EngineRepository, clean: CleanOptions, max_texts: int) -> None:
        self.pipeline, self.engines, self.clean, self.max_texts = pipeline, engines, clean, max_texts

    def execute(self, texts: list[str], name: str | None = None, engine_id: str | None = None,
                ollama_model: str | None = None) -> AnalysisOutcome:
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            raise InvalidRequestError("Add at least one non-empty text.")
        if len(texts) > self.max_texts:
            raise LimitExceededError(f"Too many texts ({len(texts)}). The limit is {self.max_texts}. Use Bulk for larger sets.")
        engine = self.engines.get(engine_id, ollama_model)
        return self.pipeline.run(pd.DataFrame({"text": texts}), "text", engine, kind="text", name=name, clean=self.clean)
