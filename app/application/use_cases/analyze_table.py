"""Use case: analyze one uploaded table (up to the single-file row limit) and save it to history."""
from __future__ import annotations

from app.application.services.analysis_service import AnalysisOutcome, AnalysisPipeline
from app.application.services.table_service import guess_text_column, parse_table
from app.domain.repositories.engine_repository import EngineRepository
from app.domain.value_objects.clean_options import CleanOptions


class AnalyzeTable:
    def __init__(self, pipeline: AnalysisPipeline, engines: EngineRepository, clean: CleanOptions) -> None:
        self.pipeline, self.engines, self.clean = pipeline, engines, clean

    def execute(self, data: bytes, filename: str | None, text_col: str | None = None, group_by: str | None = None,
                date_col: str | None = None, name: str | None = None, engine_id: str | None = None,
                ollama_model: str | None = None) -> tuple[str, AnalysisOutcome]:
        engine = self.engines.get(engine_id, ollama_model)  # fail fast if the engine is not ready
        df = parse_table(data, filename or "")
        text_col = text_col or guess_text_column(df)
        outcome = self.pipeline.run(
            df, text_col, engine, kind="table", name=name or filename, group_by=group_by or None,
            date_col=date_col or None, files=[filename or "upload"], clean=self.clean)
        return text_col, outcome
