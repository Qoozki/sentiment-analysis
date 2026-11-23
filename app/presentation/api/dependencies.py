"""Composition root: builds every object once and hands the use cases to the routes."""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from app.application.services.analysis_service import AnalysisPipeline
from app.application.services.job_service import JobRegistry
from app.application.use_cases.analyze_table import AnalyzeTable
from app.application.use_cases.analyze_texts import AnalyzeTexts
from app.application.use_cases.bulk_analysis import BulkAnalysis
from app.application.use_cases.compare_runs import CompareRuns
from app.application.use_cases.engine_info import GetEngineStatus, ListTranslationModels
from app.application.use_cases.history import DeleteRun, ExportRun, GetRun, GetRunRows, ListRuns
from app.application.use_cases.preview_table import PreviewTable
from app.domain.repositories.engine_repository import EngineRepository
from app.domain.repositories.model_catalog import ModelCatalog
from app.domain.repositories.run_repository import RunRepository
from app.infrastructure.config.settings import Settings
from app.infrastructure.ml.engine_catalog import default_specs
from app.infrastructure.ml.language_detection import tag_language
from app.infrastructure.ml.ollama_client import OllamaClient
from app.infrastructure.repositories.engine_registry import EngineRegistry
from app.infrastructure.repositories.sqlite_run_repository import SqliteRunRepository


@dataclass
class Container:
    settings: Settings
    runs: RunRepository
    engines: EngineRepository
    analyze_texts: AnalyzeTexts
    analyze_table: AnalyzeTable
    preview_table: PreviewTable
    bulk: BulkAnalysis
    list_runs: ListRuns
    get_run: GetRun
    get_run_rows: GetRunRows
    export_run: ExportRun
    delete_run: DeleteRun
    compare_runs: CompareRuns
    engine_status: GetEngineStatus
    translation_models: ListTranslationModels


def build_container(settings: Settings, engines: EngineRepository | None = None,
                    catalog: ModelCatalog | None = None, runs: RunRepository | None = None) -> Container:
    """Wire everything. Tests pass fakes for `engines`, `catalog` and `runs`."""
    runs = runs or SqliteRunRepository(settings.history_db)
    if engines is None or catalog is None:
        ollama = OllamaClient(settings.ollama_url, workers=settings.ollama_workers)
        catalog = catalog or ollama
        engines = engines or EngineRegistry(default_specs(settings, ollama), settings.default_engine)
    pipeline = AnalysisPipeline(runs, language_tagger=tag_language)
    return Container(
        settings=settings, runs=runs, engines=engines,
        analyze_texts=AnalyzeTexts(pipeline, engines, settings.clean, settings.max_texts),
        analyze_table=AnalyzeTable(pipeline, engines, settings.clean),
        preview_table=PreviewTable(settings.bulk_max_rows, settings.bulk_max_bytes),
        bulk=BulkAnalysis(pipeline, engines, JobRegistry(), settings.clean, settings.bulk_max_rows,
                          settings.max_files, settings.bulk_max_bytes),
        list_runs=ListRuns(runs), get_run=GetRun(runs), get_run_rows=GetRunRows(runs),
        export_run=ExportRun(runs), delete_run=DeleteRun(runs), compare_runs=CompareRuns(runs),
        engine_status=GetEngineStatus(engines, settings.bulk_max_rows, settings.max_files),
        translation_models=ListTranslationModels(catalog, settings.ollama_model),
    )


def get_container(request: Request) -> Container:
    return request.app.state.container
