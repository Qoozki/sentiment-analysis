# Architecture

## The rule

Dependencies point inward only:

```
presentation  ->  infrastructure  ->  application  ->  domain
```

- **domain** imports nothing from the other layers.
- **application** imports only domain, and no frameworks (no FastAPI, SQLite, torch, urllib...).
- **infrastructure** implements the domain's ports (interfaces).
- **presentation** is the only layer that knows about HTTP, and wires everything together.

`tests/unit/test_architecture.py` fails the build if an import breaks this.

## Ports and adapters

| Port (domain) | Adapter (infrastructure) |
|---|---|
| `RunRepository` | `SqliteRunRepository` |
| `EngineRepository` | `EngineRegistry` (loads engines in background threads) |
| `ModelCatalog` | `OllamaClient` |
| `SentimentEngine` (protocol) | `SentimentModel` (Hugging Face), `TranslateEngine` (Ollama + English model) |

Objects are built once in `presentation/api/dependencies.py` (`build_container`). Tests build the same container with fakes.

## Where the old files went

| Before | After |
|---|---|
| `app.py` | `presentation/api/main.py` (app factory), `routes/*.py`, `errors.py`, `dependencies.py`; env vars in `infrastructure/config/settings.py` |
| `service.py` | `application/services/analysis_service.py` (pipeline), `job_service.py`; `application/use_cases/bulk_analysis.py` |
| `helpers.py` | `application/services/table_service.py`, `summary_service.py`, `comparison_service.py`; `TableError` in `domain/errors.py` |
| `cleaning.py` | `application/services/cleaning_service.py`; `CleanOptions` in `domain/value_objects/clean_options.py` |
| `db.py` | `infrastructure/repositories/sqlite_run_repository.py` (port in `domain/repositories/run_repository.py`) |
| `engines.py` | `infrastructure/ml/translate_engine.py`, `ollama_client.py`, `engine_catalog.py`; `infrastructure/repositories/engine_registry.py`; errors in `domain/errors.py` |
| `sentiment_system.py` | `infrastructure/ml/hf_sentiment_model.py`, `language_detection.py`; scale in `domain/value_objects/sentiment_scale.py`; scoring in `analysis_service.py`; CLI in `presentation/cli.py` |
| `static/index.html` | `ui/index.html` |

Renames: `OllamaError` -> `TranslationError`, `OllamaUnavailable` -> `TranslationUnavailable`, `Registry` -> `EngineRegistry`, `add_language` -> `tag_language`, `analyze` -> `score_table`, `analyze_and_store` -> `AnalysisPipeline.run`.

## HTTP error mapping

Use cases raise domain errors; `presentation/api/errors.py` turns them into `{"detail": message}` (which the UI displays):

| Error | Status |
|---|---|
| `EngineError` | its own `status` (400 unknown engine, 503 loading/failed) |
| `LimitExceededError` | 413 |
| `NotFoundError` | 404 |
| any other `ValueError` (`TableError`, `InvalidRequestError`, `TranslationError`...) | 400 |

## Common changes

- **New engine:** write a class with `predict(texts, progress=None) -> list[Prediction]` in `infrastructure/ml/`, add an `EngineSpec` in `engine_catalog.py`.
- **New endpoint:** add a use case in `application/use_cases/`, register it in `build_container`, add a route in `presentation/api/routes/`.
- **Different database:** implement `RunRepository`, pass it to `build_container(runs=...)`.
