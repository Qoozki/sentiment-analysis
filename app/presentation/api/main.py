"""Sentiment analysis website (FastAPI): quick analysis, bulk jobs, history and comparison.

Run:   uvicorn app.main:app --reload
Open:  http://127.0.0.1:8000        (interactive API docs: /docs)
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.infrastructure.config.settings import Settings
from app.presentation.api.dependencies import Container, build_container
from app.presentation.api.errors import register_error_handlers
from app.presentation.api.routes import analysis, bulk, compare, history, system


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or (container.settings if container else Settings.from_env())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        c = container or build_container(settings)
        c.runs.init()
        c.engines.start()  # every engine loads in the background; the site is usable at once
        app.state.container = c
        yield

    app = FastAPI(title="Sentiment Analysis", version="3.0", lifespan=lifespan)
    register_error_handlers(app)
    for module in (system, analysis, bulk, history, compare):
        app.include_router(module.router)
    return app


app = create_app()
