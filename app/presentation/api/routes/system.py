"""The page itself, health and the list of installed translation models."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.presentation.api.dependencies import Container, get_container

router = APIRouter()


@router.get("/", include_in_schema=False)
def index(c: Container = Depends(get_container)) -> FileResponse:
    return FileResponse(c.settings.ui_dir / "index.html")


@router.get("/api/health")
def health(c: Container = Depends(get_container)) -> dict:
    return c.engine_status.execute()


@router.get("/api/ollama/models")
def ollama_models(c: Container = Depends(get_container)) -> dict:
    """Models installed in Ollama, for the translation picker."""
    return c.translation_models.execute()
