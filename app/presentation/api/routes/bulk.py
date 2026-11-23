"""Bulk analysis of one or more tables as a background job."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile

from app.presentation.api.dependencies import Container, get_container

router = APIRouter()


@router.post("/api/bulk")
async def bulk(
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
    text_col: str = Form(...),
    group_by: str | None = Form(None),
    date_col: str | None = Form(None),
    name: str | None = Form(None),
    engine: str | None = Form(None),
    ollama_model: str | None = Form(None),
    c: Container = Depends(get_container),
) -> dict:
    """Start a bulk analysis of one or more tables. Poll /api/jobs/{job_id} for progress."""
    job_id, selected = c.bulk.start(len(files), engine, ollama_model)
    payload = [(f.filename or f"file{i + 1}", await f.read()) for i, f in enumerate(files)]
    background.add_task(c.bulk.run, job_id, payload, selected, text_col, group_by or None, date_col or None, name)
    return {"job_id": job_id}


@router.get("/api/jobs/{job_id}")
def job_status(job_id: str, c: Container = Depends(get_container)) -> dict:
    return c.bulk.status(job_id)
