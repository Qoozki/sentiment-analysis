"""Quick analysis: typed texts, and a single uploaded table.

Plain `def` endpoints on purpose: FastAPI runs them in a thread pool, so a slow
model call does not block other requests.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.presentation.api.dependencies import Container, get_container
from app.presentation.schemas.analysis import TextsIn

router = APIRouter()


@router.post("/api/analyze")
def analyze_texts(body: TextsIn, include_rows: bool = True, c: Container = Depends(get_container)) -> dict:
    """Classify a list of texts and save the analysis to history."""
    o = c.analyze_texts.execute(body.texts, body.name, body.engine, body.ollama_model)
    return {"run_id": o.run_id, "text_col": "text", "columns": ["text"], "summary": o.summary,
            **({"results": o.records} if include_rows else {})}


@router.post("/api/preview")
async def preview(file: UploadFile = File(...), c: Container = Depends(get_container)) -> dict:
    """Read an uploaded table and suggest the text and date columns."""
    return c.preview_table.execute(await file.read(), file.filename)


@router.post("/api/analyze-file")
def analyze_file(
    file: UploadFile = File(...),
    text_col: str | None = Form(None),
    group_by: str | None = Form(None),
    date_col: str | None = Form(None),
    name: str | None = Form(None),
    engine: str | None = Form(None),
    ollama_model: str | None = Form(None),
    include_rows: bool = True,
    c: Container = Depends(get_container),
) -> dict:
    """Analyze one table (up to 5,000 rows) and save it to history."""
    text_col, o = c.analyze_table.execute(file.file.read(), file.filename, text_col, group_by, date_col,
                                          name, engine, ollama_model)
    return {"run_id": o.run_id, "text_col": text_col, "columns": o.columns, "summary": o.summary,
            **({"results": o.records} if include_rows else {})}
