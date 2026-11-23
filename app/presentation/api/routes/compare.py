"""Compare two saved analyses."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.presentation.api.dependencies import Container, get_container

router = APIRouter()


@router.get("/api/compare")
def compare(base: int, latest: int, c: Container = Depends(get_container)) -> dict:
    """Compare two saved analyses. The older one (lower id) is treated as the baseline."""
    return c.compare_runs.execute(base, latest)
