"""Saved analyses: list, open, page through rows, export as CSV, delete."""
from __future__ import annotations

import csv
import io

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from app.presentation.api.dependencies import Container, get_container

router = APIRouter()


@router.get("/api/runs")
def runs(c: Container = Depends(get_container)) -> dict:
    return c.list_runs.execute()


@router.get("/api/runs/{run_id}")
def run_detail(run_id: int, c: Container = Depends(get_container)) -> dict:
    return c.get_run.execute(run_id)


@router.get("/api/runs/{run_id}/rows")
def run_rows(
    run_id: int,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    label: str | None = None,
    q: str | None = None,
    sort: str = "original",
    c: Container = Depends(get_container),
) -> dict:
    return c.get_run_rows.execute(run_id, offset, limit, label, q, sort)


@router.get("/api/runs/{run_id}/export.csv")
def export_csv(run_id: int, c: Container = Depends(get_container)) -> StreamingResponse:
    rows = c.export_run.execute(run_id)

    def stream():
        buf, writer, keys = io.StringIO(), None, None
        for i, row in enumerate(rows):
            if writer is None:
                keys = list(row.keys())
                writer = csv.writer(buf)
                buf.write("\ufeff")  # byte-order mark so Excel reads Arabic correctly
                writer.writerow(keys)
            writer.writerow(["" if row.get(k) is None else row.get(k) for k in keys])
            if i % 500 == 0:
                yield buf.getvalue()
                buf.seek(0)
                buf.truncate()
        yield buf.getvalue()

    return StreamingResponse(stream(), media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": f'attachment; filename="sentiment_run_{run_id}.csv"'})


@router.delete("/api/runs/{run_id}")
def delete_run(run_id: int, c: Container = Depends(get_container)) -> dict:
    return c.delete_run.execute(run_id)
