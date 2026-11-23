"""Use case: analyze many tables as one background job. The browser polls the job for progress."""
from __future__ import annotations

import logging

import pandas as pd

from app.application.services.analysis_service import AnalysisPipeline
from app.application.services.job_service import JobRegistry
from app.application.services.table_service import parse_table
from app.domain.errors import InvalidRequestError, NotFoundError, TableError
from app.domain.repositories.engine_repository import EngineRepository
from app.domain.value_objects.clean_options import CleanOptions

log = logging.getLogger("sentiment")


class BulkAnalysis:
    def __init__(self, pipeline: AnalysisPipeline, engines: EngineRepository, jobs: JobRegistry,
                 clean: CleanOptions, max_rows: int, max_files: int, max_bytes: int) -> None:
        self.pipeline, self.engines, self.jobs, self.clean = pipeline, engines, jobs, clean
        self.max_rows, self.max_files, self.max_bytes = max_rows, max_files, max_bytes

    def start(self, n_files: int, engine_id: str | None, ollama_model: str | None):
        """Validate the request and create the job. Returns (job_id, engine); call run() with them."""
        if n_files == 0:
            raise InvalidRequestError("Choose at least one file.")
        if n_files > self.max_files:
            raise InvalidRequestError(f"Too many files ({n_files}). The limit is {self.max_files}.")
        engine = self.engines.get(engine_id, ollama_model)
        return self.jobs.create(), engine

    def status(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if job is None:
            raise NotFoundError("Unknown job. The server may have restarted; check History for the result.")
        return {k: job[k] for k in ("state", "done", "total", "run_id", "error")}

    def run(self, job_id: str, files: list[tuple[str, bytes]], engine, text_col: str, group_by: str | None,
            date_col: str | None, name: str | None, model_name: str = "") -> None:
        """Runs in a background thread. Never raises: problems are written to the job."""
        try:
            self.jobs.update(job_id, state="reading")
            frames, total = [], 0
            for fname, data in files:
                try:
                    df = parse_table(data, fname, max_rows=self.max_rows, max_bytes=self.max_bytes)
                except TableError as exc:
                    raise TableError(f"{fname}: {exc}") from exc
                if text_col not in df.columns:
                    raise TableError(f"{fname}: there is no column named '{text_col}'.")
                if "source_file" in df.columns:
                    df = df.rename(columns={"source_file": "source_file_original"})
                df.insert(0, "source_file", fname)
                frames.append(df)
                total += len(df)
                if total > self.max_rows:
                    raise TableError(f"Together the files have more than {self.max_rows:,} rows, which is the limit.")
            merged = pd.concat(frames, ignore_index=True, sort=False)
            self.jobs.update(job_id, state="analyzing", total=len(merged), done=0)
            outcome = self.pipeline.run(
                merged, text_col, engine, kind="bulk", name=name, group_by=group_by, date_col=date_col,
                files=[f for f, _ in files], model_name=model_name, clean=self.clean,
                progress=lambda done, tot: self.jobs.update(job_id, done=done, total=tot),
            )
            self.jobs.update(job_id, state="done", run_id=outcome.run_id, done=len(merged))
        except (TableError, ValueError) as exc:
            self.jobs.update(job_id, state="error", error=str(exc))
        except Exception as exc:  # keep the server alive and tell the user something useful
            log.exception("Bulk job failed")
            self.jobs.update(job_id, state="error", error=f"Unexpected error: {exc}")
