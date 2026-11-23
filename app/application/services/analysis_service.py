"""The analysis workflow: clean the table, score every row with an engine, summarize, save to history."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import pandas as pd

from app.application.services.cleaning_service import clean_table
from app.application.services.cleaning_service import describe as describe_cleaning
from app.application.services.summary_service import build_insights, build_summary
from app.application.services.table_service import to_records
from app.domain.entities.run_meta import RunMeta
from app.domain.errors import TableError
from app.domain.repositories.run_repository import RunRepository
from app.domain.value_objects.clean_options import CleanOptions
from app.domain.value_objects.sentiment_scale import label_for

_FALLBACK_LOCK = threading.Lock()  # engines carry their own lock; this is only for ones that do not

# Optional step that adds a "language" column. Injected so this layer does not depend on langdetect.
LanguageTagger = Callable[[pd.DataFrame, str], None]


@dataclass
class AnalysisOutcome:
    run_id: int
    summary: dict
    records: list[dict]
    columns: list[str]  # columns of the (cleaned) table that was analyzed


def score_table(df: pd.DataFrame, text_col: str, engine, progress=None,
                language_tagger: LanguageTagger | None = None) -> pd.DataFrame:
    """Add sentiment_detail / sentiment_label / sentiment_score / confidence (and any engine extras) to a copy."""
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Available: {', '.join(map(str, df.columns))}")
    out = df.copy()
    texts = out[text_col].fillna("").astype(str).tolist()
    preds = engine.predict(texts, progress=progress)
    out["sentiment_detail"] = [p[0] for p in preds]
    out["sentiment_label"] = [label_for(p[1]) for p in preds]
    out["sentiment_score"] = [round(p[1], 3) for p in preds]
    out["confidence"] = [round(p[2], 3) for p in preds]
    extras = [p[3] if len(p) > 3 else None for p in preds]  # engines may add columns, e.g. english_text
    keys = list(dict.fromkeys(k for e in extras if e for k in e))
    for k in keys:
        out[k] = [(e or {}).get(k) for e in extras]
    blank = out[text_col].isna() | (out[text_col].astype(str).str.strip() == "")
    out.loc[blank, ["sentiment_detail", "sentiment_label", "sentiment_score", "confidence"]] = ["Neutral", "Neutral", 0.0, None]
    if language_tagger:
        language_tagger(out, text_col)
    return out


class AnalysisPipeline:
    def __init__(self, runs: RunRepository, language_tagger: LanguageTagger | None = None) -> None:
        self.runs = runs
        self.language_tagger = language_tagger

    def run(self, df: pd.DataFrame, text_col: str, engine, *, kind: str, name: str | None = None,
            group_by: str | None = None, date_col: str | None = None, files: list[str] | None = None,
            progress=None, model_name: str = "", clean: CleanOptions | None = None) -> AnalysisOutcome:
        """Classify every row, build the summary, save to history."""
        if text_col not in df.columns:
            raise TableError(f"Column '{text_col}' not found.")
        for label, col in (("Group", group_by), ("Date", date_col)):
            if col and col not in df.columns:
                raise TableError(f"{label} column '{col}' not found.")
        if not df[text_col].fillna("").astype(str).str.strip().ne("").any():
            raise TableError(f"The column '{text_col}' has no text to analyze.")

        report = None
        if clean is not None:  # drop empty / duplicate rows first, so they never reach the model or the statistics
            df, report = clean_table(df, text_col, clean, keep=(group_by, date_col))
            if df.empty:
                raise TableError("After cleaning, no rows were left to analyze. " + describe_cleaning(report)
                                 + " Try less strict cleaning options.")

        with getattr(engine, "lock", None) or _FALLBACK_LOCK:  # per engine, so different engines can run at the same time
            result = score_table(df, text_col, engine, progress=progress, language_tagger=self.language_tagger)
        summary = build_summary(result, text_col, group_by, date_col)
        summary["insights"] = build_insights(summary)
        _translation_insights(result, summary)
        if report and report["enabled"]:
            summary["cleaning"] = report
            summary["insights"].insert(0, {"level": "info", "text": describe_cleaning(report)})
        describe = getattr(engine, "describe", None)
        model_name = describe() if describe else (getattr(engine, "label", None) or model_name)
        records = to_records(result)
        run_id = self.runs.save_run(
            RunMeta(
                name=(name or "").strip() or _default_name(),
                kind=kind, text_col=text_col, group_col=group_by, date_col=date_col,
                model=model_name, columns=list(df.columns), files=files or [],
            ),
            summary,
            records,
        )
        return AnalysisOutcome(run_id, summary, records, list(df.columns))


def _default_name() -> str:
    return datetime.now().strftime("Analysis %Y-%m-%d %H:%M")


def _translation_insights(result: pd.DataFrame, summary: dict) -> None:
    """When the translating engine was used, say how many rows were translated and flag any that failed."""
    if "translation_status" not in result.columns:
        return
    counts = result["translation_status"].value_counts()
    done, failed = int(counts.get("translated", 0)), int(counts.get("failed", 0))
    summary["translation"] = {"translated": done, "failed": failed}
    if done:
        summary["insights"].append({"level": "info", "text": (
            f"{done:,} row{'s were' if done != 1 else ' was'} translated to English before scoring. Translation quality "
            "affects those scores, so spot-check some rows using the English translation column.")})
    if failed:
        summary["insights"].append({"level": "warn", "text": (
            f"{failed:,} row{'s' if failed != 1 else ''} could not be translated and {'were' if failed != 1 else 'was'} scored "
            "in the original language by the English model, so treat those scores as unreliable.")})
