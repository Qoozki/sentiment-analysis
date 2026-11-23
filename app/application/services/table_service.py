"""Reading uploaded tables and guessing which columns hold the text and the dates."""
from __future__ import annotations

import io
import json
import re

import pandas as pd

from app.domain.errors import TableError

MAX_ROWS = 5000
MAX_BYTES = 10 * 1024 * 1024
BULK_MAX_BYTES = 50 * 1024 * 1024
DATE_HINTS = ("date", "time", "created", "timestamp", "posted", "day", "month", "period", "التاريخ", "تاريخ")

# --------------------------------------------------------------------------
# Reading tables
# --------------------------------------------------------------------------
def parse_table(data: bytes, filename: str, max_rows: int = MAX_ROWS, max_bytes: int = MAX_BYTES) -> pd.DataFrame:
    name = (filename or "").lower()
    if len(data) > max_bytes:
        raise TableError(f"File is larger than {max_bytes // (1024 * 1024)} MB.")
    try:
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(data))
        elif name.endswith(".json"):
            df = pd.read_json(io.BytesIO(data))
        else:
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = data.decode("cp1256", errors="replace")  # common Arabic Windows encoding
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception as exc:
        raise TableError(f"Could not read the file as a table: {exc}") from exc
    if df.empty or len(df.columns) == 0:
        raise TableError("The table has no rows.")
    if len(df) > max_rows:
        raise TableError(f"The table has {len(df):,} rows; the limit is {max_rows:,}.")
    df.columns = [str(c) for c in df.columns]
    return df


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse a column of mixed date strings; unparseable values become NaT."""
    if pd.api.types.is_datetime64_any_dtype(series):
        d = pd.to_datetime(series, utc=True)
    else:
        s = series.astype(str).where(series.notna())
        try:
            d = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
        except (TypeError, ValueError):  # older pandas
            d = pd.to_datetime(s, errors="coerce", utc=True)
    return d.dt.tz_localize(None)


def guess_date_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
        if any(h in str(c).lower() for h in DATE_HINTS) and not pd.api.types.is_numeric_dtype(df[c]):
            if parse_dates(df[c].head(200)).notna().mean() >= 0.8:
                return c
    return None


def to_records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe records (NaN becomes null, Arabic stays readable)."""
    return json.loads(df.to_json(orient="records", force_ascii=False))


_DATE_LIKE = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")


def guess_text_column(df: pd.DataFrame) -> str:
    """Pick the column that looks most like free text (most words per cell).

    Numbers, booleans and date-like columns are skipped. Raises ValueError (never exits)
    so the web server can report it to the user instead of shutting down.
    """
    best, best_score = None, -1.0
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        vals = s.dropna().astype(str).head(500)
        if vals.empty:
            continue
        if vals.map(lambda v: bool(_DATE_LIKE.match(v))).mean() >= 0.8:
            continue
        score = vals.str.split().str.len().mean() + vals.str.len().mean() / 1000  # words first, length breaks ties
        if score > best_score:
            best, best_score = c, score
    if best is None:
        raise ValueError("No text column found in this table. Choose the text column yourself.")
    return best
