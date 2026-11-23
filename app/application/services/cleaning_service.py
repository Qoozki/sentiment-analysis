"""Data cleaning: normalize review text, drop empty and duplicate rows, report what changed."""
from __future__ import annotations

import html
import re
import unicodedata

import pandas as pd

from app.domain.value_objects.clean_options import CleanOptions

# Placeholders that mean "no review". Only applied to the text column: elsewhere a value like "NA" can be real.
NULL_TOKENS = {"", "n/a", "na", "#n/a", "nan", "null", "none", "nil", "-", "--", "?"}
MAX_EXAMPLES = 3  # per reason, shown in the report
_WS = re.compile(r"\s+")
_TAG = re.compile(r"</?[A-Za-z!][^<>]{0,300}>")  # needs a letter after "<", so "I <3 it" is left alone
_BLOCK_TAG = re.compile(r"</?(?:br|p|div|li|ul|ol|tr|td|th|table|h[1-6]|hr|blockquote)\b[^<>]{0,300}>", re.I)  # these separate words
_URL = re.compile(r"(?:https?://|www\.)[^\s<>]+", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_HIDDEN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u00ad\u200b\u200e\u200f\u202a-\u202e\u2060\u2066-\u2069\ufeff]")
_ARABIC_MARKS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")  # diacritics + tatweel
_QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
                         "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'"})
_STRETCH = re.compile(r"([^\W\d_])\1{2,}")  # a letter repeated 3+ times


def _blank(s: pd.Series) -> pd.Series:
    """Missing or empty/whitespace-only."""
    return s.isna() | s.astype(str).str.strip().eq("")


def _no_text(s: pd.Series) -> pd.Series:
    """Like _blank, plus placeholders such as N/A or null (text column only)."""
    return s.isna() | s.astype(str).str.strip().str.lower().isin(NULL_TOKENS)


def _normalize(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.casefold().str.replace(_WS, " ", regex=True).str.strip()


def normalize_text(text: str) -> str:
    """Remove noise from one review without changing its wording (see the module docstring)."""

    t = unicodedata.normalize("NFKC", html.unescape(text))
    t = _BLOCK_TAG.sub(" ", t)  # <br>, <p>... become a space; inline tags such as <b> vanish without leaving one
    t = _TAG.sub("", t)
    t = _URL.sub(" ", t)
    t = _EMAIL.sub(" ", t)
    t = _HIDDEN.sub("", t)
    t = _ARABIC_MARKS.sub("", t)
    t = t.translate(_QUOTES)
    t = _STRETCH.sub(r"\1\1", t)
    t = t.lower()
    return _WS.sub(" ", t).strip()


def clean_table(df: pd.DataFrame, text_col: str, opts: CleanOptions | None = None, keep: tuple = ()) -> tuple[pd.DataFrame, dict]:
    """Return (cleaned copy, report). `keep` lists extra columns (group, date) that must not be dropped."""
    opts = opts or CleanOptions()
    df = df.reset_index(drop=True)
    n0 = len(df)
    report: dict = {
        "enabled": opts.active, "nulls": opts.nulls, "duplicates": opts.duplicates,
        "rows_before": n0, "rows_after": n0,
        "empty_text": 0, "missing_values": 0, "duplicates_removed": 0, "columns_dropped": [], "examples": [],
        "normalized": 0, "original_column": None,
    }
    if not opts.active or n0 == 0:
        return df, report

    # row numbers as the person sees them: position within each source file, or within the table
    if "source_file" in df.columns:
        rownum = df.groupby("source_file").cumcount() + 1
    else:
        rownum = pd.Series(range(1, n0 + 1), index=df.index)
    used = [c for c in keep if c and c in df.columns and c != text_col]
    protected = {text_col, *used, "source_file"}

    def example(reason: str, mask: pd.Series) -> None:
        for i in df.index[mask][:MAX_EXAMPLES]:
            raw = df.at[i, text_col]
            report["examples"].append({
                "reason": reason, "row": int(rownum[i]),
                "source": str(df.at[i, "source_file"]) if "source_file" in df.columns else None,
                "text": "" if pd.isna(raw) else str(raw).strip()[:80],
            })

    alive = pd.Series(True, index=df.index)  # rows still kept

    # 0. normalize the review text (before the empty and duplicate checks, so they see the cleaned text)
    changed = pd.Series(False, index=df.index)
    if opts.normalize:
        orig = df[text_col]
        norm = orig.map(lambda v: v if pd.isna(v) else normalize_text(str(v)))
        before = orig.where(orig.notna(), "").astype(str)
        changed = orig.notna() & (before != norm.where(norm.notna(), "").astype(str))
        df = df.assign(**{text_col: norm})

    # 1. missing values
    if opts.nulls != "none":
        empty_cols = [c for c in df.columns if c not in protected and _blank(df[c]).all()]
        report["columns_dropped"] = [str(c) for c in empty_cols]
        no_text = _no_text(df[text_col])
        other = pd.Series(False, index=df.index)
        if opts.nulls == "used":
            scope = used
        elif opts.nulls == "any":
            scope = [c for c in df.columns if c not in empty_cols and c not in (text_col, "source_file")]
        else:
            scope = []
        for c in scope:
            other |= _blank(df[c])
        report["empty_text"] = int(no_text.sum())
        report["missing_values"] = int((other & ~no_text).sum())
        example("empty text", no_text)
        example("missing value", other & ~no_text)
        alive &= ~(no_text | other)
        if empty_cols:
            df = df.drop(columns=empty_cols)

    # 2. duplicates (among the rows that survived step 1; the first occurrence is kept)
    if opts.duplicates != "none":
        sub = df[alive]
        if opts.duplicates == "rows":
            dup = sub.duplicated(subset=[c for c in df.columns if c != "source_file"], keep="first")
        else:
            key = _normalize(sub[text_col])
            dup = key.duplicated(keep="first") & key.ne("")
        dup = dup.reindex(df.index, fill_value=False)
        report["duplicates_removed"] = int(dup.sum())
        example("duplicate", dup)
        alive &= ~dup

    out = df[alive].reset_index(drop=True)
    kept_changed = changed & alive
    if kept_changed.any():  # keep the original wording of every row we changed, for auditing
        col = f"{text_col}_original"
        while col in out.columns:
            col += "_"
        out[col] = before.where(kept_changed)[alive].reset_index(drop=True)
        report["original_column"], report["normalized"] = col, int(kept_changed.sum())
        for i in df.index[kept_changed][:MAX_EXAMPLES]:
            report["examples"].append({
                "reason": "normalized", "row": int(rownum[i]),
                "source": str(df.at[i, "source_file"]) if "source_file" in df.columns else None,
                "text": before[i].strip()[:80], "after": str(df.at[i, text_col])[:80],
            })
    report["rows_after"] = len(out)
    return out, report


def describe(report: dict) -> str:
    """One readable sentence about what cleaning did."""
    if not report.get("enabled"):
        return ""
    removed = report["rows_before"] - report["rows_after"]
    parts = []
    if report["empty_text"]:
        parts.append(f"{report['empty_text']:,} with empty text")
    if report["missing_values"]:
        parts.append(f"{report['missing_values']:,} with a missing value in another column")
    if report["duplicates_removed"]:
        n = report["duplicates_removed"]
        parts.append(f"{n:,} duplicate{'s' if n != 1 else ''}")
    if removed:
        text = f"Cleaning removed {removed:,} of {report['rows_before']:,} rows ({removed / report['rows_before'] * 100:.1f}%): {', '.join(parts)}."
    else:
        text = "Cleaning found no empty or duplicate rows to remove."
    cols = report["columns_dropped"]
    if cols:
        text += f" Dropped {len(cols)} completely empty column{'s' if len(cols) != 1 else ''}: {', '.join(cols)}."
    if report.get("normalized"):
        n = report["normalized"]
        text += (f" Normalized the text of {n:,} row{'s' if n != 1 else ''} (spacing, HTML, links, hidden characters, stretched letters);"
                 f" the original wording is in the \u201c{report['original_column']}\u201d column.")
    return text
