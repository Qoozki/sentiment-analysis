"""Optional: label each row's language when langdetect is installed."""
from __future__ import annotations

import pandas as pd


def tag_language(df: pd.DataFrame, text_col: str) -> None:
    """Optional: label each row's language if langdetect is installed."""
    try:
        from langdetect import DetectorFactory, detect
    except ImportError:
        return
    DetectorFactory.seed = 0

    def lang(t: str) -> str:
        letters = [c for c in t if c.isalpha()]
        if letters and sum("\u0600" <= c <= "\u06ff" for c in letters) / len(letters) >= 0.5:
            return "ar"  # langdetect sometimes mislabels Arabic as fa/ur
        try:
            return detect(t)
        except Exception:
            return "und"

    df["language"] = df[text_col].fillna("").astype(str).map(lang)

