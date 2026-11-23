#!/usr/bin/env python3
"""
Command-line sentiment analysis for tables, powered by a pre-trained model.

Uses tabularisai/multilingual-sentiment-analysis (Hugging Face): 23 languages
including Arabic and English, five levels from Very Negative to Very Positive.
No word lists or hand-written rules: every row is classified by the model.
Note: that model's license is CC BY-NC 4.0 (non-commercial use).

Setup
  pip install -r requirements.txt

Usage (from the project root)
  python -m app.presentation.cli reviews.csv --text-col review
  python -m app.presentation.cli reviews.csv --text-col review --group-by product
  python -m app.presentation.cli --demo
  python -m app.presentation.cli reviews.csv --model clapAI/modernBERT-base-multilingual-sentiment

Output: the original table plus
  sentiment_detail   Very Negative / Negative / Neutral / Positive / Very Positive
  sentiment_label    Positive / Neutral / Negative
  sentiment_score    -1 .. +1 (probability-weighted average of the levels)
  confidence         probability of the predicted level
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from app.application.services.analysis_service import score_table
from app.application.services.cleaning_service import clean_table, describe
from app.application.services.table_service import guess_text_column
from app.domain.value_objects.clean_options import CleanOptions
from app.infrastructure.config.settings import DEFAULT_MODEL
from app.infrastructure.ml.hf_sentiment_model import SentimentModel
from app.infrastructure.ml.language_detection import tag_language


def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sys.exit(f"File not found: {path}")
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    if p.suffix.lower() == ".json":
        return pd.read_json(p)
    return pd.read_csv(p, sep=None, engine="python")  # sniffs comma / tab / semicolon


def summarize(df: pd.DataFrame, group_by: str | None) -> str:
    n = len(df)
    lines = [f"\nRows analyzed: {n}", f"Average score: {df['sentiment_score'].mean():+.3f}", ""]
    counts = df["sentiment_label"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    for lab, c in counts.items():
        pct = c / n * 100 if n else 0
        lines.append(f"  {lab:<9} {c:>5}  {pct:5.1f}%  {'#' * int(pct / 2)}")
    for col in ("language", group_by):
        if col and col in df.columns and (col != "language" or df[col].nunique() > 1):
            g = df.groupby(col)["sentiment_score"].agg(rows="count", avg_score="mean").round(3)
            lines += ["", f"By {col}:", g.sort_values("avg_score", ascending=False).to_string()]
    return "\n".join(lines)


DEMO = pd.DataFrame(
    {
        "product": ["Headphones", "Headphones", "Keyboard", "Keyboard", "Monitor", "Monitor", "Monitor", "Headphones"],
        "review": [
            "Absolutely love these. The sound is excellent and the battery is amazing!",
            "Stopped working after two weeks. Terrible quality and support was rude.",
            "Comfortable and fast, but the keys are a bit noisy.",
            "Not good. The space bar broke and I want a refund.",
            "Delivery was late and the screen arrived damaged. Really disappointed.",
            "السماعات رائعة جدا وأنصح بها الجميع",
            "الجودة سيئة جدا وما أعجبني الصوت، مش حلو",
            "تجربة ممتازة وأفضل شاشة اشتريتها!",
        ],
    }
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-trained-model sentiment analysis for tables.")
    ap.add_argument("input", nargs="?", help="CSV / TSV / Excel / JSON file")
    ap.add_argument("--text-col", help="column with the text (auto-detected if omitted)")
    ap.add_argument("--group-by", help="column to average sentiment by")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Hugging Face model (default: {DEFAULT_MODEL})")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--output", help="output CSV (default: <input>_sentiment.csv)")
    ap.add_argument("--demo", action="store_true", help="run on built-in sample data")
    args = ap.parse_args()

    if args.demo:
        df, src = DEMO, Path("demo.csv")
        args.text_col = args.text_col or "review"
        args.group_by = args.group_by or "product"
    elif args.input:
        df, src = load_table(args.input), Path(args.input)
    else:
        ap.error("provide an input file or use --demo")

    try:
        text_col = args.text_col or guess_text_column(df)
        df, report = clean_table(df, text_col, CleanOptions(), keep=(args.group_by,))
        if report["enabled"]:
            print(describe(report), file=sys.stderr)
        if df.empty:
            sys.exit("No rows left after cleaning (every row was empty or a duplicate).")
        model = SentimentModel(args.model, args.batch_size)
        result = score_table(df, text_col, model, language_tagger=tag_language)
    except (ValueError, RuntimeError) as exc:
        sys.exit(str(exc))

    out_path = args.output or str(src.with_name(f"{src.stem}_sentiment.csv"))
    result.to_csv(out_path, index=False, encoding="utf-8-sig")  # utf-8-sig keeps Arabic readable in Excel

    shown = [text_col, "sentiment_detail", "sentiment_score"]
    with pd.option_context("display.max_colwidth", 60, "display.width", 140):
        print(result[shown].to_string(index=False))
    print(summarize(result, args.group_by))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

