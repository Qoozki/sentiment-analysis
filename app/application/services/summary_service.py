"""Summary statistics and plain-language insights for one scored table."""
from __future__ import annotations

import math

import pandas as pd

from app.application.services.formatting import signed as _f
from app.application.services.table_service import parse_dates
from app.domain.value_objects.sentiment_scale import LABELS, LEVEL_ORDER

LOW_CONFIDENCE = 0.5


# --------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------
def _pct(x: float, n: int) -> float:
    return round(x / n * 100, 1) if n else 0.0


def _group_table(scored: pd.DataFrame, col: str, limit: int = 50) -> tuple[list[dict], bool]:
    rows = []
    for name, sub in scored.groupby(col, dropna=False):
        n = len(sub)
        pos = int((sub["sentiment_label"] == "Positive").sum())
        neg = int((sub["sentiment_label"] == "Negative").sum())
        rows.append(
            {
                "name": "(blank)" if pd.isna(name) else str(name),
                "rows": n,
                "avg": round(float(sub["sentiment_score"].mean()), 3),
                "pos_pct": _pct(pos, n),
                "neg_pct": _pct(neg, n),
                "net": round(_pct(pos, n) - _pct(neg, n), 1),
            }
        )
    truncated = len(rows) > limit
    if truncated:
        rows = sorted(rows, key=lambda r: -r["rows"])[:limit]
    rows.sort(key=lambda r: r["avg"], reverse=True)
    return rows, truncated


def _trend(scored: pd.DataFrame, date_col: str) -> dict | None:
    dates = parse_dates(scored[date_col])
    ok = dates.notna()
    if ok.sum() < 6:
        return None
    d = scored.loc[ok].assign(_d=dates[ok])
    span = (d["_d"].max() - d["_d"].min()).days
    freq, unit = ("D", "day") if span <= 45 else ("W", "week") if span <= 400 else ("M", "month")
    d = d.assign(_p=d["_d"].dt.to_period(freq))
    points = []
    for p, sub in d.groupby("_p"):
        n = len(sub)
        points.append(
            {
                "label": str(p.start_time.date()),
                "rows": n,
                "avg": round(float(sub["sentiment_score"].mean()), 3),
                "pos_pct": _pct(int((sub["sentiment_label"] == "Positive").sum()), n),
                "neg_pct": _pct(int((sub["sentiment_label"] == "Negative").sum()), n),
            }
        )
    if len(points) < 2:
        return None
    d = d.sort_values("_d")
    half = len(d) // 2
    return {
        "unit": unit,
        "points": points,
        "dated_rows": int(ok.sum()),
        "first_half_avg": round(float(d["sentiment_score"].iloc[:half].mean()), 3),
        "second_half_avg": round(float(d["sentiment_score"].iloc[half:].mean()), 3),
    }


def _examples(sub: pd.DataFrame, text_col: str, group_by: str | None) -> list[dict]:
    out = []
    for _, r in sub.iterrows():
        out.append(
            {
                "text": str(r[text_col])[:300],
                "score": round(float(r["sentiment_score"]), 3),
                "confidence": None if pd.isna(r["confidence"]) else round(float(r["confidence"]), 3),
                "group": None if not group_by or pd.isna(r.get(group_by)) else str(r.get(group_by)),
            }
        )
    return out


def build_summary(df: pd.DataFrame, text_col: str, group_by: str | None = None, date_col: str | None = None) -> dict:
    total = len(df)
    scored = df[df["confidence"].notna()] if "confidence" in df.columns else df  # blanks have no confidence
    n = len(scored)
    if n == 0:
        raise ValueError("No scored rows to summarize.")
    sc = scored["sentiment_score"]
    counts = scored["sentiment_label"].value_counts().reindex(LABELS, fill_value=0)
    pos, neu, neg = (int(counts[k]) for k in LABELS)
    details = scored["sentiment_detail"].value_counts()
    conf = scored["confidence"].dropna()

    summary: dict = {
        "rows": total,
        "scored": n,
        "empty_rows": total - n,
        "average_score": round(float(sc.mean()), 3),
        "stats": {
            "mean": round(float(sc.mean()), 3),
            "median": round(float(sc.median()), 3),
            "std": round(float(sc.std(ddof=0)), 3),
            "min": round(float(sc.min()), 3),
            "max": round(float(sc.max()), 3),
            "p10": round(float(sc.quantile(0.1)), 3),
            "p90": round(float(sc.quantile(0.9)), 3),
        },
        "labels": {"Positive": pos, "Neutral": neu, "Negative": neg},
        "label_pct": {"Positive": _pct(pos, n), "Neutral": _pct(neu, n), "Negative": _pct(neg, n)},
        "net_sentiment": round(_pct(pos, n) - _pct(neg, n), 1),
        "strong_pct": _pct(int((sc.abs() >= 0.5).sum()), n),
        "details": {k: int(details[k]) for k in LEVEL_ORDER if k in details.index},
        "avg_confidence": round(float(conf.mean()), 3) if len(conf) else None,
        "low_confidence_pct": _pct(int((conf < LOW_CONFIDENCE).sum()), len(conf)) if len(conf) else 0.0,
        "by_language": None,
        "by_group": None,
        "by_source": None,
        "date_col": date_col if date_col in df.columns else None,
        "trend": None,
    }
    if "language" in scored.columns and scored["language"].nunique() > 1:
        summary["by_language"], _ = _group_table(scored, "language")
    if group_by and group_by in scored.columns:
        items, trunc = _group_table(scored, group_by)
        summary["by_group"] = {"column": group_by, "items": items, "truncated": trunc}
    if "source_file" in scored.columns and scored["source_file"].nunique() > 1:
        summary["by_source"], _ = _group_table(scored, "source_file")
    if date_col and date_col in scored.columns:
        summary["trend"] = _trend(scored, date_col)

    pos_rows = scored[sc > 0].sort_values(["sentiment_score", "confidence"], ascending=False).head(5)
    neg_rows = scored[sc < 0].sort_values(["sentiment_score", "confidence"], ascending=[True, False]).head(5)
    summary["top_positive"] = _examples(pos_rows, text_col, group_by)
    summary["top_negative"] = _examples(neg_rows, text_col, group_by)
    return summary


# --------------------------------------------------------------------------
# Plain-language insights for decision-making
# --------------------------------------------------------------------------
def build_insights(s: dict) -> list[dict]:
    out: list[dict] = []

    def add(level: str, text: str) -> None:
        out.append({"level": level, "text": text})

    n, net = s["scored"], s["net_sentiment"]
    pos, neg = s["label_pct"]["Positive"], s["label_pct"]["Negative"]

    if net >= 30:
        add("good", f"Overall sentiment is strongly favourable: {pos:.0f}% positive against {neg:.0f}% negative (net {net:+.0f}).")
    elif net >= 0:
        add("info", f"Overall sentiment is mildly positive: {pos:.0f}% positive against {neg:.0f}% negative (net {net:+.0f}).")
    elif net > -20:
        add("warn", f"Overall sentiment leans slightly negative: {neg:.0f}% negative against {pos:.0f}% positive (net {net:+.0f}).")
    else:
        add("warn", f"Overall sentiment is clearly negative: {neg:.0f}% negative against {pos:.0f}% positive (net {net:+.0f}). This needs attention.")

    bg = s.get("by_group")
    if bg:
        min_rows = max(3, math.ceil(0.02 * n))
        items = [i for i in bg["items"] if i["rows"] >= min_rows]
        if len(items) >= 2:
            worst, best = min(items, key=lambda i: i["avg"]), max(items, key=lambda i: i["avg"])
            col = bg["column"]
            if worst["avg"] < 0 or worst["neg_pct"] >= 40:
                add("warn", f"Weakest {col}: \u201c{worst['name']}\u201d averages {_f(worst['avg'])} with {worst['neg_pct']:.0f}% negative across {worst['rows']} rows. Start there.")
            else:
                add("info", f"Lowest-scoring {col}: \u201c{worst['name']}\u201d at {_f(worst['avg'])}, though it is not negative.")
            add("good" if best["avg"] > 0 else "info", f"Strongest {col}: \u201c{best['name']}\u201d averages {_f(best['avg'])} ({best['pos_pct']:.0f}% positive, {best['rows']} rows). Worth learning from.")
            others = [i for i in items if i["neg_pct"] >= 40 and i is not worst]
            if others:
                add("warn", f"{len(others)} other {col} value{'s' if len(others) != 1 else ''} also {'have' if len(others) != 1 else 'has'} 40% or more negative rows.")

    tr = s.get("trend")
    if tr and tr["dated_rows"] >= 20:
        delta = tr["second_half_avg"] - tr["first_half_avg"]
        if abs(delta) >= 0.1:
            word = "improving" if delta > 0 else "declining"
            add("good" if delta > 0 else "warn", f"Sentiment is {word} over time: the later half of the period averages {_f(tr['second_half_avg'])} versus {_f(tr['first_half_avg'])} in the earlier half.")
        else:
            add("info", "Sentiment is steady over time: no meaningful shift between the first and second half of the period.")

    langs = [l for l in (s.get("by_language") or []) if l["rows"] >= 5]
    if len(langs) >= 2:
        hi, lo = max(langs, key=lambda l: l["avg"]), min(langs, key=lambda l: l["avg"])
        if hi["avg"] - lo["avg"] >= 0.25:
            add("info", f"Sentiment differs by language: \u201c{hi['name']}\u201d averages {_f(hi['avg'])} versus {_f(lo['avg'])} for \u201c{lo['name']}\u201d. That can be a real difference or lower model accuracy in one language, so spot-check a sample of both.")

    if s["strong_pct"] >= 60 and pos >= 25 and neg >= 25:
        add("info", f"Opinions are polarised: {s['strong_pct']:.0f}% of rows are strongly positive or strongly negative. The average hides this, so read both groups.")

    if s["low_confidence_pct"] >= 25:
        add("warn", f"{s['low_confidence_pct']:.0f}% of predictions have confidence below 50%. Sort by confidence and check those rows before acting on them.")
    if n < 30:
        add("info", f"Only {n} rows were scored, so a few comments can move the averages a lot. Gather more data before acting on small differences.")
    if s["empty_rows"]:
        add("info", f"{s['empty_rows']} empty row{'s were' if s['empty_rows'] != 1 else ' was'} skipped and left out of the statistics.")
    return out
