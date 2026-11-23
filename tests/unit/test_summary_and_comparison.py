import pandas as pd

from app.application.services.comparison_service import compare_summaries
from app.application.services.summary_service import build_insights, build_summary


def scored(scores, products=None):
    labels = ["Positive" if s >= .15 else "Negative" if s <= -.15 else "Neutral" for s in scores]
    return pd.DataFrame({
        "review": [f"r{i}" for i in range(len(scores))], "sentiment_score": scores, "sentiment_label": labels,
        "sentiment_detail": labels, "confidence": [0.9] * len(scores),
        "product": products or ["a"] * len(scores),
    })


def test_summary_counts_and_net_sentiment():
    s = build_summary(scored([0.9, 0.9, 0.0, -0.9]), "review")
    assert s["labels"] == {"Positive": 2, "Neutral": 1, "Negative": 1}
    assert s["net_sentiment"] == 25.0 and s["rows"] == s["scored"] == 4


def test_blank_rows_are_excluded_from_statistics():
    df = scored([0.9, 0.0])
    df.loc[1, "confidence"] = None  # how blanks are marked
    s = build_summary(df, "review")
    assert s["scored"] == 1 and s["empty_rows"] == 1


def test_group_insight_names_weakest_group():
    df = scored([0.9] * 10 + [-0.9] * 10, ["good"] * 10 + ["bad"] * 10)
    s = build_summary(df, "review", group_by="product")
    texts = " ".join(i["text"] for i in build_insights(s))
    assert "Weakest product" in texts and "bad" in texts


def test_comparison_detects_real_change():
    a = build_summary(scored([0.8] * 40 + [-0.8] * 10), "review")
    b = build_summary(scored([0.8] * 10 + [-0.8] * 40), "review")
    out = compare_summaries(a, b)
    assert out["avg"]["delta"] < 0 and out["significance"]["meaningful"]
    assert out["insights"][0]["level"] == "warn"


def test_comparison_same_data_is_not_meaningful():
    a = build_summary(scored([0.8, -0.8] * 20), "review")
    assert not compare_summaries(a, a)["significance"]["meaningful"]
