from app.domain.entities.run_meta import RunMeta

SUMMARY = {"label_pct": {"Positive": 50, "Neutral": 0, "Negative": 50}, "rows": 0, "stats": {"mean": 0.0}, "net_sentiment": 0}


def save(runs, n, name="x"):
    rows = [{"text": f"r{i}", "sentiment_label": "Positive" if i % 2 else "Negative", "sentiment_score": i / n} for i in range(n)]
    return runs.save_run(RunMeta(name=name, kind="text", text_col="text", columns=["text"], files=["f.csv"]),
                         {**SUMMARY, "rows": n}, rows)


def test_save_get_list_delete(runs):
    rid = save(runs, 4, "first")
    run = runs.get_run(rid)
    assert run["name"] == "first" and run["columns"] == ["text"] and run["files"] == ["f.csv"]
    assert [r["id"] for r in runs.list_runs()] == [rid]
    assert runs.delete_run(rid) and runs.get_run(rid) is None and not runs.delete_run(rid)


def test_filter_search_and_sort(runs):
    rid = save(runs, 10)
    total, rows = runs.get_rows(rid, label="Positive")
    assert total == 5 and all(r["sentiment_label"] == "Positive" for r in rows)
    assert runs.get_rows(rid, q="r7")[0] == 1
    assert runs.get_rows(rid, q="100%")[0] == 0           # % is escaped, not a wildcard
    top = runs.get_rows(rid, limit=1, sort="score_desc")[1][0]
    assert top["text"] == "r9"


def test_iter_rows_returns_every_row_for_large_runs(runs):
    """Regression: the old iterator advanced by its batch size (1000) while pages were capped at 500."""
    rid = save(runs, 1200)
    texts = [r["text"] for r in runs.iter_rows(rid)]
    assert len(texts) == 1200 and texts == [f"r{i}" for i in range(1200)]


def test_page_size_is_capped(runs):
    rid = save(runs, 600)
    assert len(runs.get_rows(rid, limit=10_000)[1]) == 500
