import pytest

from app.application.services.analysis_service import AnalysisPipeline
from app.application.services.job_service import JobRegistry
from app.application.use_cases.analyze_table import AnalyzeTable
from app.application.use_cases.analyze_texts import AnalyzeTexts
from app.application.use_cases.bulk_analysis import BulkAnalysis
from app.application.use_cases.compare_runs import CompareRuns
from app.application.use_cases.history import DeleteRun, ExportRun, GetRun, GetRunRows
from app.application.use_cases.preview_table import PreviewTable
from app.domain.errors import InvalidRequestError, LimitExceededError, NotFoundError, TableError
from app.domain.value_objects.clean_options import CleanOptions
from tests.conftest import fixture_bytes

CLEAN = CleanOptions()
MONTHS = ["reviews_2026-08.csv", "reviews_2026-09.csv", "reviews_2026-10.csv"]


def pipeline(runs):
    return AnalysisPipeline(runs)


def test_analyze_texts_saves_run_and_skips_blanks(runs, engines):
    uc = AnalyzeTexts(pipeline(runs), engines, CLEAN, max_texts=5)
    out = uc.execute(["I love it", "terrible", "   ", ""], name="demo")
    assert out.summary["rows"] == 2 and runs.get_run(out.run_id)["name"] == "demo"
    assert runs.get_run(out.run_id)["model"] == "fake-engine"


def test_analyze_texts_validation(runs, engines):
    uc = AnalyzeTexts(pipeline(runs), engines, CLEAN, max_texts=2)
    with pytest.raises(InvalidRequestError):
        uc.execute(["  "])
    with pytest.raises(LimitExceededError):
        uc.execute(["a", "b", "c"])


def test_analyze_table_guesses_columns_and_cleans(runs, engines):
    uc = AnalyzeTable(pipeline(runs), engines, CLEAN)
    text_col, out = uc.execute(fixture_bytes("messy_reviews.csv"), "messy_reviews.csv", group_by="product", date_col="date")
    assert text_col == "review"
    assert out.summary["cleaning"]["rows_after"] < out.summary["cleaning"]["rows_before"]  # empties/duplicates removed
    assert out.summary["by_group"]["column"] == "product" and out.summary["trend"] is not None
    assert "review_original" in out.columns


def test_analyze_table_rejects_unknown_columns(runs, engines):
    uc = AnalyzeTable(pipeline(runs), engines, CLEAN)
    with pytest.raises(TableError):
        uc.execute(fixture_bytes("all_reviews.csv"), "a.csv", text_col="nope")
    with pytest.raises(TableError):
        uc.execute(fixture_bytes("all_reviews.csv"), "a.csv", group_by="nope")


def test_preview_suggests_columns():
    out = PreviewTable(50000, 50 * 1024 * 1024).execute(fixture_bytes("all_reviews.csv"), "all_reviews.csv")
    assert out["text_col"] == "review" and out["date_col"] == "date" and out["rows"] == 450 and len(out["sample"]) == 3


def make_bulk(runs, engines, max_files=20):
    return BulkAnalysis(pipeline(runs), engines, JobRegistry(), CLEAN, 50000, max_files, 50 * 1024 * 1024)


def test_bulk_merges_files_and_reports_progress(runs, engines):
    bulk = make_bulk(runs, engines)
    job_id, engine = bulk.start(3, None, None)
    bulk.run(job_id, [(f, fixture_bytes(f)) for f in MONTHS], engine, "review", "product", "date", "q3")
    job = bulk.status(job_id)
    assert job["state"] == "done" and job["done"] == job["total"] == 450
    run = runs.get_run(job["run_id"])
    assert run["kind"] == "bulk" and run["files"] == MONTHS
    assert {s["name"] for s in run["summary"]["by_source"]} == set(MONTHS)
    assert len(run["summary"]["by_source"]) == 3


def test_bulk_errors_land_in_the_job_not_as_exceptions(runs, engines):
    bulk = make_bulk(runs, engines)
    job_id, engine = bulk.start(1, None, None)
    bulk.run(job_id, [(MONTHS[0], fixture_bytes(MONTHS[0]))], engine, "nope", None, None, None)
    job = bulk.status(job_id)
    assert job["state"] == "error" and "no column named 'nope'" in job["error"]


def test_bulk_start_validation_and_unknown_job(runs, engines):
    bulk = make_bulk(runs, engines, max_files=2)
    with pytest.raises(InvalidRequestError):
        bulk.start(0, None, None)
    with pytest.raises(InvalidRequestError):
        bulk.start(3, None, None)
    with pytest.raises(NotFoundError):
        bulk.status("missing")


def run_month(runs, engines, name):
    out = AnalyzeTable(pipeline(runs), engines, CLEAN).execute(fixture_bytes(name), name, group_by="product")[1]
    return out.run_id


def test_compare_orders_by_age_and_flags_model_change(runs, engines):
    a, b = run_month(runs, engines, MONTHS[0]), run_month(runs, engines, MONTHS[1])
    cmp = CompareRuns(runs)
    out = cmp.execute(b, a)  # arguments in the "wrong" order
    assert out["base"]["id"] == a and out["latest"]["id"] == b and out["groups"] is not None
    with pytest.raises(InvalidRequestError):
        cmp.execute(a, a)
    with pytest.raises(NotFoundError):
        cmp.execute(a, 9999)


def test_history_use_cases(runs, engines):
    rid = run_month(runs, engines, MONTHS[0])
    assert GetRun(runs).execute(rid)["summary"]["rows"] == 150
    assert GetRunRows(runs).execute(rid, limit=5)["total"] == 150
    assert len(list(ExportRun(runs).execute(rid))) == 150
    assert DeleteRun(runs).execute(rid) == {"deleted": rid}
    for uc in (GetRun(runs), ExportRun(runs), DeleteRun(runs)):
        with pytest.raises(NotFoundError):
            uc.execute(rid)


def test_export_of_a_large_table_is_complete(runs, engines):
    out = AnalyzeTable(pipeline(runs), engines, CleanOptions("none", "none", False)).execute(
        fixture_bytes("all_reviews.csv"), "all_reviews.csv")[1]
    assert len(list(ExportRun(runs).execute(out.run_id))) == 450
