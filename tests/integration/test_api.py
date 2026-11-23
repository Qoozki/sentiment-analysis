"""End-to-end through HTTP with the fake engine. Skipped automatically when FastAPI is not installed."""
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.infrastructure.config.settings import Settings  # noqa: E402
from app.presentation.api.dependencies import build_container  # noqa: E402
from app.presentation.api.main import create_app  # noqa: E402
from tests.conftest import FIXTURES, FakeCatalog, FakeEngines  # noqa: E402


@pytest.fixture
def client(tmp_path):
    settings = Settings(history_db=str(tmp_path / "h.db"), max_texts=3, max_files=2)
    container = build_container(settings, engines=FakeEngines(), catalog=FakeCatalog())
    with TestClient(create_app(container=container)) as c:
        yield c


def csv_file(name):
    return (name, (FIXTURES / name).read_bytes(), "text/csv")


def test_page_and_health(client):
    assert "<html" in client.get("/").text.lower()
    body = client.get("/api/health").json()
    assert body["ready"] and body["default_engine"] == "fake" and body["limits"]["files"] == 2
    assert client.get("/api/ollama/models").json()["reachable"]


def test_analyze_texts_and_errors(client):
    ok = client.post("/api/analyze", json={"texts": ["I love it", "terrible"]})
    assert ok.status_code == 200 and ok.json()["summary"]["rows"] == 2 and len(ok.json()["results"]) == 2
    assert client.post("/api/analyze?include_rows=false", json={"texts": ["fine"]}).json().keys() >= {"run_id", "summary"}
    blank = client.post("/api/analyze", json={"texts": ["  "]})
    assert blank.status_code == 400 and blank.json()["detail"] == "Add at least one non-empty text."
    assert client.post("/api/analyze", json={"texts": list("abcd")}).status_code == 413
    unknown = client.post("/api/analyze", json={"texts": ["x"], "engine": "nope"})
    assert unknown.status_code == 400 and "Unknown model" in unknown.json()["detail"]


def test_preview_and_analyze_file(client):
    prev = client.post("/api/preview", files={"file": csv_file("all_reviews.csv")}).json()
    assert prev["text_col"] == "review" and prev["rows"] == 450
    res = client.post("/api/analyze-file", files={"file": csv_file("messy_reviews.csv")},
                      data={"group_by": "product", "date_col": "date"}).json()
    assert res["text_col"] == "review" and res["summary"]["by_group"]["column"] == "product"


def test_bulk_job_then_history_export_compare_delete(client):
    files = [("files", csv_file(f)) for f in ("reviews_2026-08.csv", "reviews_2026-09.csv")]
    job_id = client.post("/api/bulk", files=files, data={"text_col": "review", "group_by": "product"}).json()["job_id"]
    for _ in range(50):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert job["state"] == "done" and job["total"] == 300

    assert len(client.get("/api/runs").json()["runs"]) == 1
    rid = job["run_id"]
    assert client.get(f"/api/runs/{rid}").json()["kind"] == "bulk"
    page = client.get(f"/api/runs/{rid}/rows", params={"limit": 10, "label": "Positive"}).json()
    assert len(page["rows"]) == 10 and page["total"] > 10

    export = client.get(f"/api/runs/{rid}/export.csv")
    assert export.status_code == 200 and export.text.startswith("\ufeff")
    assert len(export.text.strip().splitlines()) == 301  # header + every row

    other = client.post("/api/analyze", json={"texts": ["I love it"]}).json()["run_id"]
    assert client.get("/api/compare", params={"base": rid, "latest": other}).status_code == 200
    assert client.get("/api/compare", params={"base": rid, "latest": rid}).status_code == 400

    assert client.delete(f"/api/runs/{rid}").json() == {"deleted": rid}
    assert client.get(f"/api/runs/{rid}").status_code == 404
    assert client.get("/api/jobs/unknown").status_code == 404


def test_bulk_rejects_too_many_files(client):
    files = [("files", csv_file("reviews_2026-08.csv"))] * 3
    r = client.post("/api/bulk", files=files, data={"text_col": "review"})
    assert r.status_code == 400 and "Too many files" in r.json()["detail"]
