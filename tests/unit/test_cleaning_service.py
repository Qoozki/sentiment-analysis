import pandas as pd

from app.application.services.cleaning_service import clean_table, describe, normalize_text
from app.domain.value_objects.clean_options import CleanOptions


def test_normalize_removes_noise_but_keeps_wording():
    assert normalize_text("<b>Great</b> product!!  see https://x.com now") == "great product!! see now"
    assert normalize_text("Sooooo goood") == "soo good"
    assert normalize_text("I <3 it") == "i <3 it"            # not an HTML tag
    assert normalize_text("line<br>break") == "line break"   # block tags separate words


def test_normalize_arabic_marks_and_tatweel():
    assert normalize_text("رَائِـــع") == "رائع"


def test_empty_and_placeholder_rows_are_dropped():
    df = pd.DataFrame({"review": ["good", "", "   ", "N/A", None, "bad"], "product": list("abcdef")})
    out, report = clean_table(df, "review", CleanOptions(duplicates="none", normalize=False), keep=("product",))
    assert out["review"].tolist() == ["good", "bad"]
    assert report["empty_text"] == 4 and report["rows_after"] == 2


def test_duplicates_keep_first_occurrence():
    df = pd.DataFrame({"review": ["same", "same", "other"], "id": [1, 2, 3]})
    out, report = clean_table(df, "review", CleanOptions(duplicates="text"))
    assert out["id"].tolist() == [1, 3] and report["duplicates_removed"] == 1


def test_changed_rows_keep_original_wording():
    df = pd.DataFrame({"review": ["<b>Nice</b>", "plain"]})
    out, report = clean_table(df, "review", CleanOptions())
    assert report["original_column"] == "review_original"
    assert out["review_original"].tolist()[0] == "<b>Nice</b>"
    assert "Normalized the text of 1 row" in describe(report)


def test_inactive_options_change_nothing():
    df = pd.DataFrame({"review": ["a", "a", ""]})
    out, report = clean_table(df, "review", CleanOptions("none", "none", False))
    assert len(out) == 3 and not report["enabled"] and describe(report) == ""
