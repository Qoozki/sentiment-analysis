import pandas as pd

from app.application.services.table_service import guess_date_column, guess_text_column, parse_table, to_records
from app.domain.errors import TableError
from tests.conftest import fixture_bytes


def test_parse_csv_with_bom_and_arabic():
    df = parse_table("id,review\n1,رائع جدا\n".encode("utf-8-sig"), "a.csv")
    assert list(df.columns) == ["id", "review"] and df.loc[0, "review"] == "رائع جدا"


def test_parse_cp1256_fallback():
    df = parse_table("id,review\n1,رائع\n".encode("cp1256"), "a.csv")
    assert df.loc[0, "review"] == "رائع"


def test_parse_limits_and_errors():
    for data, kw in [(b"a,b\n1,2\n3,4\n", {"max_rows": 1}), (b"x" * 10, {"max_bytes": 5}), (b"", {})]:
        try:
            parse_table(data, "a.csv", **kw)
        except TableError:
            continue
        raise AssertionError("expected TableError")


def test_parse_xlsx_fixture_matches_csv():
    a = parse_table(fixture_bytes("all_reviews.xlsx"), "all_reviews.xlsx")
    b = parse_table(fixture_bytes("all_reviews.csv"), "all_reviews.csv")
    assert len(a) == len(b) == 450


def test_guess_columns_on_real_fixture():
    df = parse_table(fixture_bytes("all_reviews.csv"), "all_reviews.csv")
    assert guess_text_column(df) == "review"
    assert guess_date_column(df) == "date"


def test_guess_text_column_rejects_numbers_only():
    try:
        guess_text_column(pd.DataFrame({"a": [1, 2], "b": [True, False]}))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_to_records_is_json_safe():
    assert to_records(pd.DataFrame({"a": [1.0, float("nan")]})) == [{"a": 1.0}, {"a": None}]
