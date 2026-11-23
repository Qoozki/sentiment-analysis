"""Use case: read an uploaded table and suggest its text and date columns."""
from __future__ import annotations

from app.application.services.table_service import guess_date_column, guess_text_column, parse_table, to_records


class PreviewTable:
    def __init__(self, max_rows: int, max_bytes: int) -> None:
        self.max_rows, self.max_bytes = max_rows, max_bytes

    def execute(self, data: bytes, filename: str | None) -> dict:
        df = parse_table(data, filename or "", max_rows=self.max_rows, max_bytes=self.max_bytes)
        try:
            text_col = guess_text_column(df)
        except ValueError:
            text_col = None  # nothing looks like text; the page lets the user choose
        return {"columns": list(df.columns), "rows": len(df), "text_col": text_col,
                "date_col": guess_date_column(df), "sample": to_records(df.head(3))}
