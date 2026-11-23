"""SQLite storage for analysis history (standard library only)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from app.domain.entities.run_meta import RunMeta
from app.domain.repositories.run_repository import RunRepository
from app.domain.value_objects.sentiment_scale import LABELS as _LABEL_LIST

LABELS = set(_LABEL_LIST)


def _like(q: str) -> str:
    return "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


class SqliteRunRepository(RunRepository):
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(self.db_path, timeout=30)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def init(self) -> None:
        with self.conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    text_col TEXT, group_col TEXT, date_col TEXT, model TEXT,
                    columns TEXT, files TEXT,
                    rows INTEGER, avg_score REAL, net_sentiment REAL,
                    pos_pct REAL, neu_pct REAL, neg_pct REAL,
                    summary TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS run_rows (
                    run_id INTEGER NOT NULL,
                    idx INTEGER NOT NULL,
                    label TEXT, score REAL, text TEXT, data TEXT NOT NULL,
                    PRIMARY KEY (run_id, idx)
                );
                """
            )

    def save_run(self, meta: RunMeta, summary: dict, rows: list[dict]) -> int:
        lp = summary["label_pct"]
        text_col = meta.text_col
        with self.conn() as c:
            cur = c.execute(
                """INSERT INTO runs (created_at,name,kind,text_col,group_col,date_col,model,columns,files,
                                     rows,avg_score,net_sentiment,pos_pct,neu_pct,neg_pct,summary)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    meta.name, meta.kind, text_col, meta.group_col, meta.date_col, meta.model,
                    json.dumps(meta.columns, ensure_ascii=False),
                    json.dumps(meta.files, ensure_ascii=False),
                    summary["rows"], summary["stats"]["mean"], summary["net_sentiment"],
                    lp["Positive"], lp["Neutral"], lp["Negative"],
                    json.dumps(summary, ensure_ascii=False),
                ),
            )
            run_id = cur.lastrowid
            c.executemany(
                "INSERT INTO run_rows (run_id,idx,label,score,text,data) VALUES (?,?,?,?,?,?)",
                (
                    (run_id, i, r.get("sentiment_label"), r.get("sentiment_score"),
                     str(r.get(text_col) or ""), json.dumps(r, ensure_ascii=False))
                    for i, r in enumerate(rows)
                ),
            )
        return int(run_id)

    def list_runs(self, limit: int = 200) -> list[dict]:
        with self.conn() as c:
            cur = c.execute(
                """SELECT id,created_at,name,kind,model,files,rows,avg_score,net_sentiment,pos_pct,neu_pct,neg_pct
                   FROM runs ORDER BY id DESC LIMIT ?""",
                (limit,),
            )
            out = []
            for r in cur:
                d = dict(r)
                d["files"] = json.loads(d["files"] or "[]")
                out.append(d)
            return out

    def get_run(self, run_id: int) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["summary"] = json.loads(d["summary"])
        d["columns"] = json.loads(d["columns"] or "[]")
        d["files"] = json.loads(d["files"] or "[]")
        return d

    def get_rows(self, run_id: int, offset: int = 0, limit: int = 50, label: str | None = None,
                 q: str | None = None, sort: str = "original") -> tuple[int, list[dict]]:
        where, args = ["run_id=?"], [run_id]
        if label in LABELS:
            where.append("label=?")
            args.append(label)
        if q:
            where.append("data LIKE ? ESCAPE '\\'")
            args.append(_like(q))
        order = {"score_desc": "score DESC, idx", "score_asc": "score ASC, idx"}.get(sort, "idx")
        w = " AND ".join(where)
        limit = max(1, min(limit, 500))
        with self.conn() as c:
            total = c.execute(f"SELECT COUNT(*) FROM run_rows WHERE {w}", args).fetchone()[0]
            cur = c.execute(f"SELECT data FROM run_rows WHERE {w} ORDER BY {order} LIMIT ? OFFSET ?",
                            args + [limit, max(0, offset)])
            return total, [json.loads(r["data"]) for r in cur]

    def iter_rows(self, run_id: int, batch: int = 500) -> Iterator[dict]:
        offset = 0
        while True:
            _, rows = self.get_rows(run_id, offset=offset, limit=batch)
            if not rows:
                return
            yield from rows
            offset += len(rows)  # not `batch`: get_rows caps its page size at 500, so rows would be skipped

    def delete_run(self, run_id: int) -> bool:
        with self.conn() as c:
            c.execute("DELETE FROM run_rows WHERE run_id=?", (run_id,))
            return c.execute("DELETE FROM runs WHERE id=?", (run_id,)).rowcount > 0
