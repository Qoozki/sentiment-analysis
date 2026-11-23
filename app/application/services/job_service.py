"""In-memory tracking of bulk jobs. The browser polls these for progress."""
from __future__ import annotations

import threading
import time
import uuid


class JobRegistry:
    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def create(self) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            for k in [k for k, v in self._jobs.items() if time.time() - v["created"] > self._ttl]:
                del self._jobs[k]  # forget old jobs; finished runs stay in history
            self._jobs[job_id] = {"created": time.time(), "state": "queued", "done": 0, "total": 0,
                                  "run_id": None, "error": None}
        return job_id

    def update(self, job_id: str, **kw) -> None:
        with self._lock:
            self._jobs[job_id].update(kw)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            j = self._jobs.get(job_id)
            return dict(j) if j else None
