"""Keeps every engine in memory: loads them in background threads and hands them out by id."""
from __future__ import annotations

import logging
import threading

from app.domain.entities.engine_spec import EngineSpec
from app.domain.errors import EngineError
from app.domain.repositories.engine_repository import EngineRepository

log = logging.getLogger("sentiment")


class EngineRegistry(EngineRepository):
    def __init__(self, specs: list[EngineSpec], default_id: str | None = None) -> None:
        if not specs:
            raise ValueError("No engines are enabled.")
        self.specs = {s.id: s for s in specs}
        self.default_id = default_id if default_id in self.specs else specs[0].id
        self.engines: dict[str, object] = {}
        self.state = {s.id: {"state": "loading", "error": None} for s in specs}
        self._lock = threading.Lock()

    def start(self) -> None:
        for spec in self.specs.values():
            threading.Thread(target=self._load, args=(spec,), name=f"load-{spec.id}", daemon=True).start()

    def _load(self, spec: EngineSpec) -> None:
        try:
            engine = spec.factory()
        except Exception as exc:  # one broken engine must not take the others down
            log.exception("Engine %s failed to load", spec.id)
            with self._lock:
                self.state[spec.id] = {"state": "error", "error": str(exc) or type(exc).__name__}
            return
        with self._lock:
            self.engines[spec.id] = engine
            self.state[spec.id] = {"state": "ready", "error": None}

    def describe(self) -> list[dict]:
        with self._lock:
            return [{"id": s.id, "name": s.name, "description": s.description, **self.state[s.id]} for s in self.specs.values()]

    def get(self, engine_id: str | None, ollama_model: str | None = None):
        eid = engine_id or self.default_id
        spec = self.specs.get(eid)
        if spec is None:
            raise EngineError(400, f"Unknown model '{eid}'. Choose one of: {', '.join(self.specs)}.")
        with self._lock:
            st, engine = dict(self.state[eid]), self.engines.get(eid)
        if st["state"] == "loading":
            raise EngineError(503, f"'{spec.name}' is still loading. Pick another model or try again in a moment.")
        if st["state"] == "error":
            raise EngineError(503, f"'{spec.name}' failed to load: {st['error']}")
        return engine.bind(ollama_model) if hasattr(engine, "bind") else engine
