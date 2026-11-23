"""Ports: a sentiment engine, and the place engines are looked up by id."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Protocol

from app.domain.entities.prediction import Prediction


class SentimentEngine(Protocol):
    """Any object with predict(). Optional extras: lock (threading.Lock), label (str),
    describe() -> str and bind(ollama_model) -> engine."""

    def predict(self, texts: list[str], progress: Callable[[int, int], None] | None = None) -> list[Prediction]: ...


class EngineRepository(ABC):
    default_id: str

    @abstractmethod
    def start(self) -> None:
        """Begin loading every engine in the background."""

    @abstractmethod
    def describe(self) -> list[dict]: ...

    @abstractmethod
    def get(self, engine_id: str | None, ollama_model: str | None = None) -> SentimentEngine:
        """The engine for one request. Raises EngineError when it is unknown or not ready."""
