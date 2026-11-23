"""The built-in engines. Settings.engines (ENGINES=...) limits which of them are loaded."""
from __future__ import annotations

from typing import Callable

from app.domain.entities.engine_spec import EngineSpec
from app.infrastructure.config.settings import Settings
from app.infrastructure.ml.hf_sentiment_model import SentimentModel
from app.infrastructure.ml.ollama_client import OllamaClient
from app.infrastructure.ml.translate_engine import TranslateEngine


def default_specs(settings: Settings, client: OllamaClient) -> list[EngineSpec]:
    multilingual, modern, english = settings.sentiment_model, settings.modernbert_model, settings.english_model

    def plain(model_id: str) -> Callable[[], object]:
        return lambda: SentimentModel(model_id)

    specs = [
        EngineSpec("multilingual", "Multilingual (tabularisai)",
                   f"{multilingual}: reads many languages directly, five sentiment levels.", plain(multilingual)),
        EngineSpec("modernbert", "ModernBERT multilingual",
                   f"{modern}: reads many languages directly.", plain(modern)),
        EngineSpec("translate", "Translate with Ollama, then classify",
                   f"Text in other languages is translated to English by a local Ollama model, then scored by {english}.",
                   lambda: TranslateEngine(SentimentModel(english), client, settings.ollama_model, english)),
    ]
    return [s for s in specs if s.id in settings.engines] or specs
