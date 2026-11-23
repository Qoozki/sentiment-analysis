"""All configuration, read once from environment variables (see README for the list)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from app.application.services.table_service import BULK_MAX_BYTES
from app.domain.value_objects.clean_options import CleanOptions

BASE_DIR = Path(__file__).resolve().parents[3]  # project root

DEFAULT_MODEL = "tabularisai/multilingual-sentiment-analysis"
MODERNBERT_MODEL = "clapAI/modernBERT-base-multilingual-sentiment"
ENGLISH_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # free English model: negative / neutral / positive


@dataclass(frozen=True)
class Settings:
    engines: tuple[str, ...] = ("multilingual", "modernbert", "translate")
    default_engine: str = "multilingual"
    sentiment_model: str = DEFAULT_MODEL
    modernbert_model: str = MODERNBERT_MODEL
    english_model: str = ENGLISH_MODEL
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_workers: int = 4
    # Data cleaning is automatic: every analysis normalizes the text and drops empty reviews and duplicates.
    clean: CleanOptions = field(default_factory=CleanOptions)
    history_db: str = str(BASE_DIR / "data" / "history.db")
    max_texts: int = 1000
    bulk_max_rows: int = 50000
    max_files: int = 20
    bulk_max_bytes: int = BULK_MAX_BYTES
    ui_dir: Path = BASE_DIR / "ui"

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.getenv
        engines = tuple(e.strip() for e in env("ENGINES", "multilingual,modernbert,translate").split(",") if e.strip())
        return cls(
            engines=engines or cls.engines,
            default_engine=env("DEFAULT_ENGINE", "multilingual"),
            sentiment_model=env("SENTIMENT_MODEL", DEFAULT_MODEL),
            modernbert_model=env("MODERNBERT_MODEL", MODERNBERT_MODEL),
            english_model=env("ENGLISH_MODEL", ENGLISH_MODEL),
            ollama_url=env("OLLAMA_URL", "http://localhost:11434"),
            ollama_model=env("OLLAMA_MODEL", "qwen2.5:7b"),
            ollama_workers=int(env("OLLAMA_WORKERS", "4")),
            clean=CleanOptions(env("CLEAN_NULLS", "text"), env("CLEAN_DUPLICATES", "rows"),
                               env("CLEAN_NORMALIZE", "1").strip().lower() not in ("0", "false", "no", "off")),
            history_db=env("HISTORY_DB", str(BASE_DIR / "data" / "history.db")),
            max_texts=int(env("MAX_TEXTS", "1000")),
            bulk_max_rows=int(env("BULK_MAX_ROWS", "50000")),
            max_files=int(env("MAX_FILES", "20")),
        )
