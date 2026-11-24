# Sentiment Analysis Platform

FastAPI website for sentiment analysis of reviews (Arabic, English and 20+ other languages):
quick analysis, bulk jobs over many files, saved history, and comparison between two analyses.

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload        # http://127.0.0.1:8000   (API docs: /docs)
```

## Test

```bash
pip install -r requirements-dev.txt
pytest
```

Tests use a fake keyword engine, so no model is downloaded. `tests/fixtures/` holds the sample review files.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `ENGINES` | `multilingual,modernbert,translate` | engines to load |
| `DEFAULT_ENGINE` | `multilingual` | engine pre-selected in the page |
| `SENTIMENT_MODEL` | `tabularisai/multilingual-sentiment-analysis` | model for `multilingual` |
| `MODERNBERT_MODEL` | `clapAI/modernBERT-base-multilingual-sentiment` | model for `modernbert` |
| `ENGLISH_MODEL` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | English classifier for `translate` |
| `OLLAMA_URL` | `http://localhost:11434` | where Ollama listens |
| `OLLAMA_MODEL` | `qwen2.5:7b` | preferred Ollama model for translation |
| `OLLAMA_WORKERS` | `4` | parallel translation requests |
| `CLEAN_NULLS` | `text` | `none` / `text` / `used` / `any` |
| `CLEAN_DUPLICATES` | `rows` | `none` / `rows` / `text` |
| `CLEAN_NORMALIZE` | `1` | `0` turns text normalization off |
| `HISTORY_DB` | `./data/history.db` | SQLite history file |
| `MAX_TEXTS` | `1000` | max texts per "type text" request |
| `BULK_MAX_ROWS` | `50000` | max rows in one bulk analysis |
| `MAX_FILES` | `20` | max files in one bulk analysis |

