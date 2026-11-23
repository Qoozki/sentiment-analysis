"""Sentiment analysis platform. Run the website with `uvicorn app.main:app`."""


def __getattr__(name: str):
    # Keeps the old command `uvicorn app:app` working. Lazy, so importing app.domain etc. never needs FastAPI.
    if name == "app":
        from app.main import app
        return app
    raise AttributeError(f"module 'app' has no attribute {name!r}")
