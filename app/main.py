"""Entry point: `uvicorn app.main:app --reload`."""
from app.presentation.api.main import app, create_app  # noqa: F401
