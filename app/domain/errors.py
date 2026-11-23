"""Errors the whole application shares. The presentation layer maps them to HTTP responses."""
from __future__ import annotations


class InvalidRequestError(ValueError):
    """The request itself is wrong (nothing to analyze, too many files...). Shown to the user, HTTP 400."""


class TableError(ValueError):
    """A problem with the uploaded data that the user can fix."""


class LimitExceededError(ValueError):
    """A size limit was exceeded. HTTP 413."""


class NotFoundError(LookupError):
    """A saved analysis or job does not exist. HTTP 404."""


class EngineError(ValueError):
    """The requested engine does not exist or is not ready. `status` is the HTTP status to use."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class TranslationError(ValueError):
    """A problem with the translation service that the user can act on."""


class TranslationUnavailable(TranslationError):
    """The translation service cannot be reached at all; there is no point trying the remaining rows."""
