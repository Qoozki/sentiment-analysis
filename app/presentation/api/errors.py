"""Maps application errors to HTTP responses. The body is {"detail": message}, which the UI shows."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.domain.errors import EngineError, LimitExceededError, NotFoundError


def register_error_handlers(app: FastAPI) -> None:
    def reply(status: int, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=status)

    # Starlette picks the handler of the most specific class, so EngineError and LimitExceededError
    # (both ValueErrors) win over the generic ValueError -> 400 below.
    @app.exception_handler(EngineError)
    async def _engine(_: Request, exc: EngineError):
        return reply(exc.status, exc)

    @app.exception_handler(LimitExceededError)
    async def _too_large(_: Request, exc: LimitExceededError):
        return reply(413, exc)

    @app.exception_handler(NotFoundError)
    async def _missing(_: Request, exc: NotFoundError):
        return reply(404, exc)

    @app.exception_handler(ValueError)  # TableError, InvalidRequestError, TranslationError, bad input...
    async def _user_error(_: Request, exc: ValueError):
        return reply(400, exc)
