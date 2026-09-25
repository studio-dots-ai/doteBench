"""Transport and application lifecycle helpers independent of service protocols."""

import threading
from contextlib import asynccontextmanager
from functools import wraps


def request_json(session, method, url, *, payload=None, timeout):
    kwargs = {"timeout": timeout}
    if payload is not None:
        kwargs["json"] = payload
    response = getattr(session, method.lower())(url, **kwargs)
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict) or result.get("error"):
        raise RuntimeError(f"Service returned an invalid response: {result}")
    return result


def service_app(*, title, health, close=None):
    from fastapi import FastAPI

    from dotebench import RELEASE

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            if close is not None:
                close()

    app = FastAPI(version=RELEASE, title=title, lifespan=lifespan)
    app.get("/health")(health)
    return app


def json_endpoint(handler):
    """Adapt an object request handler to FastAPI without choosing its route."""
    from fastapi import HTTPException

    def invoke(payload: dict):
        try:
            return handler(payload)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc

    return invoke


def serialized(handler):
    lock = threading.Lock()

    @wraps(handler)
    def invoke(*args, **kwargs):
        with lock:
            return handler(*args, **kwargs)

    return invoke
