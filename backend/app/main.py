from contextlib import asynccontextmanager
import logging
import time
from collections import defaultdict, deque
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
_rate_limit_state: dict[str, deque[float]] = defaultdict(deque)
_rate_limited_paths = {"/api/chat", "/api/search", "/api/documents/process", "/api/documents/upload"}


class RequestSizeLimitMiddleware:
    """Bound request-body buffering, including requests without Content-Length."""

    def __init__(self, app, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        declared_length = headers.get(b"content-length")
        if declared_length is not None:
            try:
                content_length = int(declared_length)
            except ValueError:
                await JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid Content-Length header."},
                )(scope, receive, send)
                return
            if content_length < 0:
                await JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid Content-Length header."},
                )(scope, receive, send)
                return
            if content_length > self.max_body_bytes:
                await JSONResponse(
                    status_code=413,
                    content={"detail": "Request body exceeds the configured size limit."},
                )(scope, receive, send)
                return

        total = 0
        async def limited_receive():
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_body_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail="Request body exceeds the configured size limit.",
                    )
            return message

        response_started = False

        async def track_response(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, track_response)
        except HTTPException as exc:
            if exc.status_code != 413 or response_started:
                raise
            await JSONResponse(
                status_code=413,
                content={"detail": "Request body exceeds the configured size limit."},
            )(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise MongoDB on startup; close connection pool on shutdown."""
    # Startup
    try:
        from app.db.connection import init_db  # noqa: PLC0415
        await init_db()
        logger.info("MongoDB ready")
    except Exception as exc:  # noqa: BLE001
        if settings.app_env.lower() == "production":
            logger.error(
                "MongoDB initialisation failed in production (error_type=%s).",
                type(exc).__name__,
            )
            from app.db.connection import close_connection  # noqa: PLC0415
            await close_connection()
            raise RuntimeError("MongoDB initialisation failed.") from None
        # Allow app to start even without DB (stub mode, local testing)
        logger.warning(
            "MongoDB initialisation skipped (error_type=%s).",
            type(exc).__name__,
        )
    yield
    # Shutdown
    from app.db.connection import close_connection  # noqa: PLC0415
    await close_connection()
    logger.info("MongoDB connection closed")


app = FastAPI(
    title=settings.app_name,
    version="0.3.0",
    description="Indian Legal AI Research Assistant — semantic retrieval with MongoDB",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Request-ID"],
)
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_body_bytes=settings.max_request_body_mb * 1024 * 1024,
)


@app.middleware("http")
async def production_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    started = time.monotonic()

    if request.url.path in _rate_limited_paths:
        now = time.monotonic()
        client_key = request.client.host if request.client else "unknown"
        timestamps = _rate_limit_state[client_key]
        while timestamps and now - timestamps[0] > settings.rate_limit_window_seconds:
            timestamps.popleft()
        if len(timestamps) >= settings.rate_limit_requests:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again later.", "request_id": request_id},
                headers={"Retry-After": str(settings.rate_limit_window_seconds), "X-Request-ID": request_id},
            )
        timestamps.append(now)

    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error(
            "Unhandled request failure request_id=%s path=%s error_type=%s",
            request_id,
            request.url.path,
            type(exc).__name__,
        )
        response = JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred.", "request_id": request_id},
        )

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
        request_id, request.method, request.url.path, response.status_code,
        (time.monotonic() - started) * 1000,
    )
    return response

app.include_router(router, prefix="/api")
app.include_router(router, prefix=settings.api_prefix)
