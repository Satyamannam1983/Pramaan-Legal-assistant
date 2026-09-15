from contextlib import asynccontextmanager
import logging
import time
from collections import defaultdict, deque
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
_rate_limit_state: dict[str, deque[float]] = defaultdict(deque)
_rate_limited_paths = {"/api/chat", "/api/search", "/api/documents/process", "/api/documents/upload"}


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
            logger.exception("MongoDB initialisation failed in production")
            raise
        # Allow app to start even without DB (stub mode, local testing)
        logger.warning(
            "MongoDB initialisation skipped (DB may not be configured yet): %s", exc
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


@app.middleware("http")
async def production_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    started = time.monotonic()

    if request.method in {"POST", "PUT", "PATCH"}:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.max_request_body_mb * 1024 * 1024:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body exceeds the configured size limit.", "request_id": request_id},
                headers={"X-Request-ID": request_id},
            )

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
    except Exception:
        logger.exception("Unhandled request failure request_id=%s path=%s", request_id, request.url.path)
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
