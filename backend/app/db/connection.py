"""Async MongoDB connection via Motor.

DATABASE_URL examples
---------------------
Local:       mongodb://localhost:27017/pramaan
Auth:        mongodb://user:password@localhost:27017/pramaan
Atlas:       mongodb+srv://user:password@cluster.mongodb.net/pramaan
"""
from __future__ import annotations

import logging
import ssl
from urllib.parse import urlparse

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ServerSelectionTimeoutError

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client singleton — created lazily on first access
# ---------------------------------------------------------------------------
_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


def connection_failure_reason(exc: Exception) -> str:
    """Return a credential-free diagnosis suitable for logs and status output."""
    message = str(exc).casefold()
    errors = getattr(exc, "errors", None)
    if isinstance(errors, dict):
        message += " " + " ".join(
            str(error).casefold() for error in errors.values()
        )

    if (
        isinstance(exc, ssl.SSLError)
        or "ssl handshake failed" in message
        or "tlsv1_alert" in message
    ):
        return (
            "TLS negotiation failed before MongoDB authentication; check the "
            "network TLS path and Atlas availability."
        )
    if "certificate verify failed" in message:
        return "TLS certificate validation failed; verify the system trust chain without disabling TLS verification."
    if "authentication failed" in message or "bad auth" in message:
        return "MongoDB rejected authentication; verify the database user and encoded URI credentials."
    if "not authorized" in message or "unauthorized" in message:
        return "MongoDB authentication succeeded but the user lacks required database permissions."
    if isinstance(exc, ServerSelectionTimeoutError):
        return (
            "No MongoDB server was selected; check cluster state, DNS, Atlas "
            "Network Access, and outbound TCP port 27017."
        )
    return "MongoDB initialization failed; check database availability, user permissions, and index compatibility."


def _resolve_db_name() -> str:
    """Extract database name from URL path, falling back to settings.database_name."""
    if settings.database_name and settings.database_name != "pramaan":
        return settings.database_name
    parsed = urlparse(settings.database_url)
    path_name = parsed.path.lstrip("/").split("?")[0]
    return path_name or settings.database_name or "pramaan"


def get_client() -> AsyncIOMotorClient:
    """Return the shared Motor client (created once)."""
    global _client  # noqa: PLW0603
    if _client is None:
        _client = AsyncIOMotorClient(
            settings.database_url,
            # Fail fast when the DB is not available — prevents hanging in tests
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=5000,
        )
        parsed = urlparse(settings.database_url)
        logger.debug("MongoDB client created for host=%s", parsed.hostname or "configured-host")
    return _client


def get_db() -> AsyncIOMotorDatabase:
    """Return the application database object."""
    global _db  # noqa: PLW0603
    if _db is None:
        _db = get_client()[_resolve_db_name()]
        logger.debug("MongoDB database: %s", _resolve_db_name())
    return _db


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """Create MongoDB indexes.

    Safe to call multiple times — MongoDB createIndex is idempotent.
    """
    db = get_db()

    # documents collection
    await db.documents.create_index("document_id", unique=True, background=True)
    logger.debug("Index: documents.document_id (unique)")

    # document_chunks collection
    await db.document_chunks.create_index("chunk_id", unique=True, background=True)
    await db.document_chunks.create_index("document_id", background=True)
    logger.debug("Indexes: document_chunks.chunk_id (unique), document_id")

    logger.info(
        "MongoDB initialised — db=%s indexes ready",
        _resolve_db_name(),
    )


async def check_connection() -> bool:
    """Return True if MongoDB is reachable."""
    try:
        await get_client().admin.command("ping")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "MongoDB health check failed (error_type=%s, reason=%s).",
            type(exc).__name__,
            connection_failure_reason(exc),
        )
        return False


async def close_connection() -> None:
    """Close the Motor client on shutdown."""
    global _client, _db  # noqa: PLW0603
    if _client is not None:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB connection closed")
