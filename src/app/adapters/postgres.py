from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
from urllib.parse import urlparse

import psycopg

from app.config.settings import settings
from app.core.logging import ModuleName, get_logger

logger = get_logger(__name__)


def _normalize_sqlalchemy_uri(uri: str) -> str:
    """Ensure SQLAlchemy URIs always target the psycopg driver."""
    if "://" not in uri:
        return uri
    scheme, remainder = uri.split("://", 1)
    if "+" in scheme:
        return uri
    if scheme.lower() in {"postgresql", "postgres"}:
        return f"postgresql+psycopg://{remainder}"
    return uri


def _describe_connection_target() -> str:
    """Human-readable description for logging without leaking secrets."""
    if settings.database_url:
        parsed = urlparse(settings.database_url)
        host = parsed.hostname or settings.db_host
        port = parsed.port or settings.db_port
        dbname = parsed.path.lstrip("/") or settings.db_name
        return f"url host={host} port={port} db={dbname}"
    return (
        f"host={settings.db_host} port={settings.db_port} "
        f"db={settings.db_name} user={settings.db_user}"
    )


def build_connection_uri() -> str:
    """
    Construct a SQLAlchemy-compatible connection string for our Postgres instance.

    Returns:
        str: postgresql:// URI built from env-driven settings.
    """
    if settings.database_url:
        return _normalize_sqlalchemy_uri(settings.database_url)
    return (
        f"postgresql+psycopg://{settings.db_user}:{settings.db_password}"
        f"@{settings.db_host}:{settings.db_port}/{settings.db_name}?sslmode=require"
    )


def build_psycopg_dsn() -> str:
    """
    Build a DSN string suitable for psycopg.connect.
    """
    if settings.database_url:
        return settings.database_url
    return (
        f"host={settings.db_host} port={settings.db_port} dbname={settings.db_name} "
        f"user={settings.db_user} password={settings.db_password}"
    )


@contextmanager
def get_connection(*, autocommit: bool = True) -> Iterator[psycopg.Connection]:
    """
    Context manager that yields a psycopg connection.

    Args:
        autocommit: Whether to enable autocommit on the connection.
    """
    connection_descriptor = _describe_connection_target()
    conn: psycopg.Connection | None = None
    try:
        conn = psycopg.connect(build_psycopg_dsn())
        conn.autocommit = autocommit
        logger.debug(
            "Opened Postgres connection (%s)",
            connection_descriptor,
            extra={"module_name": ModuleName.ADAPTER},
        )
        yield conn
    except psycopg.OperationalError:
        logger.error(
            "Failed to connect to Postgres (%s)",
            connection_descriptor,
            extra={"module_name": ModuleName.ADAPTER},
            exc_info=True,
        )
        raise
    finally:
        if conn is not None:
            conn.close()
