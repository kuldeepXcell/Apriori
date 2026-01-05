from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
from urllib.parse import urlparse, quote_plus
import socket

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


def _resolve_ipv4_host(host: str) -> str | None:
    """
    Attempt to resolve a hostname to an IPv4 address (needed on networks that block IPv6).
    """
    try:
        info = socket.getaddrinfo(host, None, family=socket.AF_INET)
    except socket.gaierror:
        logger.warning(
            "IPv4 resolution failed for host %s",
            host,
            extra={"module_name": ModuleName.ADAPTER},
        )
        return None
    if not info:
        return None
    return info[0][4][0]


def build_connection_uri() -> str:
    """
    Construct a SQLAlchemy-compatible connection string for our Postgres instance.

    Returns:
        str: postgresql:// URI built from env-driven settings.
    """
    if settings.database_url:
        return _normalize_sqlalchemy_uri(settings.database_url)
    user = quote_plus(settings.db_user)
    password = quote_plus(settings.db_password)
    host = settings.db_host
    return (
        f"postgresql+psycopg://{user}:{password}"
        f"@{host}:{settings.db_port}/{settings.db_name}?sslmode=require"
    )


def build_psycopg_dsn() -> str:
    """
    Build a DSN string suitable for psycopg.connect.
    """
    if settings.database_url:
        return settings.database_url
    hostaddr = _resolve_ipv4_host(settings.db_host)
    dsn_parts = [
        f"host={settings.db_host}",
        f"port={settings.db_port}",
        f"dbname={settings.db_name}",
        f"user={settings.db_user}",
        f"password={settings.db_password}",
        "sslmode=require",
    ]
    if hostaddr:
        dsn_parts.append(f"hostaddr={hostaddr}")
    return " ".join(dsn_parts)


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
