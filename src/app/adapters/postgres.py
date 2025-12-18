from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg

from app.config.settings import settings
from app.core.logging import ModuleName, get_logger

logger = get_logger(__name__)


def build_connection_uri() -> str:
    """
    Construct a SQLAlchemy-compatible connection string for our Postgres instance.

    Returns:
        str: postgresql:// URI built from env-driven settings.
    """
    return (
        f"postgresql+psycopg://{settings.db_user}:{settings.db_password}"
        f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
    )


def build_psycopg_dsn() -> str:
    """
    Build a DSN string suitable for psycopg.connect.
    """
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
    connection_descriptor = (
        f"host={settings.db_host} port={settings.db_port} "
        f"db={settings.db_name} user={settings.db_user}"
    )
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
