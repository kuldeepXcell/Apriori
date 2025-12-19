"""LangSmith tracing configuration."""

from __future__ import annotations

import os

from app.config.settings import Settings
from app.core.logging import ModuleName, get_logger

logger = get_logger(__name__)
_configured = False


def _ellipsize_value(value: str | None, show_prefix: int = 8, show_suffix: int = 4) -> str:
    """Ellipsize sensitive values like API keys for logging."""
    if not value:
        return "not set"
    if len(value) <= show_prefix + show_suffix:
        return value
    return f"{value[:show_prefix]}...{value[-show_suffix:]}"


def configure_langsmith(settings: Settings) -> None:
    """
    Apply LangSmith env vars once per process based on settings.
    Expects creds to be provided via env (see Settings) or .env.
    """
    global _configured
    if _configured:
        return
    _configured = True

    if not settings.langsmith_tracing:
        logger.info("LangSmith tracing disabled", extra={"module_name": ModuleName.ADAPTER})
        return

    # Log initial environment variable status
    logger.debug(
        "Initial LangSmith env vars: LANGSMITH_TRACING=%s, LANGSMITH_API_KEY=%s, LANGSMITH_ENDPOINT=%s, LANGSMITH_PROJECT=%s, LANGSMITH_WORKSPACE_ID=%s",
        os.environ.get("LANGSMITH_TRACING", "not set"),
        _ellipsize_value(os.environ.get("LANGSMITH_API_KEY")),
        os.environ.get("LANGSMITH_ENDPOINT", "not set"),
        os.environ.get("LANGSMITH_PROJECT", "not set"),
        os.environ.get("LANGSMITH_WORKSPACE_ID", "not set"),
        extra={"module_name": ModuleName.ADAPTER},
    )

    # Required
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)

    # Optional
    endpoint = settings.langsmith_endpoint or "https://api.smith.langchain.com"
    os.environ.setdefault("LANGSMITH_ENDPOINT", endpoint)
    if settings.langsmith_project:
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    if settings.langsmith_workspace_id:
        os.environ.setdefault("LANGSMITH_WORKSPACE_ID", settings.langsmith_workspace_id)

    # Log final environment variable status
    logger.debug(
        "Final LangSmith env vars: LANGSMITH_TRACING=%s, LANGSMITH_API_KEY=%s, LANGSMITH_ENDPOINT=%s, LANGSMITH_PROJECT=%s, LANGSMITH_WORKSPACE_ID=%s",
        os.environ.get("LANGSMITH_TRACING", "not set"),
        _ellipsize_value(os.environ.get("LANGSMITH_API_KEY")),
        os.environ.get("LANGSMITH_ENDPOINT", "not set"),
        os.environ.get("LANGSMITH_PROJECT", "not set"),
        os.environ.get("LANGSMITH_WORKSPACE_ID", "not set"),
        extra={"module_name": ModuleName.ADAPTER},
    )

    logger.info(
        "LangSmith tracing enabled (endpoint=%s, project=%s, workspace=%s)",
        endpoint,
        settings.langsmith_project or "default",
        settings.langsmith_workspace_id or "-",
        extra={"module_name": ModuleName.ADAPTER},
    )
