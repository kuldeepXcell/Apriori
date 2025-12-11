"""LangSmith tracing configuration."""

from __future__ import annotations

import os

from app.config.settings import Settings
from app.core.logging import ModuleName, get_logger

logger = get_logger(__name__)
_configured = False


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

    logger.info(
        "LangSmith tracing enabled (endpoint=%s, project=%s, workspace=%s)",
        endpoint,
        settings.langsmith_project or "default",
        settings.langsmith_workspace_id or "-",
        extra={"module_name": ModuleName.ADAPTER},
    )
