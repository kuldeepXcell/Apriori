"""
Thin helper around Supabase Storage's REST API for uploading chart images.
"""

from __future__ import annotations

import time
from typing import Final

import requests

from app.config.settings import settings
from app.core.logging import ModuleName, get_logger

logger = get_logger(__name__)


def _require_supabase_settings() -> tuple[str, str, str]:
    """Validate that Supabase credentials are configured."""
    if not settings.supabase_url:
        raise RuntimeError("SUPABASE_URL is not configured")
    if not settings.supabase_secret_key:
        raise RuntimeError("SUPABASE_SECRET_KEY is not configured")
    bucket = settings.supabase_feedback_bucket
    if not bucket:
        raise RuntimeError("Supabase feedback bucket is blank")
    return settings.supabase_url.rstrip("/"), settings.supabase_secret_key, bucket


def upload_chart_image(
    feedback_id: str,
    payload: bytes,
    *,
    content_type: str = "image/png",
    extension: str = "png",
) -> str:
    """
    Upload a compressed chart image to Supabase Storage and return the public URL.

    Args:
        feedback_id: Row identifier so charts land in distinct folders.
        payload: Binary image data (already compressed).
        content_type: MIME type, defaults to PNG.

    Returns:
        str: Public URL pointing at the uploaded image.
    """
    base_url, service_key, bucket = _require_supabase_settings()
    path = f"{feedback_id}/{int(time.time())}.{extension.lstrip('.')}"
    upload_url = f"{base_url}/storage/v1/object/{bucket}/{path}?upsert=true"
    headers: Final[dict[str, str]] = {
        "Authorization": f"Bearer {service_key}",
        "apikey": service_key,
        "Content-Type": content_type,
    }
    response = requests.post(upload_url, headers=headers, data=payload, timeout=30)
    if response.status_code >= 400:
        logger.error(
            "Supabase upload failed: %s %s",
            response.status_code,
            response.text,
            extra={"module_name": ModuleName.ADAPTER},
        )
        response.raise_for_status()

    public_url = f"{base_url}/storage/v1/object/public/{bucket}/{path}"
    logger.info(
        "Uploaded chart image to Supabase bucket %s (path=%s)",
        bucket,
        path,
        extra={"module_name": ModuleName.ADAPTER},
    )
    return public_url
