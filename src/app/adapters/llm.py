from collections.abc import Iterable
from typing import Any, Mapping

from openai import OpenAI

from app.config.settings import settings

# Initialize a single shared client; uses env-driven key/timeout.
client = OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout)


def chat(
    messages: list[Mapping[str, Any]],
    model: str,
    temperature: float = 0.0,
    **kwargs: Any,
):
    """Thin chat wrapper so pipelines stay client-agnostic."""
    return client.chat.completions.create(
        model=model,
        messages=list(messages) if isinstance(messages, Iterable) else messages,
        temperature=temperature,
        **kwargs,
    )

