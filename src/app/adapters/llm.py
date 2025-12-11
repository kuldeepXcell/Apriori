from collections.abc import Iterable
from typing import Any, Mapping

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config.settings import settings


def _to_lc_messages(messages: list[Mapping[str, Any]] | Iterable[Mapping[str, Any]]) -> list[Any]:
    out: list[Any] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            out.append(SystemMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


def chat(
    messages: list[Mapping[str, Any]],
    model: str,
    temperature: float = 0.0,
    **kwargs: Any,
):
    """Thin chat wrapper over LangChain ChatOpenAI."""
    llm = ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        temperature=temperature,
        timeout=settings.openai_timeout,
    )
    lc_messages = _to_lc_messages(messages if isinstance(messages, Iterable) else [messages])
    return llm.invoke(lc_messages, **kwargs)

