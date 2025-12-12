from typing import Any, Iterable as IterableType, Mapping, Sequence, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config.settings import settings

StructuredT = TypeVar("StructuredT")


def _to_lc_messages(messages: Sequence[Mapping[str, Any]]) -> list[Any]:
    out: list[Any] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            out.append(SystemMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


def build_messages(messages: Sequence[Mapping[str, Any]]) -> list[Any]:
    """Translate lightweight dict messages into LangChain message objects."""
    return _to_lc_messages(messages)


def create_chat_model(model: str, temperature: float = 0.0) -> ChatOpenAI:
    """Instantiate a LangChain ChatOpenAI with shared settings."""
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        temperature=temperature,
        timeout=settings.openai_timeout,
    )


def chat(
    messages: list[Mapping[str, Any]],
    model: str,
    temperature: float = 0.0,
    *,
    response_format: type[StructuredT] | Mapping[str, Any] | None = None,
    include_raw: bool = False,
    structured_method: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Thin chat wrapper over LangChain ChatOpenAI with optional structured output.

    When `response_format` is provided, the chat model enforces the supplied schema
    (Pydantic model, TypedDict, or JSON schema mapping) via `with_structured_output`.
    Set `include_raw=True` to receive `{"parsed": ..., "raw": AIMessage}`.
    """
    lc_messages = build_messages(messages if isinstance(messages, IterableType) else [messages])
    chat_model = create_chat_model(model=model, temperature=temperature)
    if response_format is not None:
        with_kwargs: dict[str, Any] = {
            "include_raw": include_raw,
        }
        if structured_method is not None:
            with_kwargs["method"] = structured_method
        structured_model = chat_model.with_structured_output(response_format, **with_kwargs)
        return structured_model.invoke(lc_messages, **kwargs)
    return chat_model.invoke(lc_messages, **kwargs)
