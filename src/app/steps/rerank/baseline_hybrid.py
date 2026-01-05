from __future__ import annotations

from typing import Any, Dict, Iterable

from pydantic import BaseModel, Field

from app.adapters import llm
from app.config.settings import settings
from app.core.logging import ModuleName, get_logger

logger = get_logger(__name__)

# Defaults are env-driven to allow tuning without code edits.
DEFAULT_MODEL = settings.llm_rerank_model
DEFAULT_TEMPERATURE = settings.llm_rerank_temperature


def _dedup_preserve_order(seq: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item in seen or not item:
            continue
        seen.add(item)
        out.append(item)
    return out


def _build_messages(question: str, items: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Construct a compact prompt with 3-field context for each indicator."""
    lines: list[str] = []
    for idx, item in enumerate(items, start=1):
        indicator_id = str(item.get("id", ""))
        payload = item.get("payload") or {}
        lines.append(
            "\n".join(
                [
                    f"{idx}. id: {indicator_id}",
                    f"   name: {payload.get('indicator_name', '')}",
                    f"   application: {payload.get('application_context', '')}",
                    f"   question: {payload.get('question', '')}",
                    f"   definition: {payload.get('definition', '')}",
                ]
            )
        )
    user_block = "\n".join(
        [
            "User question:",
            question,
            "",
            "Indicators (application, question, definition):",
            "\n\n".join(lines),
        ]
    )
    system_block = (
        "You are a reranker that picks the indicators most relevant to the user question. "
        "Return JSON with the ids you choose in order of relevance. "
        "If none fit, return an empty list. Do not make up ids."
    )
    return [
        {"role": "system", "content": system_block},
        {"role": "user", "content": user_block},
    ]


class RerankResponse(BaseModel):
    """Structured representation of ids selected by the reranker."""

    selected_ids: list[str] = Field(default_factory=list, description="Ordered ids relevant to the prompt.")


def _sanitize_selected_ids(selected: Iterable[str], valid_ids: set[str]) -> list[str]:
    """Filter and deduplicate reranker selections while preserving order."""

    filtered: list[str] = []
    for raw in selected:
        sid = str(raw) if raw is not None else ""
        if sid and sid in valid_ids:
            filtered.append(sid)
    return _dedup_preserve_order(filtered)


def rerank(
    question: str,
    candidates: list[Dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float | None = None,
) -> Dict[str, Any]:
    """
    Call the LLM to pick any number of indicators; returns ids in chosen order.

    Returns:
        {
            "selected_ids": [...],
            "llm_rank": {"id": rank_int},
            "raw": "<raw response or error>",
        }
    """
    if not question or not candidates:
        return {"selected_ids": [], "llm_rank": {}, "raw": None}

    chosen_model = model or DEFAULT_MODEL
    chosen_temp = DEFAULT_TEMPERATURE if temperature is None else temperature
    pipeline_messages = _build_messages(question, candidates)
    valid_ids = {str(item.get("id", "")) for item in candidates}

    try:
        llm_result = llm.chat(
            messages=pipeline_messages,
            model=chosen_model,
            temperature=chosen_temp,
            response_format=RerankResponse,
            include_raw=True,
            config={
                "tags": ["feedback-rerank"],
                "metadata": {"component": "feedback-rerank"},
            },
        )
        parsed = (
            llm_result.get("parsed")
            if isinstance(llm_result, dict)
            else llm_result
        )
        raw_message = (
            llm_result.get("raw") if isinstance(llm_result, dict) else None
        )
        raw_content = ""
        if raw_message is not None:
            raw_content = (
                raw_message.content
                if hasattr(raw_message, "content")
                else str(raw_message)
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception(
            "LLM rerank call failed",
            extra={"module_name": ModuleName.RERANK},
        )
        return {
            "selected_ids": [],
            "llm_rank": {},
            "raw": f"error: {exc}",
        }

    if not parsed:
        return {"selected_ids": [], "llm_rank": {}, "raw": raw_content}

    selected_ids = _sanitize_selected_ids(parsed.selected_ids, valid_ids)
    if not selected_ids:
        selected_ids = []

    llm_rank = {sid: idx + 1 for idx, sid in enumerate(selected_ids)}

    logger.debug(
        "LLM rerank selected %s ids: %s",
        len(selected_ids),
        selected_ids,
        extra={"module_name": ModuleName.RERANK},
    )
    return {"selected_ids": selected_ids, "llm_rank": llm_rank, "raw": raw_content}
