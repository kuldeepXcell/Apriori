from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

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


def _parse_response(content: str, valid_ids: set[str]) -> list[str]:
    try:
        parsed = json.loads(content)
    except Exception:
        return []
    if not isinstance(parsed, dict):
        return []
    selected = parsed.get("selected_ids") or parsed.get("selected") or []
    if not isinstance(selected, list):
        return []
    filtered: list[str] = []
    for raw in selected:
        sid = str(raw)
        if sid in valid_ids:
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
    messages = _build_messages(question, candidates)
    valid_ids = {str(item.get("id", "")) for item in candidates}

    try:
        resp = llm.chat(
            messages=messages,
            model=chosen_model,
            temperature=chosen_temp,
            response_format={"type": "json_object"},
        )
        content = resp.content if resp else ""
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

    if not content:
        return {"selected_ids": [], "llm_rank": {}, "raw": content}

    selected_ids = _parse_response(content, valid_ids)
    llm_rank = {sid: idx + 1 for idx, sid in enumerate(selected_ids)}

    logger.debug(
        "LLM rerank selected %s ids: %s",
        len(selected_ids),
        selected_ids,
        extra={"module_name": ModuleName.RERANK},
    )
    return {"selected_ids": selected_ids, "llm_rank": llm_rank, "raw": content}

