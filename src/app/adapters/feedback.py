from __future__ import annotations

import json
from typing import Any, Dict, List

from app.adapters.postgres import get_connection
from app.core.logging import ModuleName, get_logger

logger = get_logger(__name__)


def _normalize_indicators(payload: Dict[str, Any]) -> List[dict]:
    """Coalesce retrieved indicators and per-indicator labels into one array."""
    if payload.get("indicators"):
        return payload["indicators"]

    retrieved = payload.get("retrieved_indicators") or []
    feedback_entries = payload.get("user_feedback") or []
    feedback_index: dict[str, str | None] = {}
    for entry in feedback_entries:
        key = entry.get("id") or entry.get("normalized_indicator_name") or entry.get("indicator_name")
        if key:
            feedback_index[key] = entry.get("label")

    merged: list[dict] = []
    for indicator in retrieved:
        key = indicator.get("id") or indicator.get("normalized_indicator_name") or indicator.get("indicator_name")
        label = feedback_index.get(key)
        merged.append({**indicator, "label": label})
    return merged


def save_indicator_feedback(payload: Dict[str, Any]) -> str:
    """
    Persist feedback for a single query into the indicator_feedback table.
    """
    query_text = payload.get("query_text") or ""
    reranker_used = bool(payload.get("reranker_used"))
    indicators = _normalize_indicators(payload)
    sql = """
        INSERT INTO indicator_feedback
            (query_text, reranker_used, indicators)
        VALUES
            (%s, %s, %s::jsonb)
        RETURNING feedback_id
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    query_text,
                    reranker_used,
                    json.dumps(indicators),
                ),
            )
            row = cur.fetchone()
            feedback_id = row[0] if row else None

    feedback_id_str = str(feedback_id) if feedback_id else "unknown"
    logger.info(
        "Saved indicator feedback %s (reranker_used=%s, indicators=%s)",
        feedback_id_str,
        reranker_used,
        len(indicators),
        extra={"module_name": ModuleName.ADAPTER},
    )
    return feedback_id_str


def update_chart_feedback(
    feedback_id: str,
    *,
    chart_notes: str | None,
    chart_svg: str | None,
) -> None:
    """Attach chart feedback data to an existing indicator_feedback row."""
    sql = """
        UPDATE indicator_feedback
        SET chart_notes = %s,
            chart_svg = %s,
            chart_saved_at = NOW()
        WHERE feedback_id = %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (chart_notes, chart_svg, feedback_id))

    logger.info(
        "Updated chart feedback for %s (notes=%s, svg=%s)",
        feedback_id,
        bool(chart_notes),
        bool(chart_svg),
        extra={"module_name": ModuleName.ADAPTER},
    )


def get_feedback(feedback_id: str) -> dict[str, Any] | None:
    """Fetch a single feedback row by id."""
    sql = """
        SELECT feedback_id, created_at, query_text, reranker_used, indicators,
               chart_notes, chart_svg, chart_saved_at
        FROM indicator_feedback
        WHERE feedback_id = %s
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (feedback_id,))
        row = cur.fetchone()
        if not row:
            return None
    (
        fid,
        created_at,
        query_text,
        reranker_used,
        indicators,
        chart_notes,
        chart_svg,
        chart_saved_at,
    ) = row
    return {
        "feedback_id": fid,
        "created_at": created_at,
        "query_text": query_text,
        "reranker_used": reranker_used,
        "indicators": indicators,
        "chart_notes": chart_notes,
        "chart_svg": chart_svg,
        "chart_saved_at": chart_saved_at,
    }


def list_feedback(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    """List recent feedback rows."""
    sql = """
        SELECT feedback_id, created_at, query_text, reranker_used, indicators,
               chart_notes, chart_svg, chart_saved_at
        FROM indicator_feedback
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (limit, offset))
        rows = cur.fetchall()
    results: list[dict[str, Any]] = []
    for (
        fid,
        created_at,
        query_text,
        reranker_used,
        indicators,
        chart_notes,
        chart_svg,
        chart_saved_at,
    ) in rows:
        results.append(
            {
                "feedback_id": fid,
                "created_at": created_at,
                "query_text": query_text,
                "reranker_used": reranker_used,
                "indicators": indicators,
                "chart_notes": chart_notes,
                "chart_svg": chart_svg,
                "chart_saved_at": chart_saved_at,
            }
        )
    return results
