from __future__ import annotations

import json
from typing import Any, Dict

from app.adapters.postgres import get_connection
from app.core.logging import ModuleName, get_logger

logger = get_logger(__name__)


def save_indicator_feedback(payload: Dict[str, Any]) -> str:
    """
    Persist feedback for a single query into the indicator_feedback table.
    """
    query_text = payload.get("query_text") or ""
    reranker_used = bool(payload.get("reranker_used"))
    retrieved_indicators = payload.get("retrieved_indicators") or []
    user_feedback = payload.get("user_feedback") or []

    sql = """
        INSERT INTO indicator_feedback
            (query_text, reranker_used, retrieved_indicators, user_feedback)
        VALUES
            (%s, %s, %s::jsonb, %s::jsonb)
        RETURNING feedback_id
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    query_text,
                    reranker_used,
                    json.dumps(retrieved_indicators),
                    json.dumps(user_feedback),
                ),
            )
            row = cur.fetchone()
            feedback_id = row[0] if row else None

    feedback_id_str = str(feedback_id) if feedback_id else "unknown"
    logger.info(
        "Saved indicator feedback %s (reranker_used=%s, indicators=%s)",
        feedback_id_str,
        reranker_used,
        len(retrieved_indicators),
        extra={"module_name": ModuleName.ADAPTER},
    )
    return feedback_id_str
