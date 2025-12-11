"""Weighted multi-vector retrieval (3 dense + 1 sparse) with client-side fusion."""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_qdrant import RetrievalMode

from app.adapters import vector_store
from app.config.settings import settings
from app.core.logging import ModuleName, get_logger
from app.steps.rerank import baseline_hybrid as llm_rerank

# ---- Collection & vector names (must match ingestion) ------------------------
COLLECTION = settings.qdrant_collection
DEF_VECTOR_NAME = settings.definition_vector_name
QUESTION_VECTOR_NAME = settings.question_vector_name
CONTEXT_VECTOR_NAME = settings.context_vector_name
SPARSE_VECTOR_NAME = settings.sparse_vector_name

logger = get_logger(__name__)


# ---- Helpers -----------------------------------------------------------------
def _validate_weights(weights: Dict[str, float] | None) -> Dict[str, float] | None:
    """
    Validate weights: must have 0 < sum(weights) <= 1.
    
    Returns validated weights (with negative values clamped to 0) if valid,
    otherwise returns None to reject the request.
    """
    if not weights:
        return None
    total = sum(v for v in weights.values() if v > 0)
    if total <= 0 or total > 1.0:
        return None
    return {k: max(v, 0.0) for k, v in weights.items()}


def _doc_id(doc: Any) -> str:
    """Extract a stable id from a LangChain Document returned by Qdrant."""
    meta = getattr(doc, "metadata", {}) or {}
    return str(
        meta.get("id")
        or meta.get("point_id")
        or meta.get("uuid")
        or meta.get("normalized_indicator_name")
        or getattr(doc, "id", "")
    )


# ---- Public API --------------------------------------------------------------
def search(
    question: str,
    weights: Dict[str, float] | None = None,
    top_k: int = 15,
) -> List[Dict[str, Any]]:
    """
    Search across 3 dense + 1 sparse vectors with user-provided weights.

    - Validates weights (must have 0 < sum <= 1).
    - Embeds the question once for dense and sparse vectors.
    - Runs per-vector searches.
    - Applies weighted sum of raw scores to produce the final ranking.
    """
    if not question:
        return []

    w = _validate_weights(weights)
    if w is None:
        logger.error(
            "Invalid weights (sum must be 0 < sum <= 1): %s",
            weights,
            extra={"module_name": ModuleName.RETRIEVAL},
        )
        return []
    logger.info(
        "Retrieval request top_k=%s weights=%s",
        top_k,
        w,
        extra={"module_name": ModuleName.RETRIEVAL},
    )
    logger.debug(
        "Validated weights sum=%.3f raw=%s",
        sum(w.values()),
        weights,
        extra={"module_name": ModuleName.RETRIEVAL},
    )

    # Use top_k directly as the search limit
    search_limit = top_k

    def_hits = vector_store.hybrid_search(
        question, k=search_limit, vector_name=DEF_VECTOR_NAME, retrieval_mode=RetrievalMode.HYBRID
    )
    q_hits = vector_store.hybrid_search(
        question, k=search_limit, vector_name=QUESTION_VECTOR_NAME, retrieval_mode=RetrievalMode.HYBRID
    )
    ctx_hits = vector_store.hybrid_search(
        question, k=search_limit, vector_name=CONTEXT_VECTOR_NAME, retrieval_mode=RetrievalMode.HYBRID
    )
    sparse_hits = vector_store.hybrid_search(question, k=search_limit, retrieval_mode=RetrievalMode.SPARSE)

    # Collect per-vector scores (raw, no normalization)
    def_scores = {_doc_id(doc): score for doc, score in def_hits}
    q_scores = {_doc_id(doc): score for doc, score in q_hits}
    ctx_scores = {_doc_id(doc): score for doc, score in ctx_hits}
    sparse_scores = {_doc_id(doc): score for doc, score in sparse_hits}

    # Debug: Log top scores from each vector search
    logger.debug(
        "Dense vector '%s' top scores: %s",
        DEF_VECTOR_NAME,
        dict(sorted(def_scores.items(), key=lambda x: x[1], reverse=True)[:5]),
        extra={"module_name": ModuleName.RETRIEVAL},
    )
    logger.debug(
        "Dense vector '%s' top scores: %s",
        QUESTION_VECTOR_NAME,
        dict(sorted(q_scores.items(), key=lambda x: x[1], reverse=True)[:5]),
        extra={"module_name": ModuleName.RETRIEVAL},
    )
    logger.debug(
        "Dense vector '%s' top scores: %s",
        CONTEXT_VECTOR_NAME,
        dict(sorted(ctx_scores.items(), key=lambda x: x[1], reverse=True)[:5]),
        extra={"module_name": ModuleName.RETRIEVAL},
    )
    logger.debug(
        "Sparse vector '%s' top scores: %s",
        SPARSE_VECTOR_NAME,
        dict(sorted(sparse_scores.items(), key=lambda x: x[1], reverse=True)[:5]),
        extra={"module_name": ModuleName.RETRIEVAL},
    )

    # Merge candidates
    all_ids = set(def_scores) | set(q_scores) | set(ctx_scores) | set(sparse_scores)
    results: List[Dict[str, Any]] = []
    logger.debug(
        "Candidate ids count=%s (def=%s q=%s ctx=%s kw=%s)",
        len(all_ids),
        len(def_scores),
        len(q_scores),
        len(ctx_scores),
        len(sparse_scores),
        extra={"module_name": ModuleName.RETRIEVAL},
    )
    for pid in all_ids:
        def_score = def_scores.get(pid, 0)
        q_score = q_scores.get(pid, 0)
        ctx_score = ctx_scores.get(pid, 0)
        kw_score = sparse_scores.get(pid, 0)
        
        def_weight = w.get("definition", 0)
        q_weight = w.get("question", 0)
        ctx_weight = w.get("context", 0)
        kw_weight = w.get("keywords", 0)
        
        weighted_def = def_weight * def_score
        weighted_q = q_weight * q_score
        weighted_ctx = ctx_weight * ctx_score
        weighted_kw = kw_weight * kw_score
        
        score = weighted_def + weighted_q + weighted_ctx + weighted_kw
        
        logger.debug(
            "Point %s: def=%.4f(×%.2f=%.4f) q=%.4f(×%.2f=%.4f) ctx=%.4f(×%.2f=%.4f) kw=%.4f(×%.2f=%.4f) → final=%.4f",
            pid,
            def_score, def_weight, weighted_def,
            q_score, q_weight, weighted_q,
            ctx_score, ctx_weight, weighted_ctx,
            kw_score, kw_weight, weighted_kw,
            score,
            extra={"module_name": ModuleName.RETRIEVAL},
        )
        # Prefer payload from the first source that has it
        payload = (
            next((p.metadata for p, _ in def_hits if _doc_id(p) == pid), None)
            or next((p.metadata for p, _ in q_hits if _doc_id(p) == pid), None)
            or next((p.metadata for p, _ in ctx_hits if _doc_id(p) == pid), None)
            or next((p.metadata for p, _ in sparse_hits if _doc_id(p) == pid), None)
        )
        results.append(
            {
                "id": pid,
                "score": score,
                "payload": payload or {},
                "source_scores": {
                    "definition": def_scores.get(pid, 0),
                    "question": q_scores.get(pid, 0),
                    "context": ctx_scores.get(pid, 0),
                    "keywords": sparse_scores.get(pid, 0),
                },
            }
        )

    results.sort(key=lambda r: r["score"], reverse=True)
    top_results = results[:top_k]

    # Apply LLM rerank to annotate selections; keep all results.
    top_results = _apply_llm_rerank(question=question, results=top_results)

    # Debug: Log final top results with component breakdown
    logger.debug(
        "Final top %s results (post LLM rerank annotation):",
        len(top_results),
        extra={"module_name": ModuleName.RETRIEVAL},
    )
    for idx, res in enumerate(top_results, start=1):
        src_scores = res.get("source_scores", {})
        logger.debug(
            "  %d. Point %s: final_score=%.4f | def=%.4f q=%.4f ctx=%.4f kw=%.4f llm_selected=%s llm_rank=%s",
            idx,
            res.get("id", "unknown"),
            res.get("score", 0),
            src_scores.get("definition", 0),
            src_scores.get("question", 0),
            src_scores.get("context", 0),
            src_scores.get("keywords", 0),
            res.get("selected_by_llm", False),
            res.get("llm_rank"),
            extra={"module_name": ModuleName.RETRIEVAL},
        )

    return top_results


def _apply_llm_rerank(question: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate results with LLM selection while keeping all items."""
    if not results:
        return results

    try:
        rerank_out = llm_rerank.rerank(question=question, candidates=results)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception(
            "LLM rerank failed, returning original order",
            extra={"module_name": ModuleName.RERANK},
        )
        rerank_out = {"selected_ids": [], "llm_rank": {}, "raw": f"error: {exc}"}

    selected_ids = rerank_out.get("selected_ids") or []
    rank_map: Dict[str, int] = rerank_out.get("llm_rank") or {}

    for res in results:
        rid = str(res.get("id", ""))
        rank = rank_map.get(rid)
        res["selected_by_llm"] = rank is not None
        if rank is not None:
            res["llm_rank"] = rank

    if not selected_ids:
        return results

    indexed = list(enumerate(results))

    def sort_key(pair: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        idx, res = pair
        rid = str(res.get("id", ""))
        rank = rank_map.get(rid)
        if rank is not None:
            return (0, rank)
        return (1, idx)

    reordered = [item for _, item in sorted(indexed, key=sort_key)]
    return reordered
