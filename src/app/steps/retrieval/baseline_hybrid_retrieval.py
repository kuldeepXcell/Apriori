"""Weighted multi-vector retrieval (3 dense + 1 sparse) with client-side fusion."""

from __future__ import annotations

from typing import Any, Dict, List
from time import perf_counter

from langchain_qdrant import RetrievalMode

from app.adapters import vector_store
from app.config.settings import settings
from app.core.logging import ModuleName, get_logger
from app.steps.rerank import baseline_hybrid as llm_rerank
from app.core.langsmith import configure_langsmith

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
    id_val = (
        meta.get("_id")
        or meta.get("id")
        or meta.get("point_id")
        or meta.get("uuid")
        or meta.get("normalized_indicator_name")
        or meta.get("indicator_name")
        or getattr(doc, "id", "")
    )
    return str(id_val)


def _norm_name(meta: dict[str, Any] | None) -> str:
    """Best-effort normalized name from metadata/payload."""
    meta = meta or {}
    return (
        meta.get("normalized_indicator_name")
        or meta.get("indicator_name")
        or meta.get("_id")
        or meta.get("id")
        or ""
    )


# ---- Public API --------------------------------------------------------------
def search(
    question: str,
    weights: Dict[str, float] | None = None,
    top_k: int = 15,
    use_llm_rerank: bool = True,
) -> Dict[str, Any]:
    """
    Search across 3 dense + 1 sparse vectors with user-provided weights.

    - Validates weights (must have 0 < sum <= 1).
    - Embeds the question once for dense and sparse vectors.
    - Runs per-vector searches.
    - Applies weighted sum of raw scores to produce the final ranking.
    """
    if not question:
        return []

    configure_langsmith(settings)
    w = _validate_weights(weights)
    if w is None:
        logger.error(
            "Invalid weights (sum must be 0 < sum <= 1): %s",
            weights,
            extra={"module_name": ModuleName.RETRIEVAL},
        )
        return []
    logger.info(
        "Retrieval request top_k=%s use_llm_rerank=%s weights=%s",
        top_k,
        use_llm_rerank,
        w,
        extra={"module_name": ModuleName.RETRIEVAL},
    )
    logger.debug(
        "Validated weights sum=%.3f raw=%s",
        sum(w.values()),
        weights,
        extra={"module_name": ModuleName.RETRIEVAL},
    )

    t_start = perf_counter()

    # Use top_k directly as the search limit
    search_limit = top_k

    t_def = perf_counter()
    def_hits = vector_store.hybrid_search(
        question, k=search_limit, vector_name=DEF_VECTOR_NAME, retrieval_mode=RetrievalMode.HYBRID
    )
    t_def = perf_counter() - t_def

    t_q = perf_counter()
    q_hits = vector_store.hybrid_search(
        question, k=search_limit, vector_name=QUESTION_VECTOR_NAME, retrieval_mode=RetrievalMode.HYBRID
    )
    t_q = perf_counter() - t_q

    t_ctx = perf_counter()
    ctx_hits = vector_store.hybrid_search(
        question, k=search_limit, vector_name=CONTEXT_VECTOR_NAME, retrieval_mode=RetrievalMode.HYBRID
    )
    t_ctx = perf_counter() - t_ctx

    t_sparse = perf_counter()
    sparse_hits = vector_store.hybrid_search(question, k=search_limit, retrieval_mode=RetrievalMode.SPARSE)
    t_sparse = perf_counter() - t_sparse

    # Collect per-vector scores (raw, no normalization)
    t_collect = perf_counter()
    def_scores = {_doc_id(doc): score for doc, score in def_hits}
    q_scores = {_doc_id(doc): score for doc, score in q_hits}
    ctx_scores = {_doc_id(doc): score for doc, score in ctx_hits}
    sparse_scores = {_doc_id(doc): score for doc, score in sparse_hits}
    t_collect = perf_counter() - t_collect

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
    t_merge = perf_counter()
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
        
        # Prefer payload from the first source that has it
        payload = (
            next((p.metadata for p, _ in def_hits if _doc_id(p) == pid), None)
            or next((p.metadata for p, _ in q_hits if _doc_id(p) == pid), None)
            or next((p.metadata for p, _ in ctx_hits if _doc_id(p) == pid), None)
            or next((p.metadata for p, _ in sparse_hits if _doc_id(p) == pid), None)
        ) or {}
        if getattr(payload, "copy", None):
            payload = payload.copy()
        if isinstance(payload, dict):
            payload.setdefault("definition", next((p.page_content for p, _ in def_hits if _doc_id(p) == pid), ""))
            # question/application_context live only in metadata; leave as-is
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
    t_merge = perf_counter() - t_merge

    # Apply LLM rerank to annotate selections; keep all results.
    t_rerank = 0.0
    if use_llm_rerank:
        rerank_start = perf_counter()
        top_results = _apply_llm_rerank(question=question, results=top_results)
        t_rerank = perf_counter() - rerank_start
    else:
        for res in top_results:
            res["selected_by_llm"] = False
            res.pop("llm_rank", None)

    # Log the top results' normalized names and those selected by LLM.
    all_norms = [
        _norm_name(res.get("payload")) for res in top_results
    ]
    selected_norms = [
        _norm_name(res.get("payload")) for res in top_results if res.get("selected_by_llm")
    ]
    logger.info(
        "Top %s normalized names: %s",
        len(all_norms),
        all_norms,
        extra={"module_name": ModuleName.RETRIEVAL},
    )
    logger.info(
        "LLM-selected normalized names: %s",
        selected_norms,
        extra={"module_name": ModuleName.RETRIEVAL},
    )

    point_summaries: list[dict[str, Any]] = []
    for idx, res in enumerate(top_results, start=1):
        src_scores = res.get("source_scores", {})
        point_summaries.append(
            {
                "rank": idx,
                "id": res.get("id", "unknown"),
                "final_score": round(res.get("score", 0.0), 4),
                "def": round(src_scores.get("definition", 0.0), 4),
                "q": round(src_scores.get("question", 0.0), 4),
                "ctx": round(src_scores.get("context", 0.0), 4),
                "kw": round(src_scores.get("keywords", 0.0), 4),
                "llm_selected": res.get("selected_by_llm", False),
                "llm_rank": res.get("llm_rank"),
            }
        )
    # Log compact, formatted summary for observability without verbosity
    formatted = [
        f"#{p['rank']}: id={p['id']} final={p['final_score']:.4f} "
        f"(def={p['def']:.4f} q={p['q']:.4f} ctx={p['ctx']:.4f} kw={p['kw']:.4f}) "
        f"llm_selected={p['llm_selected']} llm_rank={p['llm_rank']}"
        for p in point_summaries
    ]
    logger.debug(
        "Top %s results:\n%s",
        len(point_summaries),
        "\n".join(formatted),
        extra={"module_name": ModuleName.RETRIEVAL},
    )

    total_time = perf_counter() - t_start
    logger.info(
        "Timings (s): def=%.3f q=%.3f ctx=%.3f sparse=%.3f collect=%.3f merge=%.3f rerank=%.3f total=%.3f",
        t_def,
        t_q,
        t_ctx,
        t_sparse,
        t_collect,
        t_merge,
        t_rerank,
        total_time,
        extra={"module_name": ModuleName.RETRIEVAL},
    )

    return {"results": top_results}


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
    # Log retrieved vs LLM-selected for observability
    try:
        retrieved_payloads = []
        for res in reordered:
            payload = res.get("payload") or {}
            retrieved_payloads.append(
                {
                    "id": res.get("id"),
                    "name": payload.get("indicator_name") or payload.get("question") or "unknown",
                    "score": res.get("score"),
                    "selected_by_llm": res.get("selected_by_llm", False),
                    "llm_rank": res.get("llm_rank"),
                }
            )
        logger.info(
            "Retrieved %d results; LLM selected %d",
            len(retrieved_payloads),
            len(selected_ids),
            extra={
                "module_name": ModuleName.RETRIEVAL,
                "retrieved": retrieved_payloads,
                "llm_selected_ids": selected_ids,
            },
        )
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to log retrieval summary", extra={"module_name": ModuleName.RETRIEVAL})
    return reordered
