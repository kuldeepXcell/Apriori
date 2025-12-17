"""Manual evaluation runner for the hybrid retrieval + rerank pipeline."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from app.adapters import vector_store
from app.config.settings import settings
from app.evals import metrics as eval_metrics
from app.steps.retrieval import baseline_hybrid_retrieval as retrieval


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Eval configuration (edit here when pointing to a new dataset/pipeline)
# ---------------------------------------------------------------------------
DATASET_NAME = "n2s_eval_set_30"
DATASET_PATH = Path("config/evals/n2s_eval_set_30.json")
OUTPUT_DIR = Path("data/processed/evals")
PIPELINE_NAME = "baseline_hybrid"
RETRIEVAL_TOP_K = 15
USE_LLM_RERANK = True
HYBRID_WEIGHTS = {
    "definition": 0.25,
    "question": 0.25,
    "context": 0.25,
    "keywords": 0.25,
}
# Temporarily evaluate only a subset of the dataset; set to None to run all.
MAX_QUESTIONS = 10
# NOTE: When we agree on how to combine metrics into a single score, reintroduce
# FINAL_SCORE_WEIGHTS and the associated logic. For now we skip the blended
# score so individual precision/recall/MRR stay visible without aggregation.
# FINAL_SCORE_WEIGHTS = {
#     "precision_weight": 0.4,
#     "recall_weight": 0.4,
#     "mrr_weight": 0.2,
# }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_indicator_name(value: str | None) -> str:
    """Best-effort normalization to compare indicators across sources."""
    if not value:
        return ""
    return (
        value.strip()
        .lower()
        .replace(" ", "_")
    )


def _dedup_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _extract_indicator(result: dict[str, Any]) -> str:
    payload = result.get("payload") or {}
    candidates = [
        payload.get("normalized_indicator_name"),
        payload.get("indicator_name"),
        payload.get("question"),
        result.get("id"),
    ]
    for candidate in candidates:
        normed = _normalize_indicator_name(str(candidate) if candidate else "")
        if normed:
            return normed
    return ""


def _load_eval_dataset(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Eval dataset must be a mapping of question -> indicators")
    normalized: dict[str, list[str]] = {}
    for question, indicators in data.items():
        norm_inds = [_normalize_indicator_name(ind) for ind in indicators if ind]
        normalized[question] = _dedup_preserve_order(norm_inds)
    return normalized


def _load_ingested_indicator_names(batch_size: int = 512) -> set[str]:
    """Pull normalized indicator names from the configured Qdrant collection."""

    names: set[str] = set()
    offset: Any = None
    collection = settings.qdrant_collection
    while True:
        points, offset = vector_store.client.scroll(
            collection_name=collection,
            limit=batch_size,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for point in points:
            payload = point.payload or {}
            raw = payload.get("normalized_indicator_name") or payload.get("indicator_name")
            norm = _normalize_indicator_name(raw)
            if norm:
                names.add(norm)
        if offset is None:
            break
    logger.info("Loaded %s ingested indicators from %s", len(names), collection)
    return names


def _build_vector_scores(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose per-vector contributions and fused score for each retrieved indicator."""

    formatted: list[dict[str, Any]] = []
    for idx, res in enumerate(results, start=1):
        payload = res.get("payload") or {}
        formatted.append(
            {
                "rank": idx,
                "indicator": _extract_indicator(res),
                "indicator_name": payload.get("indicator_name") or payload.get("normalized_indicator_name"),
                "fused_score": res.get("score", 0.0),
                "source_scores": res.get("source_scores", {}),
            }
        )
    return formatted


def _build_reranked_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for idx, res in enumerate(results, start=1):
        payload = res.get("payload") or {}
        formatted.append(
            {
                "rank": idx,
                "indicator": _extract_indicator(res),
                "indicator_name": payload.get("indicator_name") or payload.get("normalized_indicator_name"),
                "score": res.get("score", 0.0),
                "selected_by_llm": res.get("selected_by_llm", False),
                "llm_rank": res.get("llm_rank"),
            }
        )
    return formatted


def _question_report(
    idx: int,
    question: str,
    truth_indicators: list[str],
    ingested: set[str],
) -> dict[str, Any]:
    """Run retrieval/rerank for a single question and compute metrics."""

    logger.info("Running eval for question #%s: %s", idx, question)
    retrieval_start = perf_counter()
    weighted_results = retrieval.search(
        question,
        weights=HYBRID_WEIGHTS,
        top_k=RETRIEVAL_TOP_K,
        use_llm_rerank=False,
    )
    retrieval_latency_ms = (perf_counter() - retrieval_start) * 1000.0
    logger.info("Retrieval latency %.1f ms", retrieval_latency_ms)

    reranked_results = weighted_results
    rerank_latency_ms = 0.0
    if USE_LLM_RERANK:
        rerank_start = perf_counter()
        reranked_results = retrieval.search(
            question,
            weights=HYBRID_WEIGHTS,
            top_k=RETRIEVAL_TOP_K,
            use_llm_rerank=True,
        )
        rerank_latency_ms = (perf_counter() - rerank_start) * 1000.0
        logger.info("Rerank latency %.1f ms", rerank_latency_ms)

    truth_set = set(truth_indicators)
    weighted_preds = _dedup_preserve_order([_extract_indicator(res) for res in weighted_results])
    reranked_preds = _dedup_preserve_order([_extract_indicator(res) for res in reranked_results])

    precision = eval_metrics.precision_at_k(reranked_preds, truth_set, RETRIEVAL_TOP_K)
    recall = eval_metrics.recall_at_k(reranked_preds, truth_set, RETRIEVAL_TOP_K)
    mrr = eval_metrics.mean_reciprocal_rank(reranked_preds, truth_set)
    retrieved_hits = [pred for pred in weighted_preds if pred in truth_set]
    reranked_hits = [pred for pred in reranked_preds if pred in truth_set]
    missed = [truth for truth in truth_indicators if truth not in set(reranked_hits)]
    not_ingested = [truth for truth in truth_indicators if truth not in ingested]
    ingested_but_missed = [truth for truth in missed if truth in ingested]

    return {
        "query_id": idx,
        "query_text": question,
        "ground_truth_indicators": truth_indicators,
        "vector_scores": _build_vector_scores(weighted_results),
        "reranked_results": _build_reranked_results(reranked_results),
        "matches": {
            "retrieved_hits": retrieved_hits,
            "reranked_hits": reranked_hits,
            "missed_ground_truth": missed,
        },
        "database_presence": {
            "not_ingested": not_ingested,
            "ingested_but_missed": ingested_but_missed,
        },
        "metrics": {
            "precision_at_k": precision,
            "recall_at_k": recall,
            "mrr": mrr,
            "retrieval_latency_ms": retrieval_latency_ms,
            "rerank_latency_ms": rerank_latency_ms,
        },
    }


def run() -> Path:
    """Main entry point: evaluate all questions and write a timestamped JSON."""

    dataset = _load_eval_dataset(DATASET_PATH)
    ingested_names = _load_ingested_indicator_names()
    logger.info("Loaded dataset %s with %s questions", DATASET_PATH, len(dataset))
    question_results: list[dict[str, Any]] = []
    overall_start = perf_counter()

    for idx, (question, truth) in enumerate(dataset.items(), start=1):
        question_results.append(_question_report(idx, question, truth, ingested_names))
        if MAX_QUESTIONS and idx >= MAX_QUESTIONS:
            break

    elapsed_ms = (perf_counter() - overall_start) * 1000.0

    precision_avg = eval_metrics.average(q["metrics"]["precision_at_k"] for q in question_results)
    recall_avg = eval_metrics.average(q["metrics"]["recall_at_k"] for q in question_results)
    mrr_avg = eval_metrics.average(q["metrics"]["mrr"] for q in question_results)
    avg_retrieval_latency = eval_metrics.average(q["metrics"]["retrieval_latency_ms"] for q in question_results)
    avg_rerank_latency = eval_metrics.average(q["metrics"]["rerank_latency_ms"] for q in question_results)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{DATASET_NAME}-{timestamp}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{run_id}.json"

    report = {
        "metadata": {
            "run_id": run_id,
            "dataset_name": DATASET_NAME,
            "dataset_path": str(DATASET_PATH),
            "pipeline_name": PIPELINE_NAME,
            "retrieval_top_k": RETRIEVAL_TOP_K,
            "use_llm_rerank": USE_LLM_RERANK,
            "vector_weights": HYBRID_WEIGHTS,
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.llm_rerank_model if USE_LLM_RERANK else None,
            "qdrant_collection": settings.qdrant_collection,
            "started_at": timestamp,
            "duration_ms": elapsed_ms,
        },
        "aggregate_metrics": {
            "precision_at_k": precision_avg,
            "recall_at_k": recall_avg,
            "mrr": mrr_avg,
            "avg_retrieval_latency_ms": avg_retrieval_latency,
            "avg_rerank_latency_ms": avg_rerank_latency,
        },
        "questions": question_results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Eval run completed: %s", output_path)
    return output_path


if __name__ == "__main__":
    run()
