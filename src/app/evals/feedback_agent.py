"""LLM-based feedback agent that analyzes eval results and suggests fixes."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

from app.adapters import llm
from openai import APITimeoutError
from app.config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Manual configuration section
# ---------------------------------------------------------------------------
EVAL_REPORT_PATH = Path("data/processed/evals/n2s_eval_set_30-20251216T081609Z.json")
INDICATOR_CATALOG_PATH = Path("data/combined_indicators.json")
OUTPUT_DIR = Path("data/processed/evals/feedback")
MAX_QUESTIONS = 10  # set to None to analyze full report
LLM_MODEL = settings.llm_rerank_model
LLM_TEMPERATURE = 0.1
PRECISION_THRESHOLD = 0.25
RECALL_THRESHOLD = 0.35
SUCCESS_SAMPLE_LIMIT = 3


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_eval_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Eval report not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_indicator_catalog(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Indicator catalog not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    catalog: dict[str, Any] = {}
    for sheet in data.get("sheets", []):
        for indicator in sheet.get("indicators", []):
            norm = indicator.get("normalized_indicator_name")
            if not norm:
                continue
            catalog[norm] = indicator
    logger.info("Loaded %s indicators from catalog", len(catalog))
    return catalog


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _normalize(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().lower().replace(" ", "_")


def _find_vector_scores(vector_scores: list[dict[str, Any]], indicator: str) -> dict[str, Any] | None:
    target = _normalize(indicator)
    for entry in vector_scores:
        if _normalize(entry.get("indicator")) == target:
            return entry
    return None


def _sample_successes(question_block: dict[str, Any], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    hits = question_block.get("matches", {}).get("reranked_hits", [])
    samples: list[dict[str, Any]] = []
    for indicator in hits:
        norm = _normalize(indicator)
        meta = catalog.get(norm)
        if not meta:
            continue
        samples.append(
            {
                "indicator": norm,
                "definition": meta.get("definition", ""),
                "application_context": meta.get("application_context", ""),
                "question": meta.get("question", ""),
            }
        )
        if len(samples) >= SUCCESS_SAMPLE_LIMIT:
            break
    return samples


def _build_missed_entries(question_block: dict[str, Any], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    missed = question_block.get("database_presence", {}).get("ingested_but_missed", [])
    vector_scores = question_block.get("vector_scores", [])
    reranked_hits = {_normalize(n) for n in question_block.get("matches", {}).get("reranked_hits", [])}
    entries: list[dict[str, Any]] = []
    for indicator in missed:
        norm = _normalize(indicator)
        meta = catalog.get(norm, {})
        scores = _find_vector_scores(vector_scores, norm)
        if scores:
            status = "retrieved_low_score"
            if norm not in reranked_hits:
                status = "reranker_dropped"
        else:
            status = "never_retrieved"
        entries.append(
            {
                "indicator": norm,
                "indicator_name": meta.get("indicator_name", norm),
                "definition": meta.get("definition", ""),
                "application_context": meta.get("application_context", ""),
                "question": meta.get("question", ""),
                "fused_score": scores.get("fused_score") if scores else None,
                "source_scores": scores.get("source_scores") if scores else None,
                "status": status,
            }
        )
    return entries


def _build_system_flags(metrics: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if metrics.get("precision_at_k", 1.0) < PRECISION_THRESHOLD:
        flags.append("low_precision")
    if metrics.get("recall_at_k", 1.0) < RECALL_THRESHOLD:
        flags.append("low_recall")
    return flags


def _encode_toon_lines(value: Any, indent: int = 0) -> list[str]:
    """Return TOON lines for nested dict/list primitives."""

    lines: list[str] = []
    prefix = " " * indent
    if isinstance(value, dict):
        for key, val in value.items():
            if isinstance(val, dict):
                lines.append(f"{prefix}{key}:")
                lines.extend(_encode_toon_lines(val, indent + 2))
            elif isinstance(val, list):
                lines.append(f"{prefix}{key}[{len(val)}]:")
                lines.extend(_encode_toon_lines(val, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {val}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{prefix}-")
                lines.extend(_encode_toon_lines(item, indent + 2))
            elif isinstance(item, list):
                lines.append(f"{prefix}- [{len(item)}]:")
                lines.extend(_encode_toon_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {item}")
    else:
        lines.append(f"{prefix}{value}")
    return lines


def _build_prompt(question_block: dict[str, Any], catalog: dict[str, Any]) -> str:
    missed_entries = _build_missed_entries(question_block, catalog)
    if not missed_entries:
        return ""
    success_examples = _sample_successes(question_block, catalog)
    metrics = question_block.get("metrics", {})
    question_payload = {
        "question": {
            "text": question_block.get("query_text", ""),
            "metrics": metrics,
            "system_flags": _build_system_flags(metrics),
        },
        "successful_examples": success_examples,
        "missed_indicators": missed_entries,
    }
    return (
        "You audit a retrieval + LLM rerank pipeline. "
        "Explain why these indicators were missed and suggest concrete remediation steps (data ingestion, keyword/weight tuning, reranker prompt edits)."
        "\nRespond in JSON with keys summary and action_items."
        "\n\nContext (TOON):\n"
        + "\n".join(_encode_toon_lines(question_payload))
    )


def _call_llm(prompt: str, retries: int = 3, backoff: float = 2.0) -> dict[str, Any]:
    for attempt in range(1, retries + 1):
        try:
            response = llm.chat(
                messages=[
                    {"role": "system", "content": "You deliver concise, actionable retrieval diagnostics."},
                    {"role": "user", "content": prompt},
                ],
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
            )
            break
        except APITimeoutError:
            if attempt == retries:
                raise
            sleep(backoff)
            backoff *= 2
            continue
    if isinstance(response, dict) and {"summary", "action_items"} <= response.keys():
        return response  # Already structured
    if hasattr(response, "content"):
        return {"summary": response.content, "action_items": []}
    return {"summary": str(response), "action_items": []}


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def run() -> Path:
    eval_report = _load_eval_report(EVAL_REPORT_PATH)
    catalog = _load_indicator_catalog(INDICATOR_CATALOG_PATH)
    questions = eval_report.get("questions", [])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    start = perf_counter()
    for idx, question_block in enumerate(questions, start=1):
        if MAX_QUESTIONS and idx > MAX_QUESTIONS:
            break
        prompt = _build_prompt(question_block, catalog)
        if not prompt:
            continue  # nothing to diagnose
        feedback = _call_llm(prompt)
        results.append(
            {
                "query_id": question_block.get("query_id"),
                "query_text": question_block.get("query_text"),
                "metrics": question_block.get("metrics"),
                "missed_indicators": question_block.get("database_presence", {}).get("ingested_but_missed", []),
                "feedback": feedback,
            }
        )

    elapsed_ms = (perf_counter() - start) * 1000.0
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_run_id = eval_report.get("metadata", {}).get("run_id", "eval_run")
    output_path = OUTPUT_DIR / f"{base_run_id}-feedback-{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": {
                    "source_eval": str(EVAL_REPORT_PATH),
                    "run_id": base_run_id,
                    "generated_at": timestamp,
                    "llm_model": LLM_MODEL,
                    "question_count": len(results),
                    "duration_ms": elapsed_ms,
                },
                "questions": results,
            },
            handle,
            indent=2,
        )
    logger.info("Feedback written to %s", output_path)
    return output_path


if __name__ == "__main__":
    run()
