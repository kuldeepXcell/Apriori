"""Lightweight retrieval metrics used by manual eval runs."""

from __future__ import annotations

from typing import Iterable, Sequence


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def precision_at_k(predictions: Sequence[str], truths: set[str], k: int) -> float:
    """Fraction of the top-k predictions that appear in the truth set."""
    if not predictions or not truths or k <= 0:
        return 0.0
    top = [p for p in predictions[:k] if p]
    hits = sum(1 for p in top if p in truths)
    return _safe_div(hits, min(k, len(top)))


def recall_at_k(predictions: Sequence[str], truths: set[str], k: int) -> float:
    """Share of true indicators that appear in the top-k predictions."""
    if not predictions or not truths or k <= 0:
        return 0.0
    top = predictions[:k]
    hits = sum(1 for p in top if p in truths)
    return _safe_div(hits, len(truths))


def mean_reciprocal_rank(predictions: Sequence[str], truths: set[str]) -> float:
    """Reciprocal rank of the first correct prediction (0 when none match)."""
    if not predictions or not truths:
        return 0.0
    for idx, pred in enumerate(predictions):
        if pred and pred in truths:
            return 1.0 / float(idx + 1)
    return 0.0


def final_score(
    precision: float,
    recall: float,
    mrr: float,
    *,
    precision_weight: float,
    recall_weight: float,
    mrr_weight: float,
) -> float:
    """Weighted blend of precision/recall/MRR used for quick comparisons."""
    return (
        precision_weight * precision
        + recall_weight * recall
        + mrr_weight * mrr
    )


def average(values: Iterable[float]) -> float:
    """Arithmetic mean with empty-safe default."""
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
