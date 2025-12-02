"""
Modular preprocessing and postprocessing functions.
Each function has a simple signature for easy composition.
"""
import re
from typing import List
from app.models.schemas import SearchResult

# ===== PREPROCESSING FUNCTIONS =====

def clean_query(query: str) -> str:
    """Remove extra whitespace and normalize text."""
    return " ".join(query.split()).strip()

def lowercase_query(query: str) -> str:
    """Convert query to lowercase."""
    return query.lower()

def expand_abbreviations(query: str) -> str:
    """Expand common financial abbreviations."""
    abbreviations = {
        "gdp": "gross domestic product",
        "cpi": "consumer price index",
        "yoy": "year over year",
        "qoq": "quarter over quarter"
    }
    
    words = query.lower().split()
    expanded = []
    for word in words:
        if word in abbreviations:
            expanded.append(abbreviations[word])
        else:
            expanded.append(word)
    return " ".join(expanded)

# ===== POSTPROCESSING FUNCTIONS =====

def deduplicate_results(results: List[SearchResult]) -> List[SearchResult]:
    """Remove duplicate indicators based on ID."""
    seen = set()
    unique = []
    for result in results:
        if result.indicator.id not in seen:
            seen.add(result.indicator.id)
            unique.append(result)
    return unique

def filter_by_threshold(results: List[SearchResult], threshold: float = 0.5) -> List[SearchResult]:
    """Filter results below a score threshold."""
    return [r for r in results if r.score >= threshold]

def limit_results(results: List[SearchResult], limit: int = 5) -> List[SearchResult]:
    """Limit the number of results."""
    return results[:limit]

# Registry for dynamic lookup
PREPROCESSING_REGISTRY = {
    "clean_query": clean_query,
    "lowercase_query": lowercase_query,
    "expand_abbreviations": expand_abbreviations
}

POSTPROCESSING_REGISTRY = {
    "deduplicate_results": deduplicate_results,
    "filter_by_threshold": filter_by_threshold,
    "limit_results": limit_results
}
