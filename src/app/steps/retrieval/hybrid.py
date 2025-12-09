"""Hybrid multivector + HyDe retrieval against Qdrant."""
from __future__ import annotations

import json
from dataclasses import replace
from typing import List, Optional

from openai import OpenAI
from qdrant_client.http import models

from app.adapters.embedding import openai_embedding_adapter
from app.adapters.qdrant import qdrant_adapter
from app.config.settings import settings
from app.pipeline.models import (
    Indicator,
    MultivectorWeights,
    SearchHit,
    SearchResponse,
)


DEFAULT_LIMIT = 15


class HybridRetriever:
    """Encapsulates multivector and HyDe query flows."""

    def __init__(
        self,
        collection_name: str,
        embedding_model: Optional[str],
        reranker_model: Optional[str] = None,
    ) -> None:
        if not collection_name:
            raise ValueError("collection_name is required for retrieval.")
        self.collection_name = collection_name
        self.embedding_model = embedding_model or "text-embedding-3-small"
        self.reranker_model = reranker_model or "gpt-5-mini"
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        self._llm_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def multivector_search(
        self, query: str, weights: MultivectorWeights, limit: int = DEFAULT_LIMIT
    ) -> SearchResponse:
        """Search using weighted named vectors."""
        try:
            norm_w = weights.normalized()
            def_vec, ques_vec, app_vec = self._embed_query_triplet(query)
            aggregate_vec = _weighted_average(
                definition=def_vec,
                question=ques_vec,
                application=app_vec,
                weights=norm_w,
            )
            points = self._search_points(
                vector_name="definition",
                vector=aggregate_vec,
                limit=limit,
            )
            hits = _convert_points_to_hits(points.points if points else [])
            reranked = self._rerank_hits(query, hits, approach="multivector")
            return SearchResponse(
                hits=hits,
                reranked=reranked,
            )
        except Exception as exc:
            return SearchResponse(hits=[], reranked=[], error=str(exc))

    def hyde_search(self, query: str, limit: int = DEFAULT_LIMIT) -> SearchResponse:
        """Generate a hypothetical definition, embed it, and search."""
        try:
            generated_def = self._generate_definition(query)
            def_vec = openai_embedding_adapter.embed_one(
                generated_def, model=self.embedding_model
            )
            points = self._search_points(
                vector_name="definition",
                vector=def_vec,
                limit=limit,
            )
            hits = _convert_points_to_hits(points.points if points else [])
            reranked = self._rerank_hits(query, hits, approach="hyde")
            return SearchResponse(
                hits=hits,
                reranked=reranked,
                generated_definition=generated_def,
            )
        except Exception as exc:
            return SearchResponse(hits=[], reranked=[], generated_definition=None, error=str(exc))

    def _embed_query_triplet(self, query: str) -> tuple[List[float], List[float], List[float]]:
        """Embed query text three ways to align with named vectors."""
        prompts = [
            f"Definition style: {query}",
            f"Question style: {query}",
            f"Application context style: {query}",
        ]
        vectors = openai_embedding_adapter.embed(prompts, model=self.embedding_model)
        definition_vec = vectors[0] if len(vectors) > 0 else []
        question_vec = vectors[1] if len(vectors) > 1 else []
        application_vec = vectors[2] if len(vectors) > 2 else []
        return definition_vec, question_vec, application_vec

    def _search_points(
        self,
        vector_name: str,
        vector: List[float],
        limit: int,
    ) -> models.QueryResponse:
        """Execute a Qdrant query against a specific named vector."""
        nearest = models.NearestQuery(nearest=vector)
        return qdrant_adapter.client.query_points(
            collection_name=self.collection_name,
            query=nearest,
            using=vector_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

    def _generate_definition(self, query: str) -> str:
        """Generate a hypothetical definition for HyDe."""
        prompt = (
            "You generate concise indicator definitions (not questions) for retrieval. "
            "Given the user's question, output definition capturing the core concepts discussed in question."
            "We are using HyDe method to fetch relevant indicators based on cosine similarity between this definition and the definition of indicators in the database."
            "Examples:\n"
            "1) Fixed broadband subscriptions refers to fixed subscriptions to high-speed access to the public Internet "
            "(a TCP/IP connection), at downstream speeds equal to, or greater than, 256 kbit/s. This includes cable "
            "modem, DSL, fiber-to-the-home/building, other fixed (wired)-broadband subscriptions, satellite broadband "
            "and terrestrial fixed wireless broadband. This total is measured irrespective of the method of payment. "
            "It excludes subscriptions that have access to data communications (including the Internet) via mobile-cellular "
            "networks. It should include fixed WiMAX and any other fixed wireless technologies. It includes both residential "
            "subscriptions and subscriptions for organizations.\n"
            "2) Index tracking prices received by domestic producers for their output.\n"
            "3) Labor force participation rate is the proportion of the population ages 15 and older that is economically "
            "active: all people who supply labor for the production of goods and services during a specified period."
        )
        response = self._llm_client.chat.completions.create(
            model=self.reranker_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            max_completion_tokens=120,
        )
        choice = response.choices[0].message.content if response.choices else ""
        return choice.strip() if choice else query

    def _rerank_hits(
        self, query: str, hits: List[SearchHit], approach: str
    ) -> List[SearchHit]:
        """Use LLM to rerank top hits and attach reasons."""
        if not hits:
            return []

        # Prepare candidate summaries
        parts = []
        for idx, hit in enumerate(hits):
            ind = hit.indicator
            parts.append(
                f"{idx+1}. name={ind.name}; normalized_name={ind.normalized_name or ind.name}; "
                f"definition={ind.definition or 'N/A'}; "
                f"application={ind.application_context or 'N/A'}; "
                f"question={ind.question or 'N/A'}; "
                f"keywords={', '.join(ind.keywords or [])}"
            )
        candidates_text = "\n".join(parts)

        system_prompt = (
            "You are an expert financial indicator reranker. "
            "Given a user query and candidate indicators with definition, application context, and question text, "
            "return a JSON array sorted best-to-worst by semantic relevance to the query. "
            "Each element must be an object with keys: normalized_name, reason. "
            "Only include the 15 provided candidates; do not invent new names."
        )
        user_prompt = (
            f"Query: {query}\n"
            f"Approach: {approach}\n"
            f"Candidates:\n{candidates_text}\n\n"
            "Return JSON only, e.g. "
            '[{"normalized_name": "gdp_growth", "reason": "Matches GDP growth topic"}, ...]'
        )

        try:
            response = self._llm_client.chat.completions.create(
                model=self.reranker_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content if response.choices else "[]"
            parsed = json.loads(content)
        except Exception:
            return hits[:3]

        name_to_hit = {
            (h.indicator.normalized_name or h.indicator.name): h for h in hits
        }

        reranked: List[SearchHit] = []
        for item in parsed:
            norm_name = item.get("normalized_name")
            reason = item.get("reason")
            if not norm_name:
                continue
            base = name_to_hit.get(norm_name)
            if base:
                reranked.append(
                    replace(base, relevance_reason=reason)
                )
        # Fallback append any missing in original order
        seen = {h.indicator.normalized_name or h.indicator.name for h in reranked}
        for h in hits:
            name = h.indicator.normalized_name or h.indicator.name
            if name not in seen:
                reranked.append(h)
        return reranked[:len(hits)]


def _weighted_average(
    definition: List[float],
    question: List[float],
    application: List[float],
    weights: MultivectorWeights,
) -> List[float]:
    """Combine three vectors using normalized weights."""
    combined: List[float] = []
    for idx in range(max(len(definition), len(question), len(application))):
        d = definition[idx] if idx < len(definition) else 0.0
        q = question[idx] if idx < len(question) else 0.0
        a = application[idx] if idx < len(application) else 0.0
        combined.append(
            (d * weights.context) + (q * weights.question) + (a * weights.application)
        )
    return combined


def _convert_points_to_hits(points: List[models.ScoredPoint]) -> List[SearchHit]:
    hits: List[SearchHit] = []
    for pt in points:
        payload = pt.payload or {}
        indicator = Indicator(
            name=payload.get("indicator_name") or payload.get("normalized_indicator_name") or "Unknown indicator",
            normalized_name=payload.get("normalized_indicator_name"),
            subsection=payload.get("subsection"),
            subsubsection=payload.get("subsubsection"),
            definition=payload.get("definition"),
            question=payload.get("question"),
            application_context=payload.get("application_context"),
            keywords=payload.get("keywords"),
            source=payload.get("source"),
            sheet_name=payload.get("sheet_name"),
        )
        hits.append(
            SearchHit(
                indicator=indicator,
                score=pt.score or 0.0,
                relevance_reason=None,
            )
        )
    return hits

