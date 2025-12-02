from typing import List
from app.configs.pipeline_config import PipelineConfig
from app.services.qdrant_service import qdrant_service
from app.services.llm_service import llm_service
from app.models.schemas import SearchResult, FinancialIndicator
from app.pipelines.processors import PREPROCESSING_REGISTRY, POSTPROCESSING_REGISTRY
import json
import time

class RetrievalPipeline:
    """
    Base retrieval pipeline that executes the full search flow.
    Each pipeline uses a specific configuration of models and processors.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.execution_time = 0.0
        
    def search(self, query: str) -> List[SearchResult]:
        """
        Execute the full retrieval pipeline.
        
        Steps: Moderation -> Preprocess -> Embed -> Search -> Re-rank -> Postprocess
        """
        start_time = time.time()
        
        try:
            # 1. Moderation
            if not self._check_moderation(query):
                raise ValueError("Query flagged by moderation system.")
            
            # 2. Preprocessing
            processed_query = self._preprocess(query)
            
            # 3. Generate Embedding
            query_embedding = llm_service.get_embedding(
                processed_query,
                model=self.config.embedding_model
            )
            
            # 4. Vector Search
            scored_points = qdrant_service.search(
                collection_name=self.config.collection_name,
                query_vector=query_embedding,
                limit=self.config.search_limit
            )
            
            if not scored_points:
                return []
            
            # Convert to FinancialIndicator objects
            candidates = [FinancialIndicator(**point.payload) for point in scored_points]
            
            # 5. LLM Re-ranking
            results = self._rerank_with_llm(processed_query, candidates)
            
            # 6. Postprocessing
            final_results = self._postprocess(results)
            
            self.execution_time = time.time() - start_time
            return final_results
            
        except Exception as e:
            self.execution_time = time.time() - start_time
            print(f"Pipeline '{self.config.name}' error: {e}")
            raise
    
    def _check_moderation(self, query: str) -> bool:
        """Simple moderation check."""
        forbidden_words = ["ignore", "hack", "system"]
        return not any(word in query.lower() for word in forbidden_words)
    
    def _preprocess(self, query: str) -> str:
        """Apply preprocessing steps defined in config."""
        processed = query
        for step_name in self.config.preprocessing_steps:
            if step_name in PREPROCESSING_REGISTRY:
                func = PREPROCESSING_REGISTRY[step_name]
                processed = func(processed)
        return processed
    
    def _postprocess(self, results: List[SearchResult]) -> List[SearchResult]:
        """Apply postprocessing steps defined in config."""
        processed = results
        for step_name in self.config.postprocessing_steps:
            if step_name in POSTPROCESSING_REGISTRY:
                func = POSTPROCESSING_REGISTRY[step_name]
                processed = func(processed)
        return processed
    
    def _rerank_with_llm(self, query: str, candidates: List[FinancialIndicator]) -> List[SearchResult]:
        """Use LLM to re-rank and filter candidates."""
        candidates_text = "\n".join([f"{i}. {c.name}: {c.definition}" for i, c in enumerate(candidates)])
        
        prompt = f"""
You are a financial expert. Analyze the following list of financial indicators and identify which ones are most relevant to the user's query.

User Query: "{query}"

Candidates:
{candidates_text}

Return a JSON list of indices (0-based) of the relevant indicators, sorted by relevance. 
Example: [0, 3, 1]
If none are relevant, return [].
"""
        
        messages = [{"role": "user", "content": prompt}]
        response_text = llm_service.get_chat_completion(
            messages,
            model=self.config.llm_model,
            temperature=self.config.llm_temperature
        )
        
        # Parse response
        try:
            # Extract JSON from potential markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
                
            indices = json.loads(response_text)
            
            results = []
            for idx in indices[:self.config.rerank_limit]:
                if 0 <= idx < len(candidates):
                    score = 1.0 / (indices.index(idx) + 1)
                    results.append(SearchResult(
                        indicator=candidates[idx],
                        score=score,
                        relevance_reason=f"Ranked #{indices.index(idx) + 1} by {self.config.llm_model}"
                    ))
            return results
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            # Fallback: return top N from original search
            return [
                SearchResult(
                    indicator=c,
                    score=0.9 - (i * 0.1),
                    relevance_reason="Fallback (LLM parsing failed)"
                )
                for i, c in enumerate(candidates[:self.config.rerank_limit])
            ]
