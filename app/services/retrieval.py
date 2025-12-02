from typing import List
from app.services.qdrant_service import qdrant_service
from app.services.llm_service import llm_service
from app.models.schemas import SearchResult, FinancialIndicator

class RetrievalService:
    def check_moderation(self, query: str) -> bool:
        """
        Checks if the query is safe/relevant.
        TODO: Implement actual Moderation API call.
        """
        # Simple keyword check for now
        forbidden_words = ["ignore", "hack", "system"]
        for word in forbidden_words:
            if word in query.lower():
                return False
        return True

    def search_indicators(self, query: str) -> List[SearchResult]:
        """
        Orchestrates the search flow: Moderation -> Hybrid Search -> Re-ranking.
        """
        # 1. Moderation
        if not self.check_moderation(query):
            raise ValueError("Query flagged by moderation system.")

        # 2. Generate Query Embedding
        query_embedding = llm_service.get_embedding(query)

        # 3. Hybrid Search (Qdrant)
        # Currently using dense search, TODO: Add keyword filter
        sc_points = qdrant_service.search(query_vector=query_embedding, limit=15)
        
        if not sc_points:
            return []

        # Convert to FinancialIndicator objects
        candidates = []
        for point in sc_points:
            candidates.append(FinancialIndicator(**point.payload))

        # 4. LLM Context Re-ranker
        # We pass the candidates to the LLM to filter and re-rank
        final_results = self.rerank_with_llm(query, candidates)
        
        return final_results

    def rerank_with_llm(self, query: str, candidates: List[FinancialIndicator]) -> List[SearchResult]:
        """
        Uses LLM to re-rank and filter the candidates based on the query.
        """
        # Construct prompt
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
        response_text = llm_service.get_chat_completion(messages)
        
        # Parse response (Simple parsing for now, should be robust JSON parsing)
        try:
            import json
            # Extract JSON from potential markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
                
            indices = json.loads(response_text)
            
            results = []
            for idx in indices:
                if 0 <= idx < len(candidates):
                    # Assign a dummy score based on rank
                    score = 1.0 / (indices.index(idx) + 1)
                    results.append(SearchResult(
                        indicator=candidates[idx],
                        score=score,
                        relevance_reason="Selected by LLM"
                    ))
            return results
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            # Fallback: return top 5 from original search
            return [SearchResult(indicator=c, score=0.9, relevance_reason="Fallback") for c in candidates[:5]]

retrieval_service = RetrievalService()
