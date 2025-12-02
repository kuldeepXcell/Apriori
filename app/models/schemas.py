from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class FinancialIndicator(BaseModel):
    """
    Represents a single financial indicator.
    """
    id: str = Field(..., description="Unique identifier for the indicator")
    name: str = Field(..., description="Name of the indicator")
    definition: str = Field(..., description="Detailed definition of the indicator")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata (e.g., source, category)")
    
    # Search fields
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords for sparse search")
    
    class Config:
        from_attributes = True

class SearchResult(BaseModel):
    """
    Represents a search result from the retrieval system.
    """
    indicator: FinancialIndicator
    score: float
    relevance_reason: Optional[str] = None
