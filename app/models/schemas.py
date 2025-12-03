from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class FinancialIndicator(BaseModel):
    """
    Represents a single financial indicator.
    """
    id: str = Field(..., description="Unique identifier for the indicator")
    normalized_name: str = Field(..., description="Normalized slug for the indicator")
    name: str = Field(..., description="Name of the indicator")
    definition: str = Field(..., description="Detailed definition of the indicator")
    question: str = Field(..., description="Guiding question that the indicator answers")
    application_context: str = Field(..., description="How to apply or interpret the indicator")
    subsection: Optional[str] = Field(default=None, description="Parent section for the indicator")
    subsubsection: Optional[str] = Field(default=None, description="Sub-section for finer grouping")
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
