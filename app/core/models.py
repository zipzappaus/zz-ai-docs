from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class SearchQuery(BaseModel):
    text: str
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None

class SearchResult(BaseModel):
    document: Document
    score: float
