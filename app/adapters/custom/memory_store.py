from typing import List, Optional
import math
from app.core.interfaces import VectorStore
from app.core.models import Document, SearchResult

class MemoryVectorStore(VectorStore):
    def __init__(self):
        self._store: dict[str, Document] = {}

    async def add_documents(self, documents: List[Document]) -> bool:
        for doc in documents:
            self._store[doc.id] = doc
        return True

    async def search(self, query_embedding: List[float], top_k: int = 5, filters: Optional[dict] = None) -> List[SearchResult]:
        if not query_embedding:
            return []
        
        results = []
        for doc in self._store.values():
            if not doc.embedding:
                continue
            
            # Cosine similarity
            score = self._cosine_similarity(query_embedding, doc.embedding)
            results.append(SearchResult(document=doc, score=score))
        
        # Sort by score desc
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def delete(self, document_id: str) -> bool:
        if document_id in self._store:
            del self._store[document_id]
            return True
        return False

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = math.sqrt(sum(a * a for a in vec1))
        norm_b = math.sqrt(sum(b * b for b in vec2))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
