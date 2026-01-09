from typing import List
import random
from app.core.interfaces import EmbeddingModel
from app.core.models import Document

class RandomEmbeddingModel(EmbeddingModel):
    def __init__(self, check_dim: int = 1536):
        self.dim = check_dim

    async def embed_text(self, text: str) -> List[float]:
        # Deterministic random based on text length to have somewhat consistent results for same text
        # (Not for real semantic search, just for plumbing test)
        random.seed(len(text)) 
        return [random.random() for _ in range(self.dim)]

    async def embed_documents(self, documents: List[Document]) -> List[Document]:
        for doc in documents:
            if not doc.embedding:
                doc.embedding = await self.embed_text(doc.content)
        return documents
