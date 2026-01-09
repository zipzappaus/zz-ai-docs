from typing import List
from app.core.interfaces import VectorStore, EmbeddingModel
from app.core.models import Document

class IngestionService:
    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    async def ingest_documents(self, documents: List[Document]) -> bool:
        # 1. Generate embeddings
        docs_with_embeddings = await self.embedding_model.embed_documents(documents)
        
        # 2. Store in vector store
        success = await self.vector_store.add_documents(docs_with_embeddings)
        return success
