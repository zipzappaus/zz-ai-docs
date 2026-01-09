from functools import lru_cache
from app.core.config import settings
from app.core.interfaces import VectorStore, EmbeddingModel, DocumentStorage
from app.adapters.custom.memory_store import MemoryVectorStore
from app.adapters.custom.random_embedding import RandomEmbeddingModel
from app.adapters.custom.local_storage import LocalDocumentStorage
from app.services.ingestion import IngestionService

# Global singletons for memory store to persist data in-memory during runtime
_memory_store = MemoryVectorStore()
_random_embedding = RandomEmbeddingModel()
_local_storage = LocalDocumentStorage()

def get_vector_store() -> VectorStore:
    if settings.VECTOR_STORE_TYPE == "memory":
        return _memory_store
    # Placeholder for other backends
    # elif settings.VECTOR_STORE_TYPE == "azure_search":
    #     return AzureSearchStore(...)
    raise NotImplementedError(f"Vector store type {settings.VECTOR_STORE_TYPE} not implemented")

def get_embedding_model() -> EmbeddingModel:
    if settings.EMBEDDING_MODEL_TYPE == "random":
        return _random_embedding
    # Placeholder for other backends
    # elif settings.EMBEDDING_MODEL_TYPE == "openai":
    #     return OpenAIEmbedding(...)
    raise NotImplementedError(f"Embedding model type {settings.EMBEDDING_MODEL_TYPE} not implemented")

def get_document_storage() -> DocumentStorage:
    return _local_storage

def get_ingestion_service() -> IngestionService:
    store = get_vector_store()
    model = get_embedding_model()
    return IngestionService(vector_store=store, embedding_model=model)
