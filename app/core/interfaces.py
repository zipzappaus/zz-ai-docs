from abc import ABC, abstractmethod
from typing import List, Optional, Any
from app.core.models import Document, SearchResult, SearchQuery

class VectorStore(ABC):
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store."""
        pass

    @abstractmethod
    async def search(self, query_embedding: List[float], top_k: int = 5, filters: Optional[dict] = None) -> List[SearchResult]:
        """Search for documents using a query embedding."""
        pass

    @abstractmethod
    async def delete(self, document_id: str) -> bool:
        """Delete a document by ID."""
        pass

class EmbeddingModel(ABC):
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single string."""
        pass

    @abstractmethod
    async def embed_documents(self, documents: List[Document]) -> List[Document]:
        """Generate embeddings for a list of documents and update them in-place or return new objects."""
        pass

class DocumentStorage(ABC):
    @abstractmethod
    async def upload(self, file: Any, filename: str) -> str:
        """Upload a file and return the path/url."""
        pass

    @abstractmethod
    async def get_url(self, file_path: str) -> str:
        """Get accessible URL/Path for the file."""
        pass
