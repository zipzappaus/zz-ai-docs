from typing import List, Optional, Dict, Any
from app.core.interfaces import VectorStore
from app.core.models import Document, SearchResult
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class PineconeStore(VectorStore):
    """
    Pinecone vector store implementation.
    
    Provides document storage, retrieval, and semantic search using Pinecone's
    managed vector database service.
    """
    
    def __init__(self):
        """Initialize Pinecone client and connect to index."""
        try:
            from pinecone import Pinecone, ServerlessSpec
            
            if not settings.PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY is not configured")
            
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self.index_name = settings.PINECONE_INDEX_NAME
            self.dimension = settings.PINECONE_DIMENSION
            self.metric = settings.PINECONE_METRIC
            self.cloud = settings.PINECONE_CLOUD
            self.region = settings.PINECONE_REGION
            
            # Connect to index (will be created if it doesn't exist)
            self.index = None
            self._connect_to_index()
            
            logger.info(f"Pinecone client initialized for index: {self.index_name}")
            
        except ImportError:
            raise ImportError(
                "Pinecone SDK not installed. Install with: pip install pinecone-client>=3.0.0"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {e}")
            raise
    
    def _connect_to_index(self):
        """Connect to the Pinecone index, creating it if necessary."""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]
            
            if self.index_name not in index_names:
                logger.warning(f"Index '{self.index_name}' does not exist. Call ensure_index() to create it.")
                return
            
            # Connect to existing index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone index: {e}")
            raise
    
    def ensure_index(self) -> bool:
        """
        Ensure the index exists, creating it if necessary.
        
        Returns:
            bool: True if index exists or was created successfully
        """
        try:
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]
            
            if self.index_name in index_names:
                logger.info(f"Index '{self.index_name}' already exists")
                if not self.index:
                    self.index = self.pc.Index(self.index_name)
                return True
            
            # Create new index
            return self.create_index()
            
        except Exception as e:
            logger.error(f"Failed to ensure index exists: {e}")
            return False
    
    def create_index(self, dimension: int = None, metric: str = None) -> bool:
        """
        Create a new Pinecone index.
        
        Args:
            dimension: Vector dimension (default from settings)
            metric: Similarity metric - cosine, euclidean, or dotproduct (default from settings)
            
        Returns:
            bool: True if index was created successfully
        """
        try:
            from pinecone import ServerlessSpec
            
            dim = dimension or self.dimension
            met = metric or self.metric
            
            logger.info(f"Creating Pinecone index '{self.index_name}' with dimension={dim}, metric={met}")
            
            self.pc.create_index(
                name=self.index_name,
                dimension=dim,
                metric=met,
                spec=ServerlessSpec(
                    cloud=self.cloud,
                    region=self.region
                )
            )
            
            # Wait for index to be ready
            import time
            max_wait = 60  # seconds
            waited = 0
            while waited < max_wait:
                try:
                    self.index = self.pc.Index(self.index_name)
                    stats = self.index.describe_index_stats()
                    logger.info(f"Index '{self.index_name}' created successfully")
                    return True
                except Exception:
                    time.sleep(2)
                    waited += 2
            
            logger.warning(f"Index creation timed out after {max_wait}s")
            return False
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False
    
    def delete_index(self) -> bool:
        """
        Delete the Pinecone index.
        
        Returns:
            bool: True if index was deleted successfully
        """
        try:
            self.pc.delete_index(self.index_name)
            self.index = None
            logger.info(f"Deleted Pinecone index: {self.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete index: {e}")
            return False
    
    def describe_index(self) -> Dict[str, Any]:
        """
        Get index statistics and configuration.
        
        Returns:
            dict: Index metadata including dimension, metric, vector count, etc.
        """
        try:
            if not self.index:
                raise ValueError("Not connected to any index")
            
            stats = self.index.describe_index_stats()
            index_info = self.pc.describe_index(self.index_name)
            
            return {
                "name": self.index_name,
                "dimension": index_info.dimension,
                "metric": index_info.metric,
                "total_vector_count": stats.get("total_vector_count", 0),
                "namespaces": stats.get("namespaces", {}),
                "index_fullness": stats.get("index_fullness", 0.0),
            }
            
        except Exception as e:
            logger.error(f"Failed to describe index: {e}")
            return {}
    
    async def add_documents(self, documents: List[Document], namespace: str = "") -> bool:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects with embeddings
            namespace: Optional namespace for organizing vectors
            
        Returns:
            bool: True if documents were added successfully
        """
        try:
            if not self.index:
                raise ValueError("Not connected to any index. Call ensure_index() first.")
            
            if not documents:
                logger.warning("No documents to add")
                return True
            
            # Prepare vectors for upsert
            vectors = []
            for doc in documents:
                if not doc.embedding:
                    logger.warning(f"Document {doc.id} has no embedding, skipping")
                    continue
                
                # Prepare metadata (Pinecone has size limits, so keep it minimal)
                metadata = {
                    "content": doc.content[:1000],  # Truncate content for metadata
                    "created_at": doc.created_at.isoformat(),
                    **{k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))}
                }
                
                vectors.append({
                    "id": doc.id,
                    "values": doc.embedding,
                    "metadata": metadata
                })
            
            if not vectors:
                logger.warning("No documents with embeddings to add")
                return False
            
            # Batch upsert (Pinecone recommends batches of 100)
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)
                logger.info(f"Upserted batch of {len(batch)} vectors to namespace '{namespace}'")
            
            logger.info(f"Successfully added {len(vectors)} documents to Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    async def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5, 
        filters: Optional[dict] = None,
        namespace: str = ""
    ) -> List[SearchResult]:
        """
        Search for documents using a query embedding.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional metadata filters (Pinecone format)
            namespace: Optional namespace to search within
            
        Returns:
            List of SearchResult objects
        """
        try:
            if not self.index:
                raise ValueError("Not connected to any index. Call ensure_index() first.")
            
            # Query Pinecone
            query_params = {
                "vector": query_embedding,
                "top_k": top_k,
                "include_metadata": True,
                "namespace": namespace
            }
            
            if filters:
                query_params["filter"] = filters
            
            results = self.index.query(**query_params)
            
            # Convert to SearchResult objects
            search_results = []
            for match in results.get("matches", []):
                # Reconstruct Document from metadata
                metadata = match.get("metadata", {})
                content = metadata.pop("content", "")
                created_at_str = metadata.pop("created_at", None)
                
                # Parse created_at
                from datetime import datetime
                created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.utcnow()
                
                doc = Document(
                    id=match["id"],
                    content=content,
                    metadata=metadata,
                    embedding=match.get("values"),
                    created_at=created_at
                )
                
                search_results.append(SearchResult(
                    document=doc,
                    score=match["score"]
                ))
            
            logger.info(f"Search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def delete(self, document_id: str, namespace: str = "") -> bool:
        """
        Delete a document by ID.
        
        Args:
            document_id: ID of the document to delete
            namespace: Optional namespace
            
        Returns:
            bool: True if document was deleted successfully
        """
        try:
            if not self.index:
                raise ValueError("Not connected to any index. Call ensure_index() first.")
            
            self.index.delete(ids=[document_id], namespace=namespace)
            logger.info(f"Deleted document {document_id} from namespace '{namespace}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False
    
    def batch_upsert(self, vectors: List[dict], namespace: str = "") -> bool:
        """
        Batch upsert vectors for efficiency.
        
        Args:
            vectors: List of vector dicts with id, values, and metadata
            namespace: Optional namespace
            
        Returns:
            bool: True if upsert was successful
        """
        try:
            if not self.index:
                raise ValueError("Not connected to any index. Call ensure_index() first.")
            
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)
                logger.info(f"Batch upserted {len(batch)} vectors")
            
            return True
            
        except Exception as e:
            logger.error(f"Batch upsert failed: {e}")
            return False
    
    def update_metadata(self, document_id: str, metadata: dict, namespace: str = "") -> bool:
        """
        Update metadata for an existing vector.
        
        Args:
            document_id: ID of the document
            metadata: New metadata dict
            namespace: Optional namespace
            
        Returns:
            bool: True if update was successful
        """
        try:
            if not self.index:
                raise ValueError("Not connected to any index. Call ensure_index() first.")
            
            # Fetch existing vector
            fetch_result = self.index.fetch(ids=[document_id], namespace=namespace)
            
            if document_id not in fetch_result.get("vectors", {}):
                logger.warning(f"Document {document_id} not found")
                return False
            
            vector_data = fetch_result["vectors"][document_id]
            
            # Update with new metadata
            self.index.upsert(
                vectors=[{
                    "id": document_id,
                    "values": vector_data["values"],
                    "metadata": metadata
                }],
                namespace=namespace
            )
            
            logger.info(f"Updated metadata for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")
            return False
    
    def fetch_by_ids(self, ids: List[str], namespace: str = "") -> List[Document]:
        """
        Fetch vectors by their IDs.
        
        Args:
            ids: List of document IDs
            namespace: Optional namespace
            
        Returns:
            List of Document objects
        """
        try:
            if not self.index:
                raise ValueError("Not connected to any index. Call ensure_index() first.")
            
            fetch_result = self.index.fetch(ids=ids, namespace=namespace)
            
            documents = []
            for doc_id, vector_data in fetch_result.get("vectors", {}).items():
                metadata = vector_data.get("metadata", {})
                content = metadata.pop("content", "")
                created_at_str = metadata.pop("created_at", None)
                
                from datetime import datetime
                created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.utcnow()
                
                doc = Document(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                    embedding=vector_data.get("values"),
                    created_at=created_at
                )
                documents.append(doc)
            
            logger.info(f"Fetched {len(documents)} documents by ID")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to fetch documents: {e}")
            return []
    
    def list_namespaces(self) -> List[str]:
        """
        List all namespaces in the index.
        
        Returns:
            List of namespace names
        """
        try:
            if not self.index:
                raise ValueError("Not connected to any index. Call ensure_index() first.")
            
            stats = self.index.describe_index_stats()
            namespaces = list(stats.get("namespaces", {}).keys())
            
            logger.info(f"Found {len(namespaces)} namespaces")
            return namespaces
            
        except Exception as e:
            logger.error(f"Failed to list namespaces: {e}")
            return []
