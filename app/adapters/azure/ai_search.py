from typing import List, Optional, Any
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchIndexerDataSourceConnection,
    SearchIndexerDataContainer,
    SearchIndexer
)
from app.core.interfaces import VectorStore
from app.core.models import Document, SearchResult
from app.core.config import settings
import json
import logging

logger = logging.getLogger(__name__)

class AzureAISearchStore(VectorStore):
    def __init__(self):
        self.endpoint = settings.AZURE_SEARCH_ENDPOINT
        self.index_name = settings.AZURE_SEARCH_INDEX_NAME
        self.credential = self._get_credential()
        
        self.index_client = SearchIndexClient(endpoint=self.endpoint, credential=self.credential)
        self.indexer_client = SearchIndexerClient(endpoint=self.endpoint, credential=self.credential)
        self.search_client = SearchClient(endpoint=self.endpoint, index_name=self.index_name, credential=self.credential)

        # Ensure index exists on init
        self.ensure_index()

    def _get_credential(self):
        if settings.AZURE_SEARCH_KEY:
            return AzureKeyCredential(settings.AZURE_SEARCH_KEY)
        return DefaultAzureCredential()

    def ensure_index(self):
        try:
            self.index_client.get_index(self.index_name)
            logger.info(f"Index {self.index_name} exists.")
        except Exception:
            logger.info(f"Creating index {self.index_name}...")
            self.create_index()

    def create_index(self):
        # Define fields
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(
                name="content", 
                type=SearchFieldDataType.String, 
                searchable=True,
                filterable=False,
                sortable=False,
                facetable=False
            ),
            SimpleField(name="metadata", type=SearchFieldDataType.String), # Storing metadata as JSON string for flexibility
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536, # Assuming OpenAI dim, should make configurable
                vector_search_profile_name="my-vector-profile"
            )
        ]

        # Define vector search config
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(name="my-hnsw-config")
            ],
            profiles=[
                VectorSearchProfile(
                    name="my-vector-profile",
                    algorithm_configuration_name="my-hnsw-config"
                )
            ]
        )

        index = SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)
        self.index_client.create_or_update_index(index)

    def delete_index(self):
        try:
            self.index_client.delete_index(self.index_name)
            logger.info(f"Index {self.index_name} deleted.")
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            raise

    async def add_documents(self, documents: List[Document]) -> bool:
        docs_to_upload = []
        for doc in documents:
            docs_to_upload.append({
                "id": doc.id,
                "content": doc.content,
                "metadata": json.dumps(doc.metadata),
                "embedding": doc.embedding
            })
        
        try:
            result = self.search_client.upload_documents(documents=docs_to_upload)
            return all(r.succeeded for r in result)
        except Exception as e:
            logger.error(f"Error uploading documents: {e}")
            return False

    async def search(self, query_embedding: List[float], top_k: int = 5, filters: Optional[dict] = None) -> List[SearchResult]:
        from azure.search.documents.models import VectorizedQuery
        vector_query = VectorizedQuery(
             vector=query_embedding,
             k_nearest_neighbors_count=top_k,
             fields="embedding"
        )
        
        try:
            # Note: The azure-search-documents library usage for vector search might vary slightly 
            # depending on version (using `vector_queries` argument in newer versions)
            results = self.search_client.search(
                search_text="*", # Pure vector search combined with filter if needed
                vector_queries=[vector_query],
                top=top_k,
                select=["id", "content", "metadata"]
            )
            
            search_results = []
            for res in results:
                doc = Document(
                    id=res["id"],
                    content=res["content"],
                    metadata=json.loads(res["metadata"]) if res["metadata"] else {},
                    embedding=None # Don't return embedding to save bandwidth
                )
                search_results.append(SearchResult(document=doc, score=res["@search.score"]))
            return search_results
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    async def search_text(self, query_text: str, top_k: int = 5) -> List[SearchResult]:
        try:
            results = self.search_client.search(
                search_text=query_text,
                top=top_k,
                select=["id", "content", "metadata"]
            )
            
            search_results = []
            for res in results:
                doc = Document(
                    id=res["id"],
                    content=res["content"],
                    metadata=json.loads(res["metadata"]) if res["metadata"] and res["metadata"] != "null" else {},
                    embedding=None
                )
                search_results.append(SearchResult(document=doc, score=res["@search.score"]))
            return search_results
        except Exception as e:
            logger.error(f"Error searching text: {e}")
            return []

    async def delete(self, document_id: str) -> bool:
        try:
            # Need strict structure for delete
            self.search_client.delete_documents(documents=[{"id": document_id}])
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False

    # Indexer Management
    def create_blob_data_source(self, data_source_name: str, connection_string: str, container_name: str):
        container = SearchIndexerDataContainer(name=container_name)
        data_source = SearchIndexerDataSourceConnection(
            name=data_source_name,
            type="azureblob",
            connection_string=connection_string,
            container=container
        )
        self.indexer_client.create_or_update_data_source_connection(data_source)

    def create_or_update_indexer(self, indexer_name: str, data_source_name: str, target_index_name: str = None):
        target_index = target_index_name or self.index_name
        indexer = SearchIndexer(
            name=indexer_name,
            data_source_name=data_source_name,
            target_index_name=target_index
        )
        self.indexer_client.create_or_update_indexer(indexer)

    def run_indexer(self, indexer_name: str):
        self.indexer_client.run_indexer(indexer_name)

    # Hybrid Embedding Workflow Methods
    def get_documents_without_embeddings(self, batch_size: int = 100) -> List[dict]:
        """
        Fetch documents that don't have embeddings.
        Returns list of documents with id, content, and metadata.
        
        Note: We fetch all documents and assume those from the indexer don't have embeddings.
        A more robust approach would be to add a 'has_embedding' boolean field to track this.
        """
        try:
            results = self.search_client.search(
                search_text="*",
                select=["id", "content", "metadata"],
                top=batch_size
            )
            
            docs_without_embeddings = []
            for doc in results:
                # For now, we'll try to identify indexer-sourced documents
                # They typically have blob storage URLs as IDs
                doc_id = doc["id"]
                
                # If the ID looks like a blob storage URL (base64 encoded), 
                # it's likely from the indexer and needs embeddings
                if doc_id.startswith("aHR0cHM6Ly9") or len(doc_id) > 100:
                    docs_without_embeddings.append({
                        "id": doc["id"],
                        "content": doc["content"],
                        "metadata": doc.get("metadata", "{}")
                    })
            
            logger.info(f"Found {len(docs_without_embeddings)} documents without embeddings")
            return docs_without_embeddings
        except Exception as e:
            logger.error(f"Error fetching documents without embeddings: {e}")
            return []

    def update_document_embedding(self, document_id: str, embedding: List[float]) -> bool:
        """
        Update a single document's embedding field.
        """
        try:
            self.search_client.merge_documents(documents=[{
                "id": document_id,
                "embedding": embedding
            }])
            return True
        except Exception as e:
            logger.error(f"Error updating embedding for document {document_id}: {e}")
            return False

    def batch_update_embeddings(self, documents: List[dict]) -> bool:
        """
        Batch update embeddings for multiple documents.
        Each document should have 'id' and 'embedding' fields.
        """
        try:
            docs_to_update = [
                {"id": doc["id"], "embedding": doc["embedding"]}
                for doc in documents
            ]
            result = self.search_client.merge_documents(documents=docs_to_update)
            return all(r.succeeded for r in result)
        except Exception as e:
            logger.error(f"Error batch updating embeddings: {e}")
            return False
