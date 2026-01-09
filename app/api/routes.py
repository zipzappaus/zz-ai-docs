from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from app.core.models import Document, SearchQuery, SearchResult
from app.core.interfaces import VectorStore, EmbeddingModel
from app.services.ingestion import IngestionService
from app.api.dependencies import get_vector_store, get_ingestion_service, get_embedding_model

router = APIRouter()

@router.post("/ingest/text", response_model=bool)
async def ingest_text(
    document: Document,
    service: IngestionService = Depends(get_ingestion_service)
):
    """Ingest a single raw text document."""
    return await service.ingest_documents([document])

@router.post("/search", response_model=List[SearchResult])
async def search(
    query: SearchQuery,
    vector_store: VectorStore = Depends(get_vector_store),
    embedding_model: EmbeddingModel = Depends(get_embedding_model)
):
    """Search for documents."""
    # Generate embedding for the query
    query_embedding = await embedding_model.embed_text(query.text)
    
    # Search in vector store
    results = await vector_store.search(
        query_embedding=query_embedding,
        top_k=query.top_k,
        filters=query.filters
    )
    return results

@router.delete("/documents/{document_id}", response_model=bool)
async def delete_document(
    document_id: str,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Delete a document by ID."""
    return await vector_store.delete(document_id)
