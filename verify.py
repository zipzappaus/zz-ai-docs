import asyncio
import os
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

from app.core.models import Document, SearchQuery
from app.api.dependencies import get_ingestion_service, get_vector_store

async def main():
    print("Starting verification...")
    
    # Get services
    ingestion_service = get_ingestion_service()
    vector_store = get_vector_store()

    # Create dummy document
    doc = Document(
        content="The quick brown fox jumps over the lazy dog.",
        metadata={"category": "test", "author": "user"}
    )
    print(f"Created document: {doc.id}")

    # Ingest
    print("Ingesting document...")
    success = await ingestion_service.ingest_documents([doc])
    if success:
        print("Ingestion successful.")
    else:
        print("Ingestion failed.")
        return

    # Search
    print("Searching for 'brown fox'...")
    # Note: RandomEmbeddingModel is deterministic based on input length/seed, 
    # so 'brown fox' will generate a vector that has SOME relationship (or not) to the doc.
    # But since we use random numbers, cosine similarity might be low or high randomly.
    # However, MemoryVectorStore calculates it correctly. We just want to see it runs without error
    # and returns *something* or handles empty.
    
    # We manually simulate search flow since we don't have the API running
    # but we can call vector_store.search directly if we embed the query first.
    from app.api.dependencies import get_embedding_model
    embedding_model = get_embedding_model()
    query_text = "brown fox"
    query_embedding = await embedding_model.embed_text(query_text)
    
    results = await vector_store.search(query_embedding, top_k=1)
    
    print(f"Found {len(results)} results.")
    for res in results:
        print(f"Doc ID: {res.document.id}, Score: {res.score:.4f}, Content: {res.document.content}")

    print("Verification complete.")

if __name__ == "__main__":
    asyncio.run(main())
