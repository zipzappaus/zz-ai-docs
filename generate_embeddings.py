import asyncio
import sys
import os
from dotenv import load_dotenv
from typing import List

# Load env vars
load_dotenv()

# Add project root to sys.path
sys.path.append(os.getcwd())

from app.adapters.azure.ai_search import AzureAISearchStore
from app.adapters.custom.random_embedding import RandomEmbeddingModel
from app.core.config import settings

async def generate_embeddings_for_documents(batch_size: int = 100, max_documents: int = None):
    """
    Generate embeddings for documents that don't have them.
    
    Args:
        batch_size: Number of documents to fetch per batch
        max_documents: Maximum number of documents to process (None = all)
    """
    print("=" * 60)
    print("Embedding Generation Process")
    print("=" * 60)
    
    # Initialize store and embedding model
    print("\n[1/5] Initializing Azure AI Search store...")
    try:
        store = AzureAISearchStore()
        print(f"✓ Connected to {settings.AZURE_SEARCH_ENDPOINT}")
        print(f"✓ Using index: {settings.AZURE_SEARCH_INDEX_NAME}")
    except Exception as e:
        print(f"✗ Failed to initialize store: {e}")
        return
    
    print("\n[2/5] Initializing embedding model...")
    embedder = RandomEmbeddingModel()
    print("✓ Using RandomEmbeddingModel (1536 dimensions)")
    
    # Fetch documents without embeddings
    print(f"\n[3/5] Fetching documents without embeddings (batch size: {batch_size})...")
    docs_to_process = store.get_documents_without_embeddings(batch_size=batch_size)
    
    if not docs_to_process:
        print("✓ No documents found without embeddings. All documents are up to date!")
        return
    
    # Limit if max_documents specified
    if max_documents and len(docs_to_process) > max_documents:
        docs_to_process = docs_to_process[:max_documents]
        print(f"⚠ Limited to {max_documents} documents as requested")
    
    print(f"✓ Found {len(docs_to_process)} documents to process")
    
    # Generate embeddings
    print(f"\n[4/5] Generating embeddings...")
    docs_with_embeddings = []
    
    for i, doc in enumerate(docs_to_process, 1):
        try:
            # Generate embedding for document content
            embedding = await embedder.embed_text(doc["content"])
            
            docs_with_embeddings.append({
                "id": doc["id"],
                "embedding": embedding
            })
            
            # Progress indicator
            if i % 10 == 0 or i == len(docs_to_process):
                print(f"  Progress: {i}/{len(docs_to_process)} documents processed")
        
        except Exception as e:
            print(f"  ✗ Error generating embedding for document {doc['id'][:50]}...: {e}")
            continue
    
    print(f"✓ Successfully generated {len(docs_with_embeddings)} embeddings")
    
    # Update documents in batches
    print(f"\n[5/5] Updating documents with embeddings...")
    
    # Process in smaller batches for the update
    update_batch_size = 50
    total_updated = 0
    total_failed = 0
    
    for i in range(0, len(docs_with_embeddings), update_batch_size):
        batch = docs_with_embeddings[i:i + update_batch_size]
        
        try:
            success = store.batch_update_embeddings(batch)
            if success:
                total_updated += len(batch)
                print(f"  ✓ Updated batch {i//update_batch_size + 1} ({len(batch)} documents)")
            else:
                total_failed += len(batch)
                print(f"  ✗ Failed to update batch {i//update_batch_size + 1}")
        except Exception as e:
            total_failed += len(batch)
            print(f"  ✗ Error updating batch: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Documents processed: {len(docs_to_process)}")
    print(f"Embeddings generated: {len(docs_with_embeddings)}")
    print(f"Documents updated: {total_updated}")
    print(f"Documents failed: {total_failed}")
    
    if total_updated > 0:
        print("\n✓ Embedding generation complete!")
        print("\nNext steps:")
        print("  1. Wait a few seconds for index to update")
        print("  2. Test vector search with: python test_vector_search.py")
    else:
        print("\n✗ No documents were updated. Please check the errors above.")

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for indexed documents")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of documents to fetch per batch (default: 100)"
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Maximum number of documents to process (default: all)"
    )
    
    args = parser.parse_args()
    
    await generate_embeddings_for_documents(
        batch_size=args.batch_size,
        max_documents=args.max_documents
    )

if __name__ == "__main__":
    asyncio.run(main())
