import asyncio
from dotenv import load_dotenv
from app.adapters.azure.ai_search import AzureAISearchStore
from app.adapters.custom.random_embedding import RandomEmbeddingModel

load_dotenv()

async def main():
    print("=" * 60)
    print("Vector Search Test")
    print("=" * 60)
    
    store = AzureAISearchStore()
    embedder = RandomEmbeddingModel()
    
    # Test query
    query = "APRA prudential"
    print(f"\nQuery: '{query}'")
    
    # Generate query embedding
    print("\nGenerating query embedding...")
    query_embedding = await embedder.embed_text(query)
    print(f"✓ Generated embedding with {len(query_embedding)} dimensions")
    
    # Perform vector search
    print("\nPerforming vector search...")
    results = await store.search(query_embedding, top_k=5)
    
    print(f"\n{'=' * 60}")
    print(f"Results: Found {len(results)} documents")
    print(f"{'=' * 60}")
    
    if results:
        for i, res in enumerate(results, 1):
            print(f"\n{i}. Score: {res.score:.4f}")
            print(f"   ID: {res.document.id[:80]}...")
            print(f"   Content preview: {res.document.content[:200]}...")
            print(f"   Metadata: {res.document.metadata}")
    else:
        print("\n⚠ No results found!")
        print("\nPossible reasons:")
        print("  1. Documents don't have embeddings yet")
        print("  2. Run: python generate_embeddings.py")
        print("  3. Or: python -m app.cli generate-embeddings")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
