import asyncio
from dotenv import load_dotenv
from app.adapters.azure.ai_search import AzureAISearchStore

load_dotenv()

async def main():
    store = AzureAISearchStore()
    
    print("Testing TEXT search (not vector search)...")
    results = await store.search_text("APRA", top_k=5)
    
    print(f"\nFound {len(results)} results using text search:")
    for res in results:
        print(f"\n - ID: {res.document.id[:50]}...")
        print(f"   Score: {res.score}")
        print(f"   Content preview: {res.document.content[:200]}...")
        print(f"   Metadata: {res.document.metadata}")

if __name__ == "__main__":
    asyncio.run(main())
