import asyncio
from dotenv import load_dotenv
from app.adapters.azure.ai_search import AzureAISearchStore

load_dotenv()

async def main():
    store = AzureAISearchStore()
    
    # Get document count
    try:
        results = store.search_client.search(
            search_text="*",
            include_total_count=True,
            top=10,
            select=["id", "content", "metadata"]
        )
        
        print(f"Total documents in index: {results.get_count()}")
        print("\nFirst 10 documents:")
        for i, doc in enumerate(results):
            print(f"\n{i+1}. ID: {doc['id']}")
            print(f"   Content preview: {doc['content'][:200]}...")
            print(f"   Metadata: {doc.get('metadata', 'N/A')}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
