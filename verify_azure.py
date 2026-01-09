import asyncio
import os
import sys
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Add project root to sys.path
sys.path.append(os.getcwd())

from app.core.models import Document
from app.adapters.azure.ai_search import AzureAISearchStore
from app.adapters.custom.random_embedding import RandomEmbeddingModel
from app.core.config import settings

async def main():
    print("Starting Azure Verification...")
    
    # Needs valid env vars: AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY (or CLI auth)
    # and AZURE_SEARCH_INDEX_NAME
    if not os.getenv("AZURE_SEARCH_ENDPOINT"):
        print("Error: AZURE_SEARCH_ENDPOINT not set.")
        return

    print(f"Connecting to {os.getenv('AZURE_SEARCH_ENDPOINT')}...")
    try:
        store = AzureAISearchStore()
    except Exception as e:
        print(f"Failed to initialize store: {e}")
        return

    # Use random embeddings for plumbing test
    embedder = RandomEmbeddingModel()

    # 1. Create Document
    print("Creating test document...")
    doc_content = "Azure AI Search is powerful."
    embedding = await embedder.embed_text(doc_content)
    doc = Document(
        content=doc_content,
        metadata={"source": "verify_script"},
        embedding=embedding
    )
    print(f"Doc ID: {doc.id}")

    # 2. Upload
    print("Uploading document...")
    success = await store.add_documents([doc])
    if success:
        print("Upload successful.")
    else:
        print("Upload failed.")
        return

    # Wait for indexing (Azure Search creates near-real-time latency)
    print("Waiting 2 seconds for indexing...")
    await asyncio.sleep(2)

    # 3. Search
    print("Searching...")
    query_embedding = await embedder.embed_text("Azure functionality") # Random vector
    results = await store.search(query_embedding, top_k=5)
    
    print(f"Found {len(results)} results.")
    for res in results:
        print(f" - {res.document.id} (Score: {res.score}): {res.document.content}")

    # 4. Clean up (Delete)
    # print("Deleting document...")
    # deleted = await store.delete(doc.id)
    # if deleted:
    #     print("Delete successful.")
    # else:
    #     print("Delete failed.")

    # ... (Previous Search Logic)
    
    # 5. Indexer Flow verification
    print("\n--- Starting Indexer Verification ---")
    storage_connection = settings.AZURE_STORAGE_CONNECTION_STRING
    if not storage_connection:
        print("Skipping Indexer verification: AZURE_STORAGE_CONNECTION_STRING not set.")
    else:
        # Create Container and Upload dummy file
        from azure.storage.blob import BlobServiceClient
        file_path = r"C:\Data\APRA\hpg_520_fit_and_proper _2026-01-09.csv"
        container_name = "documents"
        blob_name = os.path.basename(file_path)
        print(f"Uploading blob '{blob_name}' to container '{container_name}'...")
        try:
            blob_service_client = BlobServiceClient.from_connection_string(storage_connection)
            container_client = blob_service_client.get_container_client(container_name)
            if not container_client.exists():
                container_client.create_container()
            
            blob_client = container_client.get_blob_client(blob_name)
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            print("Blob uploaded.")
        except Exception as e:
            print(f"Failed to upload blob: {e}")

        # Create Data Source
        ds_name = "blob-datasource"
        print(f"Creating Data Source '{ds_name}'...")
        try:
            store.create_blob_data_source(ds_name, storage_connection, container_name)
            print("Data Source created.")
        except Exception as e:
            print(f"Failed to create Data Source: {e}")

        # Create Indexer
        indexer_name = "blob-indexer"
        print(f"Creating Indexer '{indexer_name}'...")
        try:
            store.create_or_update_indexer(indexer_name, ds_name)
            print("Indexer created.")
        except Exception as e:
            print(f"Failed to create Indexer: {e}")
        
        # Run Indexer
        print(f"Running Indexer '{indexer_name}'...")
        try:
            store.run_indexer(indexer_name)
            print("Indexer run triggered successfully.")
        except Exception as e:
            print(f"Failed to run Indexer: {e}")

    print("Azure Verification Complete.")

if __name__ == "__main__":
    asyncio.run(main())
