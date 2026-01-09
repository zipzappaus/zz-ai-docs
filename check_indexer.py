import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents.indexes import SearchIndexerClient
from app.core.config import settings

load_dotenv()

def get_credential():
    if settings.AZURE_SEARCH_KEY:
        return AzureKeyCredential(settings.AZURE_SEARCH_KEY)
    return DefaultAzureCredential()

indexer_client = SearchIndexerClient(
    endpoint=settings.AZURE_SEARCH_ENDPOINT,
    credential=get_credential()
)

# Check indexer status
indexer_name = "blob-indexer"
try:
    status = indexer_client.get_indexer_status(indexer_name)
    print(f"Indexer: {indexer_name}")
    print(f"Status: {status.status}")
    print(f"Last result: {status.last_result}")
    if status.last_result:
        print(f"  - Status: {status.last_result.status}")
        print(f"  - Error message: {status.last_result.error_message}")
        print(f"  - Items processed: {status.last_result.items_processed}")
        print(f"  - Items failed: {status.last_result.items_failed}")
    
    print("\nExecution history:")
    for i, execution in enumerate(status.execution_history[:3]):
        print(f"\nExecution {i+1}:")
        print(f"  - Status: {execution.status}")
        print(f"  - Error message: {execution.error_message}")
        print(f"  - Items processed: {execution.items_processed}")
        print(f"  - Items failed: {execution.items_failed}")
        if execution.errors:
            print(f"  - Errors: {execution.errors}")
except Exception as e:
    print(f"Error getting indexer status: {e}")
