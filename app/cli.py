import typer
import os
import asyncio
from typing import Optional
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from app.adapters.azure.ai_search import AzureAISearchStore
from app.core.config import settings
from app.adapters.custom.random_embedding import RandomEmbeddingModel

# Load environment variables
load_dotenv()

app = typer.Typer(help="CLI for Azure AI Search operations")

def get_blob_service_client():
    connection_string = settings.AZURE_STORAGE_CONNECTION_STRING
    if not connection_string:
        typer.echo("Error: AZURE_STORAGE_CONNECTION_STRING not set.", err=True)
        raise typer.Exit(code=1)
    return BlobServiceClient.from_connection_string(connection_string)

def get_search_store():
    try:
        return AzureAISearchStore()
    except Exception as e:
        typer.echo(f"Error initializing AzureAISearchStore: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def upload_file(file_path: str, container_name: str = "documents"):
    """
    Upload a file to Azure Blob Storage.
    """
    if not os.path.exists(file_path):
        typer.echo(f"Error: File '{file_path}' does not exist.", err=True)
        raise typer.Exit(code=1)

    blob_name = os.path.basename(file_path)
    typer.echo(f"Uploading '{blob_name}' to container '{container_name}'...")

    try:
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(container_name)
        
        if not container_client.exists():
            container_client.create_container()
            typer.echo(f"Created container '{container_name}'.")

        blob_client = container_client.get_blob_client(blob_name)
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        typer.echo("Upload successful.")
    except Exception as e:
        typer.echo(f"Upload failed: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def create_index(index_name: Optional[str] = None):
    """
    Create or update the Azure AI Search Index.
    """
    # Note: index_name is usually pulled from settings in the store, 
    # but could be overridden if the store supports it (current store impl might use settings directly)
    typer.echo("Creating/Updating Index...")
    try:
        # store.create_index() is synchronous and uses create_or_update_index logic
        store = get_search_store()
        store.create_index() 
        typer.echo("Index operation complete.")
        
    except Exception as e:
        typer.echo(f"Index creation failed: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def delete_index():
    """
    Delete the Azure AI Search Index.
    """
    typer.echo("Deleting Index...")
    try:
        store = get_search_store()
        store.delete_index()
        typer.echo("Index deleted.")
    except Exception as e:
        typer.echo(f"Index deletion failed: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def create_indexer(indexer_name: str = "blob-indexer", data_source_name: str = "blob-datasource", container_name: str = "documents"):
    """
    Create Data Source and Indexer.
    """
    typer.echo(f"Creating Data Source '{data_source_name}' and Indexer '{indexer_name}'...")
    try:
        store = get_search_store() # This is sync based on verify_azure
        connection_string = settings.AZURE_STORAGE_CONNECTION_STRING
        if not connection_string:
             typer.echo("Error: AZURE_STORAGE_CONNECTION_STRING not set.", err=True)
             raise typer.Exit(code=1)

        # verify_azure.py calls are sync for create_blob_data_source ?
        # verify_azure.py: 
        # store.create_blob_data_source(...) -> looked sync in verify script (no await)
        # store.create_or_update_indexer(...) -> looked sync
        # store.run_indexer(...) -> looked sync
        
        store.create_blob_data_source(data_source_name, connection_string, container_name)
        typer.echo(f"Data Source '{data_source_name}' created.")
        
        store.create_or_update_indexer(indexer_name, data_source_name)
        typer.echo(f"Indexer '{indexer_name}' created.")
        
        store.run_indexer(indexer_name)
        typer.echo(f"Indexer '{indexer_name}' run triggered.")
        
    except Exception as e:
        typer.echo(f"Indexer operation failed: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def search(query: str, top_k: int = 5):
    """
    Search for documents.
    """
    typer.echo(f"Searching for: '{query}'")
    try:
        async def _run():
            store = get_search_store()
            embedder = RandomEmbeddingModel() # Or use real one if available
            # In verify_azure it used RandomEmbeddingModel
            embedding = await embedder.embed_text(query)
            results = await store.search(embedding, top_k=top_k)
            
            typer.echo(f"Found {len(results)} results.")
            for res in results:
                typer.echo(f" - {res.score:.4f}: {res.document.content[:100]}... (ID: {res.document.id})")
        
        asyncio.run(_run())
    except Exception as e:
        typer.echo(f"Search failed: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def generate_embeddings(batch_size: int = 100, max_documents: Optional[int] = None):
    """
    Generate embeddings for documents that don't have them.
    
    Args:
        batch_size: Number of documents to fetch per batch (default: 100)
        max_documents: Maximum number of documents to process (default: all)
    """
    typer.echo("=" * 60)
    typer.echo("Embedding Generation Process")
    typer.echo("=" * 60)
    
    async def _run():
        # Initialize store and embedding model
        typer.echo("\n[1/5] Initializing Azure AI Search store...")
        try:
            store = get_search_store()
            typer.echo(f"✓ Connected to {settings.AZURE_SEARCH_ENDPOINT}")
            typer.echo(f"✓ Using index: {settings.AZURE_SEARCH_INDEX_NAME}")
        except Exception as e:
            typer.echo(f"✗ Failed to initialize store: {e}", err=True)
            raise typer.Exit(code=1)
        
        typer.echo("\n[2/5] Initializing embedding model...")
        embedder = RandomEmbeddingModel()
        typer.echo("✓ Using RandomEmbeddingModel (1536 dimensions)")
        
        # Fetch documents without embeddings
        typer.echo(f"\n[3/5] Fetching documents without embeddings (batch size: {batch_size})...")
        docs_to_process = store.get_documents_without_embeddings(batch_size=batch_size)
        
        if not docs_to_process:
            typer.echo("✓ No documents found without embeddings. All documents are up to date!")
            return
        
        # Limit if max_documents specified
        if max_documents and len(docs_to_process) > max_documents:
            docs_to_process = docs_to_process[:max_documents]
            typer.echo(f"⚠ Limited to {max_documents} documents as requested")
        
        typer.echo(f"✓ Found {len(docs_to_process)} documents to process")
        
        # Generate embeddings
        typer.echo(f"\n[4/5] Generating embeddings...")
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
                    typer.echo(f"  Progress: {i}/{len(docs_to_process)} documents processed")
            
            except Exception as e:
                typer.echo(f"  ✗ Error generating embedding for document {doc['id'][:50]}...: {e}")
                continue
        
        typer.echo(f"✓ Successfully generated {len(docs_with_embeddings)} embeddings")
        
        # Update documents in batches
        typer.echo(f"\n[5/5] Updating documents with embeddings...")
        
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
                    typer.echo(f"  ✓ Updated batch {i//update_batch_size + 1} ({len(batch)} documents)")
                else:
                    total_failed += len(batch)
                    typer.echo(f"  ✗ Failed to update batch {i//update_batch_size + 1}")
            except Exception as e:
                total_failed += len(batch)
                typer.echo(f"  ✗ Error updating batch: {e}")
        
        # Summary
        typer.echo("\n" + "=" * 60)
        typer.echo("Summary")
        typer.echo("=" * 60)
        typer.echo(f"Documents processed: {len(docs_to_process)}")
        typer.echo(f"Embeddings generated: {len(docs_with_embeddings)}")
        typer.echo(f"Documents updated: {total_updated}")
        typer.echo(f"Documents failed: {total_failed}")
        
        if total_updated > 0:
            typer.echo("\n✓ Embedding generation complete!")
            typer.echo("\nNext steps:")
            typer.echo("  1. Wait a few seconds for index to update")
            typer.echo("  2. Test vector search with: python -m app.cli search \"your query\"")
        else:
            typer.echo("\n✗ No documents were updated. Please check the errors above.")
    
    try:
        asyncio.run(_run())
    except Exception as e:
        typer.echo(f"Embedding generation failed: {e}", err=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
