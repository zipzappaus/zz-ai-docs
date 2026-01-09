"""
Pinecone Integration Verification Script

This script tests all Pinecone functionality including:
- Connection and authentication
- Index management (create, describe, delete)
- Document upload and vectorization
- Vector search with various parameters
- Metadata filtering
- Batch operations
- Namespace support
- Document deletion and updates
"""

import asyncio
import sys
from typing import List
from datetime import datetime

# Add app to path
sys.path.insert(0, ".")

from app.adapters.pinecone.store import PineconeStore
from app.core.models import Document
from app.core.config import settings


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_test(name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "✓" if passed else "✗"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status}{reset} {name}")
    if details:
        print(f"  {details}")


def generate_mock_embedding(dimension: int = 1536) -> List[float]:
    """Generate a mock embedding vector."""
    import random
    random.seed(42)
    return [random.random() for _ in range(dimension)]


async def test_connection():
    """Test 1: Connection and Authentication"""
    print_header("Test 1: Connection and Authentication")
    
    try:
        # Check if API key is configured
        if not settings.PINECONE_API_KEY:
            print_test("API Key Configuration", False, "PINECONE_API_KEY not set in .env")
            return False
        
        print_test("API Key Configuration", True, f"API key found (length: {len(settings.PINECONE_API_KEY)})")
        
        # Initialize Pinecone store
        store = PineconeStore()
        print_test("Pinecone Client Initialization", True, f"Connected to environment: {settings.PINECONE_ENVIRONMENT}")
        
        return store
        
    except Exception as e:
        print_test("Pinecone Client Initialization", False, f"Error: {e}")
        return None


async def test_index_management(store: PineconeStore):
    """Test 2: Index Management"""
    print_header("Test 2: Index Management")
    
    try:
        # Ensure index exists
        success = store.ensure_index()
        print_test("Ensure Index Exists", success, f"Index name: {store.index_name}")
        
        if not success:
            return False
        
        # Describe index
        index_info = store.describe_index()
        if index_info:
            print_test("Describe Index", True, 
                      f"Dimension: {index_info.get('dimension')}, "
                      f"Metric: {index_info.get('metric')}, "
                      f"Vectors: {index_info.get('total_vector_count', 0)}")
        else:
            print_test("Describe Index", False, "Failed to get index info")
            return False
        
        return True
        
    except Exception as e:
        print_test("Index Management", False, f"Error: {e}")
        return False


async def test_document_upload(store: PineconeStore):
    """Test 3: Document Upload"""
    print_header("Test 3: Document Upload")
    
    try:
        # Create test documents with embeddings
        test_docs = [
            Document(
                id="test-doc-1",
                content="This is a test document about machine learning and artificial intelligence.",
                metadata={"category": "AI", "source": "test"},
                embedding=generate_mock_embedding(settings.PINECONE_DIMENSION)
            ),
            Document(
                id="test-doc-2",
                content="Python is a popular programming language for data science and web development.",
                metadata={"category": "Programming", "source": "test"},
                embedding=generate_mock_embedding(settings.PINECONE_DIMENSION)
            ),
            Document(
                id="test-doc-3",
                content="Vector databases enable semantic search and similarity matching.",
                metadata={"category": "Database", "source": "test"},
                embedding=generate_mock_embedding(settings.PINECONE_DIMENSION)
            ),
        ]
        
        # Upload documents
        success = await store.add_documents(test_docs)
        print_test("Upload Documents", success, f"Uploaded {len(test_docs)} documents")
        
        if not success:
            return False, []
        
        # Wait for indexing
        import time
        print("  Waiting for indexing to complete...")
        time.sleep(2)
        
        # Verify upload
        index_info = store.describe_index()
        vector_count = index_info.get('total_vector_count', 0)
        print_test("Verify Upload", vector_count >= len(test_docs), 
                   f"Index now contains {vector_count} vectors")
        
        return True, test_docs
        
    except Exception as e:
        print_test("Document Upload", False, f"Error: {e}")
        return False, []


async def test_vector_search(store: PineconeStore, test_docs: List[Document]):
    """Test 4: Vector Search"""
    print_header("Test 4: Vector Search")
    
    try:
        # Use first document's embedding as query
        query_embedding = test_docs[0].embedding
        
        # Search
        results = await store.search(query_embedding, top_k=3)
        
        print_test("Vector Search", len(results) > 0, f"Found {len(results)} results")
        
        if results:
            print("\n  Top Results:")
            for i, result in enumerate(results[:3], 1):
                print(f"    {i}. Score: {result.score:.4f}")
                print(f"       Content: {result.document.content[:60]}...")
                print(f"       Metadata: {result.document.metadata}")
        
        return len(results) > 0
        
    except Exception as e:
        print_test("Vector Search", False, f"Error: {e}")
        return False


async def test_metadata_filtering(store: PineconeStore, test_docs: List[Document]):
    """Test 5: Metadata Filtering"""
    print_header("Test 5: Metadata Filtering")
    
    try:
        # Search with metadata filter
        query_embedding = generate_mock_embedding(settings.PINECONE_DIMENSION)
        
        # Filter for category = "AI"
        filters = {"category": {"$eq": "AI"}}
        results = await store.search(query_embedding, top_k=5, filters=filters)
        
        # Check if results match filter
        all_match = all(r.document.metadata.get("category") == "AI" for r in results)
        
        print_test("Metadata Filtering", all_match and len(results) > 0, 
                   f"Found {len(results)} results with category='AI'")
        
        if results:
            for result in results:
                print(f"  - {result.document.metadata.get('category')}: {result.document.content[:50]}...")
        
        return all_match
        
    except Exception as e:
        print_test("Metadata Filtering", False, f"Error: {e}")
        return False


async def test_batch_operations(store: PineconeStore):
    """Test 6: Batch Operations"""
    print_header("Test 6: Batch Operations")
    
    try:
        # Create a large batch of documents
        batch_size = 150
        batch_docs = []
        
        for i in range(batch_size):
            doc = Document(
                id=f"batch-doc-{i}",
                content=f"Batch document number {i} for testing large uploads.",
                metadata={"batch": True, "index": i},
                embedding=generate_mock_embedding(settings.PINECONE_DIMENSION)
            )
            batch_docs.append(doc)
        
        # Upload batch
        success = await store.add_documents(batch_docs)
        print_test("Batch Upload", success, f"Uploaded {batch_size} documents in batches")
        
        # Wait for indexing
        import time
        time.sleep(3)
        
        # Verify
        index_info = store.describe_index()
        vector_count = index_info.get('total_vector_count', 0)
        print_test("Verify Batch Upload", vector_count >= batch_size, 
                   f"Index now contains {vector_count} vectors")
        
        return success
        
    except Exception as e:
        print_test("Batch Operations", False, f"Error: {e}")
        return False


async def test_namespace_support(store: PineconeStore):
    """Test 7: Namespace Support"""
    print_header("Test 7: Namespace Support")
    
    try:
        # Upload to different namespaces
        namespace1 = "test-namespace-1"
        namespace2 = "test-namespace-2"
        
        doc1 = Document(
            id="ns1-doc",
            content="Document in namespace 1",
            metadata={"namespace": namespace1},
            embedding=generate_mock_embedding(settings.PINECONE_DIMENSION)
        )
        
        doc2 = Document(
            id="ns2-doc",
            content="Document in namespace 2",
            metadata={"namespace": namespace2},
            embedding=generate_mock_embedding(settings.PINECONE_DIMENSION)
        )
        
        success1 = await store.add_documents([doc1], namespace=namespace1)
        success2 = await store.add_documents([doc2], namespace=namespace2)
        
        print_test("Upload to Namespaces", success1 and success2, 
                   f"Uploaded to {namespace1} and {namespace2}")
        
        # Wait for indexing
        import time
        time.sleep(2)
        
        # List namespaces
        namespaces = store.list_namespaces()
        has_namespaces = namespace1 in namespaces and namespace2 in namespaces
        
        print_test("List Namespaces", has_namespaces, 
                   f"Found namespaces: {', '.join(namespaces)}")
        
        # Search within namespace
        query_embedding = generate_mock_embedding(settings.PINECONE_DIMENSION)
        results = await store.search(query_embedding, top_k=5, namespace=namespace1)
        
        print_test("Search in Namespace", len(results) > 0, 
                   f"Found {len(results)} results in {namespace1}")
        
        return success1 and success2
        
    except Exception as e:
        print_test("Namespace Support", False, f"Error: {e}")
        return False


async def test_delete_operations(store: PineconeStore):
    """Test 8: Delete Operations"""
    print_header("Test 8: Delete Operations")
    
    try:
        # Create and upload a document
        doc = Document(
            id="delete-test-doc",
            content="This document will be deleted",
            metadata={"test": "delete"},
            embedding=generate_mock_embedding(settings.PINECONE_DIMENSION)
        )
        
        await store.add_documents([doc])
        
        # Wait for indexing
        import time
        time.sleep(2)
        
        # Delete the document
        success = await store.delete(doc.id)
        print_test("Delete Document", success, f"Deleted document: {doc.id}")
        
        # Wait for deletion to propagate
        time.sleep(2)
        
        # Verify deletion by searching
        results = await store.search(doc.embedding, top_k=10)
        deleted = all(r.document.id != doc.id for r in results)
        
        print_test("Verify Deletion", deleted, "Document no longer appears in search results")
        
        return success and deleted
        
    except Exception as e:
        print_test("Delete Operations", False, f"Error: {e}")
        return False


async def test_update_metadata(store: PineconeStore):
    """Test 9: Update Metadata"""
    print_header("Test 9: Update Metadata")
    
    try:
        # Create and upload a document
        doc = Document(
            id="update-test-doc",
            content="This document's metadata will be updated",
            metadata={"version": 1, "status": "draft"},
            embedding=generate_mock_embedding(settings.PINECONE_DIMENSION)
        )
        
        await store.add_documents([doc])
        
        # Wait for indexing
        import time
        time.sleep(2)
        
        # Update metadata
        new_metadata = {
            "content": doc.content[:1000],
            "created_at": doc.created_at.isoformat(),
            "version": 2,
            "status": "published"
        }
        
        success = store.update_metadata(doc.id, new_metadata)
        print_test("Update Metadata", success, "Updated document metadata")
        
        # Wait for update to propagate
        time.sleep(2)
        
        # Fetch and verify
        docs = store.fetch_by_ids([doc.id])
        if docs:
            updated_doc = docs[0]
            metadata_updated = updated_doc.metadata.get("version") == 2
            print_test("Verify Metadata Update", metadata_updated, 
                      f"New metadata: {updated_doc.metadata}")
        else:
            print_test("Verify Metadata Update", False, "Could not fetch updated document")
            return False
        
        return success
        
    except Exception as e:
        print_test("Update Metadata", False, f"Error: {e}")
        return False


async def cleanup(store: PineconeStore):
    """Cleanup: Delete test index (optional)"""
    print_header("Cleanup (Optional)")
    
    response = input("\nDo you want to delete the test index? (y/N): ")
    
    if response.lower() == 'y':
        try:
            success = store.delete_index()
            print_test("Delete Test Index", success, f"Deleted index: {store.index_name}")
        except Exception as e:
            print_test("Delete Test Index", False, f"Error: {e}")
    else:
        print("  Skipping index deletion. You can manually delete it from the Pinecone console.")


async def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("  PINECONE INTEGRATION VERIFICATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Environment: {settings.PINECONE_ENVIRONMENT}")
    print(f"  Index Name: {settings.PINECONE_INDEX_NAME}")
    print(f"  Dimension: {settings.PINECONE_DIMENSION}")
    print(f"  Metric: {settings.PINECONE_METRIC}")
    print(f"  Cloud: {settings.PINECONE_CLOUD}")
    print(f"  Region: {settings.PINECONE_REGION}")
    
    # Track test results
    results = {}
    
    # Test 1: Connection
    store = await test_connection()
    results["Connection"] = store is not None
    
    if not store:
        print("\n❌ Cannot proceed without valid connection. Please check your configuration.")
        return
    
    # Test 2: Index Management
    results["Index Management"] = await test_index_management(store)
    
    if not results["Index Management"]:
        print("\n❌ Cannot proceed without valid index.")
        return
    
    # Test 3: Document Upload
    upload_success, test_docs = await test_document_upload(store)
    results["Document Upload"] = upload_success
    
    if not upload_success:
        print("\n⚠️  Document upload failed. Some tests may be skipped.")
        test_docs = []
    
    # Test 4: Vector Search
    if test_docs:
        results["Vector Search"] = await test_vector_search(store, test_docs)
    
    # Test 5: Metadata Filtering
    if test_docs:
        results["Metadata Filtering"] = await test_metadata_filtering(store, test_docs)
    
    # Test 6: Batch Operations
    results["Batch Operations"] = await test_batch_operations(store)
    
    # Test 7: Namespace Support
    results["Namespace Support"] = await test_namespace_support(store)
    
    # Test 8: Delete Operations
    results["Delete Operations"] = await test_delete_operations(store)
    
    # Test 9: Update Metadata
    results["Update Metadata"] = await test_update_metadata(store)
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✓" if passed_test else "✗"
        color = "\033[92m" if passed_test else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{status}{reset} {test_name}")
    
    print(f"\n{'=' * 70}")
    print(f"  Results: {passed}/{total} tests passed")
    print(f"{'=' * 70}\n")
    
    if passed == total:
        print("🎉 All tests passed! Pinecone integration is working correctly.\n")
    else:
        print("⚠️  Some tests failed. Please review the errors above.\n")
    
    # Cleanup
    await cleanup(store)


if __name__ == "__main__":
    asyncio.run(main())
