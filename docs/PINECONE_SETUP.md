# Pinecone Setup and Usage Guide

This guide provides comprehensive instructions for setting up and using Pinecone as a vector database for the document search API.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Account Setup](#account-setup)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)

---

## Introduction

Pinecone is a fully managed vector database that provides fast, scalable similarity search. This integration enables semantic document search using vector embeddings.

**Key Features:**
- Serverless architecture (no infrastructure management)
- Sub-second query latency
- Metadata filtering
- Namespace support for multi-tenancy
- Automatic scaling

---

## Prerequisites

Before setting up Pinecone, ensure you have:

1. **Python 3.8+** installed
2. **Project dependencies** installed: `pip install -r requirements.txt`
3. **Embedding service** configured (OpenAI, Azure OpenAI, or custom)
4. **Pinecone account** (see Account Setup below)

---

## Account Setup

### Step 1: Create Pinecone Account

1. Visit [https://www.pinecone.io/](https://www.pinecone.io/)
2. Click **"Start Free"** or **"Sign Up"**
3. Sign up using email, Google, or GitHub
4. Verify your email address

### Step 2: Create a Project

1. After logging in, create your first project
2. Enter a project name (e.g., `zz-docs-ai`)
3. Select cloud provider and region:
   - **AWS**: `us-east-1`, `us-west-2`, `eu-west-1`
   - **GCP**: `us-central1`, `us-west1`
   - **Azure**: `eastus`, `westus2`
   
   💡 **Tip**: Choose a region close to your application for lower latency

### Step 3: Get API Key

1. Navigate to **API Keys** in the left sidebar
2. Click **"Create API Key"**
3. Name your key (e.g., `zz-docs-ai-dev`)
4. **Copy the API key immediately** (it won't be shown again)
5. Store it securely

### Step 4: Note Your Environment

Your Pinecone environment is typically shown in:
- Console URL (e.g., `us-east-1-aws`)
- Project settings

Common environments:
- `us-east-1-aws`
- `us-west-2-aws`
- `gcp-starter`
- `eu-west1-gcp`

---

## Configuration

### Environment Variables

Add the following to your `.env` file:

```bash
# Pinecone Configuration
PINECONE_API_KEY=your-api-key-here
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=zz-docs-ai-index
PINECONE_DIMENSION=1536
PINECONE_METRIC=cosine
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

### Configuration Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `PINECONE_API_KEY` | Your Pinecone API key | (required) | - |
| `PINECONE_ENVIRONMENT` | Pinecone environment | `us-east-1-aws` | See console |
| `PINECONE_INDEX_NAME` | Name for your index | `zz-docs-ai-index` | Any valid name |
| `PINECONE_DIMENSION` | Vector dimension | `1536` | Must match embedding model |
| `PINECONE_METRIC` | Similarity metric | `cosine` | `cosine`, `euclidean`, `dotproduct` |
| `PINECONE_CLOUD` | Cloud provider | `aws` | `aws`, `gcp`, `azure` |
| `PINECONE_REGION` | Cloud region | `us-east-1` | See cloud provider docs |

⚠️ **Important**: `PINECONE_DIMENSION` must match your embedding model:
- OpenAI `text-embedding-ada-002`: 1536
- OpenAI `text-embedding-3-small`: 1536
- OpenAI `text-embedding-3-large`: 3072

---

## Usage Examples

### Basic Usage with CLI

```bash
# Upload a document
python -m app.cli upload-document --file document.txt --adapter pinecone

# Search for documents
python -m app.cli search --query "machine learning" --adapter pinecone
```

### Python API Usage

#### Initialize Pinecone Store

```python
from app.adapters.pinecone.store import PineconeStore

# Initialize
store = PineconeStore()

# Ensure index exists (creates if needed)
store.ensure_index()
```

#### Upload Documents

```python
from app.core.models import Document

# Create documents with embeddings
documents = [
    Document(
        id="doc-1",
        content="Machine learning is a subset of AI.",
        metadata={"category": "AI", "author": "John"},
        embedding=[0.1, 0.2, ...]  # Your embedding vector
    ),
    Document(
        id="doc-2",
        content="Python is great for data science.",
        metadata={"category": "Programming", "author": "Jane"},
        embedding=[0.3, 0.4, ...]
    )
]

# Upload to Pinecone
await store.add_documents(documents)
```

#### Search Documents

```python
# Generate query embedding (using your embedding service)
query_embedding = [0.15, 0.25, ...]  # Your query vector

# Search
results = await store.search(
    query_embedding=query_embedding,
    top_k=5
)

# Process results
for result in results:
    print(f"Score: {result.score}")
    print(f"Content: {result.document.content}")
    print(f"Metadata: {result.document.metadata}")
```

#### Search with Metadata Filters

```python
# Filter by category
filters = {"category": {"$eq": "AI"}}

results = await store.search(
    query_embedding=query_embedding,
    top_k=5,
    filters=filters
)
```

**Filter Operators:**
- `$eq`: Equal to
- `$ne`: Not equal to
- `$gt`: Greater than
- `$gte`: Greater than or equal to
- `$lt`: Less than
- `$lte`: Less than or equal to
- `$in`: In array
- `$nin`: Not in array

#### Use Namespaces

```python
# Upload to namespace
await store.add_documents(documents, namespace="user-123")

# Search within namespace
results = await store.search(
    query_embedding=query_embedding,
    top_k=5,
    namespace="user-123"
)

# List all namespaces
namespaces = store.list_namespaces()
```

#### Delete Documents

```python
# Delete by ID
await store.delete("doc-1")

# Delete from specific namespace
await store.delete("doc-1", namespace="user-123")
```

#### Update Metadata

```python
# Update document metadata
store.update_metadata(
    document_id="doc-1",
    metadata={
        "content": "Updated content...",
        "category": "AI",
        "updated": True
    }
)
```

#### Index Management

```python
# Describe index
info = store.describe_index()
print(f"Vectors: {info['total_vector_count']}")
print(f"Dimension: {info['dimension']}")
print(f"Metric: {info['metric']}")

# Delete index (careful!)
store.delete_index()

# Create new index
store.create_index(dimension=1536, metric="cosine")
```

---

## Troubleshooting

### Common Issues

#### 1. "PINECONE_API_KEY is not configured"

**Solution**: Add your API key to `.env`:
```bash
PINECONE_API_KEY=your-actual-api-key
```

#### 2. "Index does not exist"

**Solution**: Create the index:
```python
store = PineconeStore()
store.ensure_index()
```

Or manually create it in the Pinecone console.

#### 3. "Dimension mismatch"

**Error**: Vector dimension doesn't match index dimension.

**Solution**: Ensure `PINECONE_DIMENSION` matches your embedding model. If you need to change it, you must delete and recreate the index.

#### 4. "Upsert failed: metadata too large"

**Error**: Metadata exceeds Pinecone's 40KB limit per vector.

**Solution**: The adapter automatically truncates content to 1000 characters. If you have large custom metadata, reduce its size.

#### 5. Slow search performance

**Possible causes**:
- Index not fully initialized (wait a few seconds after creation)
- Large number of vectors (consider upgrading plan)
- Network latency (choose closer region)

**Solution**: 
- Wait for index to be ready
- Check index stats: `store.describe_index()`
- Consider upgrading to a paid plan for better performance

#### 6. Rate limiting errors

**Error**: Too many requests.

**Solution**: 
- Free tier has rate limits
- Implement retry logic with exponential backoff
- Upgrade to paid plan for higher limits

---

## Best Practices

### 1. Index Configuration

- **Choose the right metric**:
  - `cosine`: Best for normalized embeddings (most common)
  - `euclidean`: For absolute distance
  - `dotproduct`: For non-normalized vectors

- **Set correct dimension**: Must match your embedding model exactly

### 2. Metadata Design

- Keep metadata small (< 40KB per vector)
- Use simple types: strings, numbers, booleans
- Index frequently filtered fields
- Avoid deeply nested objects

### 3. Batch Operations

- Upload in batches of 100 vectors for efficiency
- The adapter handles this automatically
- For large uploads, monitor progress

### 4. Namespaces

- Use namespaces for multi-tenancy (e.g., per-user data)
- Namespaces are isolated (searches don't cross boundaries)
- List namespaces to manage them: `store.list_namespaces()`

### 5. Cost Optimization

- **Free tier**: 1 index, up to 100K vectors
- Delete unused indexes to save costs
- Monitor vector count: `store.describe_index()`
- Consider serverless for variable workloads

### 6. Performance

- Choose a region close to your application
- Use metadata filters to reduce search space
- Adjust `top_k` based on your needs (lower = faster)
- Cache frequent queries if possible

### 7. Error Handling

- Always wrap operations in try-except blocks
- Check return values (most methods return bool for success)
- Log errors for debugging
- Implement retry logic for transient failures

---

## API Reference

### PineconeStore Methods

#### `__init__()`
Initialize Pinecone client and connect to index.

#### `ensure_index() -> bool`
Ensure index exists, creating if necessary.

#### `create_index(dimension: int = None, metric: str = None) -> bool`
Create a new index with specified configuration.

#### `delete_index() -> bool`
Delete the configured index.

#### `describe_index() -> dict`
Get index statistics and configuration.

#### `add_documents(documents: List[Document], namespace: str = "") -> bool`
Upload documents with embeddings to Pinecone.

#### `search(query_embedding: List[float], top_k: int = 5, filters: dict = None, namespace: str = "") -> List[SearchResult]`
Search for similar documents using vector embedding.

#### `delete(document_id: str, namespace: str = "") -> bool`
Delete a document by ID.

#### `batch_upsert(vectors: List[dict], namespace: str = "") -> bool`
Batch upsert vectors for efficiency.

#### `update_metadata(document_id: str, metadata: dict, namespace: str = "") -> bool`
Update metadata for an existing vector.

#### `fetch_by_ids(ids: List[str], namespace: str = "") -> List[Document]`
Fetch documents by their IDs.

#### `list_namespaces() -> List[str]`
List all namespaces in the index.

---

## External Resources

- [Pinecone Documentation](https://docs.pinecone.io/)
- [Pinecone Python SDK](https://github.com/pinecone-io/pinecone-python-client)
- [Pinecone API Reference](https://docs.pinecone.io/reference/api/introduction)
- [Pinecone Pricing](https://www.pinecone.io/pricing/)
- [Pinecone Console](https://app.pinecone.io/)

---

## Support

For issues specific to this integration:
1. Check the troubleshooting section above
2. Run the verification script: `python verify_pinecone.py`
3. Review logs for error details

For Pinecone-specific issues:
- [Pinecone Support](https://support.pinecone.io/)
- [Pinecone Community](https://community.pinecone.io/)
