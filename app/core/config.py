from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    API_TITLE: str = "Flexible Document Search API"
    API_VERSION: str = "v1"
    
    # Options: memory, azure_search, pinecone, milvus
    VECTOR_STORE_TYPE: str = "memory" 
    
    # Options: random (for testing), openai, azure_openai
    EMBEDDING_MODEL_TYPE: str = "random" 

    # Azure AI Search Config
    AZURE_SEARCH_ENDPOINT: Optional[str] = None
    AZURE_SEARCH_KEY: Optional[str] = None
    AZURE_SEARCH_INDEX_NAME: str = "documents"
    
    # Azure Storage Config
    AZURE_STORAGE_CONNECTION_STRING: Optional[str] = None

    # OpenAI / Azure OpenAI Config
    OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_DEPLOYMENT_NAME: Optional[str] = None

    # Pinecone Config
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: str = "us-east-1-aws"
    PINECONE_INDEX_NAME: str = "zz-docs-ai-index"
    PINECONE_DIMENSION: int = 1536
    PINECONE_METRIC: str = "cosine"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    class Config:
        env_file = ".env"

settings = Settings()
