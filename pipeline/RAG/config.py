"""
Configuration settings for the Hierarchical RAG system using Azure OpenAI.
"""
import os
from pathlib import Path

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, use environment variables only

# Azure OpenAI API Configuration
# Set these environment variables in your .env file:
# AZURE_OPENAI_API_KEY
# AZURE_OPENAI_ENDPOINT
# AZURE_OPENAI_API_VERSION
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://eastus2aiforhana.openai.azure.com/")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Model Configuration
LLM_DEPLOYMENT_NAME = "gpt-4o"  # Azure deployment name instead of model
EMBEDDING_DEPLOYMENT_NAME = "text-embedding-3-small"  # Azure deployment name for embeddings

# ChromaDB Configuration
COLLECTION_NAME = "floor_plans_v2"
DEFAULT_PERSIST_DIR = "vector_store"
