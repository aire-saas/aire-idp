"""
Configuration settings for the Hierarchical RAG system.
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

# OpenAI API Configuration
# Set your API key as an environment variable: OPENAI_API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model Configuration
LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

# ChromaDB Configuration
COLLECTION_NAME = "floor_plans_v2"
DEFAULT_PERSIST_DIR = "vector_store"
