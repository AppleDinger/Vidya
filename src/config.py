"""
Configuration Module for Vidya RAG System.

Centralizes environment variables, model identifiers, chunking parameters,
and directory paths for reproducible and maintainable execution.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
UPLOADS_DIR = DATA_DIR / "uploads"

# Ensure data directories exist
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# Load environment variables from .env file
load_dotenv(dotenv_path=BASE_DIR / ".env")

# API Keys & Secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# LLM & Embedding Settings
# gpt-4o-mini offers high factual accuracy, fast generation, and low inference cost
DEFAULT_LLM_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
# text-embedding-3-small offers top-tier retrieval performance at fractional cost
DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_TEMPERATURE = 0.0

# Chunking & Token Budgeting Strategy
# - chunk_size=1000 characters (~200-250 tokens): Captures full semantic thoughts/paragraphs
# - chunk_overlap=200 characters (~40-50 tokens): Preserves contextual continuity across split boundaries
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Retrieval Parameters
DEFAULT_RETRIEVAL_K = 4

def is_api_key_configured() -> bool:
    """Check if a valid OpenAI API key is present in environment or config."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(key and not key.startswith("your_openai_api_key"))
