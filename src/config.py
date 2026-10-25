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
# Prioritize GEMINI_API_KEY, with fallback to GOOGLE_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")


# LLM & Embedding Settings
# gemini-1.5-pro offers superior reasoning, long-context grounding, and zero-hallucination discipline
DEFAULT_LLM_MODEL = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro")
# models/text-embedding-004 delivers 768-dimensional state-of-the-art MTEB retrieval performance
DEFAULT_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
DEFAULT_TEMPERATURE = 0.0

# Chunking & Token Budgeting Strategy
# - chunk_size=1000 characters (~200-250 tokens): Captures full semantic thoughts/paragraphs
# - chunk_overlap=200 characters (~40-50 tokens): Preserves contextual continuity across split boundaries
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Retrieval Parameters
DEFAULT_RETRIEVAL_K = 4


def get_active_api_key() -> str:
    """Returns the active Gemini or Google API key from environment."""
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "") or GEMINI_API_KEY).strip()


def is_api_key_configured() -> bool:
    """Check if a valid Gemini/Google API key is present in environment or config."""
    key = get_active_api_key()
    return bool(key and not key.startswith("your_"))

