"""
Ingestion & Vector Storage Pipeline for Vidya.

Handles PDF loading, token-aware document chunking using RecursiveCharacterTextSplitter,
embedding generation with Google Gemini text-embedding-004, and persistent ChromaDB indexing.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from src.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_EMBEDDING_MODEL,
    CHROMA_DIR,
    GEMINI_API_KEY,
)



def load_pdf(file_path: str | Path) -> List[Document]:
    """
    Extracts text and metadata from a PDF file using PyPDFLoader.

    Args:
        file_path: Filesystem path to the PDF document.

    Returns:
        List of Document objects with normalized 1-indexed page numbers.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"PDF file not found at: {file_path}")

    loader = PyPDFLoader(str(path_obj))
    raw_docs = loader.load()

    # Normalize metadata:
    # 1. Provide 1-indexed page numbering for intuitive citations (PyPDF uses 0-indexed).
    # 2. Extract clean file basename rather than full absolute path.
    for doc in raw_docs:
        page_num = doc.metadata.get("page", 0)
        doc.metadata["page"] = page_num + 1  # 1-indexed for human readability
        doc.metadata["file_name"] = path_obj.name
        doc.metadata["source"] = path_obj.name

    return raw_docs


def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Splits loaded documents into semantically coherent chunks using RecursiveCharacterTextSplitter.

    WHY RECURSIVE CHARACTER TEXT SPLITTER?
    --------------------------------------
    1. Preserves Semantic Hierarchy: It recursively tries splitting along paragraphs ("\\n\\n"),
       then sentences ("\\n"), then words (" "), and finally characters (""). This prevents cutting
       sentences or arguments in half.
    2. Token Budgeting: 1000 characters corresponds to ~200-250 English tokens, which is the sweet spot
       for embedding density (neither too sparse nor losing granular details).
    3. Overlap Safeguard: 200 characters (~40-50 tokens) ensures that concepts crossing boundary
       points are retained in both adjacent vectors, eliminating retrieval blindness at chunk seams.

    Args:
        documents: Raw page-level Document objects.
        chunk_size: Maximum character length of each chunk.
        chunk_overlap: Overlap length in characters between adjacent chunks.

    Returns:
        List of split Document chunks with chunk_index and lineage metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    # Enrich metadata with chunk identifiers for transparent inspectability in UI
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx + 1
        chunk.metadata["total_chunks"] = len(chunks)

    return chunks


def get_embeddings(api_key: Optional[str] = None) -> GoogleGenerativeAIEmbeddings:
    """
    Initializes Google Gemini text-embedding-004 embeddings.

    Embedding Choice: `models/text-embedding-004`
    - Dimensions: 768
    - Delivers top-tier retrieval performance on MTEB benchmarks.
    - Native support for multilingual, code, and technical document domains.
    """
    active_api_key = api_key or GEMINI_API_KEY
    if not active_api_key:
        raise ValueError("Gemini/Google API key must be provided or configured in .env")

    return GoogleGenerativeAIEmbeddings(
        model=DEFAULT_EMBEDDING_MODEL,
        google_api_key=active_api_key,
    )


def create_or_update_vectorstore(
    chunks: List[Document],
    persist_directory: str | Path = CHROMA_DIR,
    collection_name: str = "vidya_collection",
    api_key: Optional[str] = None,
) -> Chroma:
    """
    Generates Google Gemini embeddings and persists chunks into a ChromaDB vector store.

    Args:
        chunks: List of split Document chunks.
        persist_directory: Local disk path to store ChromaDB files.
        collection_name: Unique collection identifier in Chroma.
        api_key: Gemini/Google API Key (falls back to environment variable).

    Returns:
        Configured Chroma vectorstore instance.
    """
    embeddings = get_embeddings(api_key=api_key)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )

    return vectorstore


def get_vectorstore(
    persist_directory: str | Path = CHROMA_DIR,
    collection_name: str = "vidya_collection",
    api_key: Optional[str] = None,
) -> Chroma:
    """
    Retrieves a reference to an already existing ChromaDB vectorstore using Gemini embeddings.
    """
    embeddings = get_embeddings(api_key=api_key)

    return Chroma(
        persist_directory=str(persist_directory),
        collection_name=collection_name,
        embedding_function=embeddings,
    )


def clear_vectorstore(persist_directory: str | Path = CHROMA_DIR) -> None:
    """
    Safely deletes persisted ChromaDB data to allow fresh re-indexing.
    """
    dir_path = Path(persist_directory)
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
