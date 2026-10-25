"""
Sanity and Unit Verification Script for Vidya.
Validates chunking logic, metadata preservation, and chain prompts locally.
"""

import sys
from pathlib import Path
from langchain_core.documents import Document

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.ingestion import chunk_documents
from src.chain import CONTEXTUALIZE_Q_PROMPT, QA_PROMPT


def test_chunking_and_metadata():
    print("[1/4] Testing chunking and metadata preservation...")
    sample_text = (
        "Retrieval-Augmented Generation (RAG) is an AI architecture that enhances "
        "Large Language Models by grounding them on external knowledge bases.\n\n"
        "Vidya uses Google Gemini's text-embedding-004 and ChromaDB for vector retrieval. "
        "Each chunk maintains strict page lineage and metadata to guarantee that every "
        "synthesized fact can be cross-examined against original PDF sources."
    )
    mock_doc = Document(
        page_content=sample_text,
        metadata={"source": "test_spec.pdf", "page": 1, "file_name": "test_spec.pdf"},
    )

    chunks = chunk_documents([mock_doc], chunk_size=150, chunk_overlap=30)
    assert len(chunks) > 0, "No chunks generated!"
    assert chunks[0].metadata["page"] == 1, "Page metadata lost during chunking!"
    assert chunks[0].metadata["file_name"] == "test_spec.pdf", "Filename lost!"
    assert "chunk_index" in chunks[0].metadata, "chunk_index missing in chunk metadata!"
    print(f"      Successfully generated {len(chunks)} chunks with verified metadata.")


def test_prompts():
    print("[2/4] Verifying prompt templates and anti-hallucination guardrails...")
    assert "Vidya" in QA_PROMPT.messages[0].prompt.template
    assert "I cannot find this information" in QA_PROMPT.messages[0].prompt.template
    assert "standalone question" in CONTEXTUALIZE_Q_PROMPT.messages[0].prompt.template
    print("      Prompts validated with strict anti-hallucination constraints.")


def test_imports():
    print("[3/4] Validating module imports and dependency integrity...")
    import src.config
    import src.ingestion
    import src.chain
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    print("      All Vidya core modules & Google GenAI dependencies imported cleanly.")



def test_mcp_server():
    print("[4/4] Verifying MCP Server initialization & tools...")
    from src.mcp_server import mcp, search_documents, list_indexed_documents
    assert mcp.name == "Vidya-Document-Intelligence", f"Unexpected MCP server name: {mcp.name}"
    # Test tool behavior with empty/unconfigured state to verify graceful error handling
    empty_result = list_indexed_documents()
    assert isinstance(empty_result, str), "list_indexed_documents should return a string summary"
    print("      MCP Server initialized and tools registered successfully.")


if __name__ == "__main__":
    print("========================================")
    print("Running Vidya Sanity & Integration Tests")
    print("========================================")
    try:
        test_imports()
        test_chunking_and_metadata()
        test_prompts()
        test_mcp_server()
        print("\nAll sanity checks passed successfully!")
    except Exception as e:
        print(f"\nSanity check failed: {e}")
        sys.exit(1)
