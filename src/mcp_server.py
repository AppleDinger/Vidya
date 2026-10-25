"""
Model Context Protocol (MCP) Server for Vidya.

Exposes Vidya's persistent ChromaDB document knowledge base to external AI coding agents,
Claude Desktop, Cursor, and IDE extensions via the standardized FastMCP interface.
"""

import os
import sys
from pathlib import Path
from typing import Set

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from mcp.server.fastmcp import FastMCP
except (ImportError, ModuleNotFoundError):
    from mcp.server.mcpserver import MCPServer as FastMCP

from src.config import CHROMA_DIR, GEMINI_API_KEY, is_api_key_configured
from src.ingestion import get_vectorstore

# Initialize FastMCP / MCPServer with identity
mcp = FastMCP("Vidya-Document-Intelligence")



def _ensure_active_api_key() -> None:
    """Verifies that a Gemini/Google API key is present for generating query embeddings."""
    if not is_api_key_configured():
        raise ValueError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not configured. Please set GEMINI_API_KEY in your .env "
            "or environment variables before starting the MCP server."
        )



@mcp.tool()
def search_documents(query: str, k: int = 4) -> str:
    """
    Search indexed documents in Vidya's vector database using semantic similarity.

    Args:
        query: The natural language search query or topic to look up.
        k: Number of relevant context chunks to retrieve (default: 4).

    Returns:
        A structured string containing retrieved text chunks, page numbers, and source metadata.
    """
    try:
        _ensure_active_api_key()

        if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
            return (
                "⚠️ No documents have been indexed yet in Vidya.\n"
                "Please run the Vidya application (`streamlit run app.py`) and upload/index "
                "PDF documents to populate the ChromaDB vector store."
            )

        vectorstore = get_vectorstore(persist_directory=CHROMA_DIR)
        results = vectorstore.similarity_search(query, k=k)

        if not results:
            return f"No relevant content found matching the query: '{query}'"

        formatted_outputs = []
        for idx, doc in enumerate(results, 1):
            source = doc.metadata.get("source", doc.metadata.get("file_name", "Unknown Source"))
            page = doc.metadata.get("page", "N/A")
            chunk_idx = doc.metadata.get("chunk_index", "N/A")
            total_chunks = doc.metadata.get("total_chunks", "N/A")

            header = f"[Chunk {idx}/{len(results)}] Source: {source} | Page: {page} (Chunk #{chunk_idx}/{total_chunks})"
            separator = "-" * len(header)
            formatted_outputs.append(f"{header}\n{separator}\n{doc.page_content.strip()}\n")

        return "\n".join(formatted_outputs)

    except Exception as e:
        return f"Error executing document search in Vidya MCP server: {str(e)}"


@mcp.tool()
def list_indexed_documents() -> str:
    """
    Lists all distinct documents currently indexed in Vidya's persistent vector store.

    Returns:
        Summary of indexed documents including distinct filenames, total chunk count,
        and database location.
    """
    try:
        _ensure_active_api_key()

        if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
            return "No documents are currently indexed in Vidya (ChromaDB is empty)."

        vectorstore = get_vectorstore(persist_directory=CHROMA_DIR)
        collection_data = vectorstore.get()

        metadatas = collection_data.get("metadatas", [])
        total_chunks = len(metadatas)

        if total_chunks == 0:
            return "ChromaDB collection exists but contains 0 indexed chunks."

        # Aggregate unique sources and page ranges
        distinct_sources: Set[str] = set()
        source_pages = {}

        for meta in metadatas:
            if meta:
                source = meta.get("source", meta.get("file_name", "Unknown Document"))
                distinct_sources.add(source)
                page = meta.get("page")
                if page:
                    source_pages.setdefault(source, set()).add(page)

        summary_lines = [
            "📚 **Vidya Indexed Knowledge Base**",
            f"- **Total Indexed Chunks:** {total_chunks}",
            f"- **Total Unique Documents:** {len(distinct_sources)}",
            f"- **Storage Path:** `{CHROMA_DIR}`\n",
            "**Indexed Documents:**",
        ]

        for source in sorted(distinct_sources):
            pages = source_pages.get(source, set())
            pages_str = f"({len(pages)} indexed pages)" if pages else ""
            summary_lines.append(f"  • {source} {pages_str}")

        return "\n".join(summary_lines)

    except Exception as e:
        return f"Error reading indexed documents from Vidya MCP server: {str(e)}"


if __name__ == "__main__":
    # Standard stdio transport for MCP clients (Cursor, Claude Desktop, etc.)
    mcp.run()
