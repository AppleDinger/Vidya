# AGENTS.md — Agentic Operating Guidelines for Vidya

Welcome, AI Agent! This document specifies the operational standards, architectural boundaries, and execution protocols for working inside the **Vidya** codebase. Always read and adhere to these directives before analyzing, editing, or testing code.

---

## 🎯 1. Project Mission & Core Philosophy

**Vidya** (विद्या, Sanskrit for *Knowledge* or *Learning*) is an enterprise-grade Document Intelligence & Conversational RAG platform. 
The system transforms unstructured multi-page PDFs into an interactive knowledge base with **verifiable, citation-backed answers and strict anti-hallucination guardrails**.

### Core Tenet: Zero Hallucination
- **Absolute Grounding:** Any claim made by the assistant must be directly supported by retrieved vector chunks.
- **Auditable Lineage:** Chunks must retain document source names, page numbers, and chunk indices.
- **Graceful Fallback:** If context is insufficient, explicitly state: *"I cannot find this information in the uploaded document(s)."* Never fabricate plausible conjectures.

---

## 💻 2. Tech Stack & Environment

- **Runtime:** Python 3.10+ (Tested on Python 3.10 - 3.13)
- **Virtual Environment Directory:** `./venv`
  - Windows Activation: `.\venv\Scripts\Activate.ps1`
  - Unix/macOS Activation: `source venv/bin/activate`
- **Primary Libraries:**
  - `streamlit`: Reactive user interface with chat history and collapsible chunk inspectability.
  - `langchain`, `langchain-openai`, `langchain-community`: LCEL orchestration and conversational retrieval chains.
  - `langchain-chroma`, `chromadb`: Persistent, disk-backed vector database.
  - `pypdf`: PDF text extraction with page metadata preservation.
  - `mcp[cli]`: Model Context Protocol server (`FastMCP`) exposing search tools to external agents.
  - `python-dotenv`: Environment configuration management.

---

## 🛠️ 3. Build, Test & Run Commands

All commands should be executed from the repository root:

```bash
# 1. Install / Update Dependencies
pip install -r requirements.txt

# 2. Run Sanity & Unit Verification Tests
python test_sanity.py

# 3. Launch Streamlit Web Application
streamlit run app.py

# 4. Run / Inspect MCP Server via FastMCP CLI
mcp dev src/mcp_server.py

# 5. Direct STDIO MCP Execution (for Cursor / Claude Desktop)
python src/mcp_server.py
```

---

## 🧱 4. Architectural Boundaries & Module Separation

Maintain strict separation of concerns across the codebase:

```text
Vidya/
├── app.py                  # PRESENTATION LAYER: Streamlit UI, state management, chat rendering
├── src/
│   ├── config.py           # CONFIGURATION: Centralized settings, model defaults, paths
│   ├── ingestion.py        # INGESTION: PDF parsing, recursive chunking, ChromaDB persistence
│   ├── chain.py            # REASONING: History-aware retrieval & grounded answering chain
│   └── mcp_server.py       # INTEROPERABILITY: FastMCP server exposing tools to external agents
└── data/
    └── chroma_db/          # DATA STORE: Local persistent vector collections (git-ignored)
```

### Module Rules:
1. **`src/config.py`**:
   - Single source of truth for paths (`CHROMA_DIR`, `DATA_DIR`), default models (`gpt-4o-mini`, `text-embedding-3-small`), and chunking hyperparameters.
   - Do NOT hardcode file paths or API keys anywhere else in the code.
2. **`src/ingestion.py`**:
   - Responsible for raw data conversion into indexed embeddings.
   - Must normalize page numbers to 1-indexed for human readability.
   - Must preserve `source`, `file_name`, `page`, and `chunk_index` in document metadata.
3. **`src/chain.py`**:
   - Manages prompt engineering, query reformulation, and LCEL chain construction.
   - Keep temperature strictly at `0.0`.
   - Never strip context documents from the output payload — the UI and MCP tools depend on inspectable source chunks.
4. **`src/mcp_server.py`**:
   - Exposes clean, typed tools (`search_documents`, `list_indexed_documents`) to external agents.
   - Handles missing API keys and empty vector stores gracefully without unhandled exceptions.
5. **`app.py`**:
   - Presentation only. Do not put raw embedding mathematics or direct Chroma collection mutations in `app.py`. Call helpers from `src/ingestion.py` and `src/chain.py`.

---

## 🛡️ 5. Security & Secret Safeguards

- **Strict `.gitignore` enforcement:** Never commit `.env`, `.env.local`, API keys, or local vector store data (`data/chroma_db/`).
- **OpenAI Key Validation:** Always verify `OPENAI_API_KEY` exists via `src.config.is_api_key_configured()` before calling external OpenAI endpoints.
- **Fail Gracefully:** Inform the user cleanly via UI warning or CLI message if their API key is missing.

---

## 📝 6. Coding & Git Conventions

- **Type Hints:** Use explicit Python type annotations (`List[Document]`, `Optional[str]`, `str | Path`).
- **Docstrings:** Document every public function with Google/Sphinx style docstrings outlining parameters, returns, and architectural rationale.
- **Atomic Commits:** Follow conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`.
- **Formatting:** Adhere to PEP 8 standard formatting.
