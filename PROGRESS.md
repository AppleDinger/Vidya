# 📊 Vidya — Project Progress & Engineering Roadmap

*Tracking development velocity, system milestones, and architectural decisions for Vidya.*

---

## 📍 1. Current State
- **Current Version:** `v0.2.0` (Core RAG Pipeline + Model Context Protocol Server)
- **Status:** Active Development & Protocol Integration
- **Target Audience:** Technical Recruiters, AI Engineering Teams, External LLM Agents (Cursor / Claude Desktop)

---

## ✅ 2. Completed Milestones

- [x] **Project Scaffolding & Virtual Environment**: Modular Python package structure (`src/`, `data/`), dependency specification (`requirements.txt`), and automated virtual environment setup (`venv`).
- [x] **Configuration & Secret Safeguards**: Centralized config engine (`src/config.py`) with strict `.env` loading, `.gitignore` filters, and fallback validation.
- [x] **Document Ingestion Pipeline**: High-fidelity PDF loading via `PyPDFLoader` with normalized 1-indexed page numbering and file lineage metadata.
- [x] **Semantic Chunking Strategy**: `RecursiveCharacterTextSplitter` configured at 1000 characters with 200 character overlap to preserve semantic continuity across chunk seams.
- [x] **ChromaDB Persistent Indexing**: Vector database initialization with OpenAI `text-embedding-3-small` embeddings (1536 dimensions) persisted to disk.
- [x] **Conversational Retrieval Engine**: History-aware query reformulation (`create_history_aware_retriever`) paired with LCEL document combination chains.
- [x] **Zero-Hallucination Guardrails**: Deterministic system prompting (`temperature=0.0`) enforcing inline page citations (`[Page X]`) and explicit fallback when information is absent.
- [x] **Streamlit UI with Source Inspectability**: Custom-styled chat interface (`app.py`) featuring conversational bubbles and collapsible expanders detailing raw chunks, page numbers, and chunk lineage.
- [x] **Unit & Sanity Verification Suite**: Automated local test script (`test_sanity.py`) verifying chunking mechanics, metadata retention, and prompt integrity.

---

## 🔄 3. In Progress (Current Phase)

- [ ] **Model Context Protocol (MCP) Server**: Exposing Vidya's vector database via `FastMCP` (`src/mcp_server.py`) with standardized tools (`search_documents`, `list_indexed_documents`).
- [ ] **Agent Tooling Verification**: Testing tool invocations via FastMCP inspector CLI (`mcp dev src/mcp_server.py`) and standard JSON configuration (`mcp_config.json.example`).
- [ ] **Agentic Guidelines (`AGENTS.md`)**: Linux Foundation / Agentic AI compliant operating manual for autonomous and human pair-programmers.

---

## 🔮 4. Upcoming Roadmap

- [ ] **Hybrid Search Integration**: Combining sparse lexical search (BM25) with dense semantic embeddings (Reciprocal Rank Fusion - RRF) for superior technical keyword retrieval.
- [ ] **RAGAS Automated Evaluation Pipeline**: Automated scoring across Faithfulness, Answer Relevance, Context Recall, and Context Precision.
- [ ] **Cloud Deployment**: Hosting on Streamlit Community Cloud with encrypted secrets management.
- [ ] **Portfolio Demonstration Asset**: Interactive walkthrough video and animated architectural showcase for GitHub and LinkedIn.

---

## 📜 5. Architecture Changelog

### [v0.2.0] — 2026-10-24
- **Added:** FastMCP server in `src/mcp_server.py` exposing `search_documents` and `list_indexed_documents`.
- **Added:** `mcp_config.json.example` for Claude Desktop and Cursor integration.
- **Added:** `AGENTS.md` specifying operational rules for AI coding agents.
- **Added:** `PROGRESS.md` for team and recruiter visibility into development milestones.
- **Updated:** `requirements.txt` to include `mcp[cli]`.

### [v0.1.0] — 2026-10-24
- **Added:** Initial project structure, `requirements.txt`, `.gitignore`, and `.env.example`.
- **Added:** `src/config.py` for centralized hyperparameters and environment detection.
- **Added:** `src/ingestion.py` for PDF parsing, text splitting, and ChromaDB vector persistence.
- **Added:** `src/chain.py` featuring history-aware retrieval and citation prompting.
- **Added:** `app.py` Streamlit frontend with source inspectability expanders.
- **Added:** `README.md` with Mermaid architecture diagram, technical decisions, and resume bullet points.
- **Added:** `test_sanity.py` verification test suite.
