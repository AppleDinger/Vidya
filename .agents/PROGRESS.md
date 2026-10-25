# 📊 Vidya — Project Progress & Engineering Roadmap

*Tracking development velocity, system milestones, and architectural decisions for Vidya.*

---

## 📍 1. Current State
- **Current Version:** `v0.3.0` (Google Gemini 1.5 Pro + text-embedding-004 + Model Context Protocol Server)
- **Status:** Active Development & Protocol Integration
- **Target Audience:** Technical Recruiters, AI Engineering Teams, External LLM Agents (Cursor / Claude Desktop)

---

## ✅ 2. Completed Milestones

- [x] **Project Scaffolding & Virtual Environment**: Modular Python package structure (`src/`, `data/`), dependency specification (`requirements.txt`), and automated virtual environment setup (`venv`).
- [x] **Configuration & Secret Safeguards**: Centralized config engine (`src/config.py`) with strict `.env` loading, `.gitignore` filters, and fallback validation for `GEMINI_API_KEY` / `GOOGLE_API_KEY`.
- [x] **Document Ingestion Pipeline**: High-fidelity PDF loading via `PyPDFLoader` with normalized 1-indexed page numbering and file lineage metadata.
- [x] **Semantic Chunking Strategy**: `RecursiveCharacterTextSplitter` configured at 1000 characters with 200 character overlap to preserve semantic continuity across chunk seams.
- [x] **ChromaDB Persistent Indexing with Google GenAI**: Vector database initialization with Google `models/text-embedding-004` embeddings (768 dimensions) persisted to disk.
- [x] **Conversational Retrieval Engine**: History-aware query reformulation (`create_history_aware_retriever`) paired with LCEL document combination chains.
- [x] **Zero-Hallucination Guardrails**: Deterministic system prompting (`temperature=0.0`) enforcing inline page citations (`[Page X]`) and explicit fallback when information is absent.
- [x] **Streamlit UI with Source Inspectability & Model Selector**: Custom-styled chat interface (`app.py`) featuring conversational bubbles, collapsible expanders detailing raw chunks, page numbers, and model switching (`gemini-1.5-pro` / `gemini-1.5-flash`).
- [x] **Model Context Protocol (MCP) Server**: Exposing Vidya's vector database via `FastMCP` / `MCPServer` (`src/mcp_server.py`) with standardized tools (`search_documents`, `list_indexed_documents`) compatible with `mcp` 1.x and 2.x.
- [x] **Agentic Guidelines (`AGENTS.md`)**: Operating manual defining architectural boundaries, type safety, and zero-hallucination standards.
- [x] **Unit & Sanity Verification Suite**: Automated local test script (`test_sanity.py`) verifying chunking mechanics, metadata retention, and prompt integrity.

---

## 🔄 3. In Progress (Current Phase)

- [ ] **Agent Tooling Verification**: Live verification of tool invocations via FastMCP inspector CLI (`mcp dev src/mcp_server.py`) and standard JSON configuration (`mcp_config.json.example`).
- [ ] **Empirical Document Q&A Test**: Validating multi-page research PDF retrieval with live `GEMINI_API_KEY`.

---

## 🔮 4. Upcoming Roadmap

### 🚀 Phase 1 (v0.4.0) — Hybrid Search & Retrieval Augmentation
- [ ] **BM25 Lexical Search Integration**: Add `rank_bm25` sparse keyword retriever alongside dense `text-embedding-004` vectors.
- [ ] **Reciprocal Rank Fusion (RRF)**: Implement LangChain's `EnsembleRetriever` to intelligently merge sparse keyword and dense semantic scores.
- [ ] **Context Reranker (FlashRank / Cross-Encoder)**: Second-stage reranking to prioritize highest-relevance chunks before prompt synthesis.

### 📈 Phase 2 (v0.5.0) — RAGAS Automated Evaluation Pipeline
- [ ] **Synthetic QA Dataset Generation**: Automatic generation of test question-answer pairs from representative technical PDFs.
- [ ] **Automated Scoring Suite**: Empirical evaluation of *Faithfulness*, *Answer Relevance*, *Context Recall*, and *Context Precision*.
- [ ] **Regression Benchmarking**: CI-integrated test harness preventing retrieval degradation during chunking or model updates.

### ☁️ Phase 3 (v0.6.0) — Cloud Deployment & Containerization
- [ ] **Streamlit Community Cloud Deployment**: Live public demonstration with encrypted secret management for `GEMINI_API_KEY`.
- [ ] **Docker Containerization**: Production-ready `Dockerfile` and `docker-compose.yml` for isolated deployment.

### 🎥 Phase 4 (v0.7.0) — Portfolio Demonstration Asset
- [ ] **Interactive Video Walkthrough**: High-resolution recorded walkthrough demonstrating zero-hallucination citations and collapsible chunk auditing.
- [ ] **Recruiter Showcase**: Architecture diagrams, performance benchmarks, and resume highlights published to GitHub and LinkedIn.

---

## 📜 5. Architecture Changelog

### [v0.3.0] — 2026-10-25
- **Migrated:** Replaced OpenAI LLM and embeddings with **Google Gemini 1.5 Pro** and **Google text-embedding-004** (768 dimensions).
- **Added:** `langchain-google-genai` integration with dual support for `GEMINI_API_KEY` and `GOOGLE_API_KEY`.
- **Added:** Dynamic model selector in Streamlit sidebar (`gemini-1.5-pro` for deep reasoning, `gemini-1.5-flash` for high throughput).
- **Added:** Cross-version compatibility in `src/mcp_server.py` supporting both `mcp` 1.x (`FastMCP`) and 2.x (`MCPServer`).
- **Added:** Resilient import fallback in `src/chain.py` across LangChain 0.x and 1.x (`langchain_classic`).
- **Updated:** `README.md`, `AGENTS.md`, `.env.example`, and `test_sanity.py` to reflect the Google Gemini architecture.

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

