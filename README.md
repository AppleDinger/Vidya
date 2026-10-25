# 📖 Vidya (विद्या) — Document Intelligence & Grounded Conversational RAG

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-green.svg)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-orange.svg)](https://www.trychroma.com/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-1.5%20Pro%20%7C%20Embeddings--004-4285F4.svg)](https://aistudio.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Vidya** (Sanskrit for *Knowledge* or *Learning*) is an enterprise-grade Retrieval-Augmented Generation (RAG) system engineered for high-precision document intelligence. Powered by **Google Gemini 1.5 Pro** and **Google text-embedding-004**, it transforms dense, unstructured PDFs (research papers, financial filings, technical documentation) into an interactive, multi-turn conversational knowledge base with **verifiable, citation-backed answers and strict anti-hallucination guardrails**.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Ingestion["1. Document Ingestion Pipeline"]
        PDF["Uploaded PDF(s)"] --> Loader["PyPDFLoader (Page-level Extraction)"]
        Loader --> Splitter["RecursiveCharacterTextSplitter\n(chunk_size=1000, overlap=200)"]
        Splitter --> Embedder["Google text-embedding-004\n(768 Dimensions)"]
        Embedder --> Chroma[("ChromaDB Vector Store\n(Persistent Collections)")]
    end

    subgraph Retrieval["2. Grounded Conversational Retrieval"]
        UserQuery["User Query"] --> History["Chat History / Session Memory"]
        History --> Reformulator["History-Aware Query Reformulator\n(Gemini 1.5 Pro)"]
        Reformulator --> Standalone["Standalone Search Query"]
        Standalone --> Retriever["Similarity Vector Search (Top-K)"]
        Chroma --> Retriever
        Retriever --> Context["Top-K Retrieved Chunks + Page Metadata"]
    end

    subgraph Generation["3. Factual Synthesis & Verification"]
        Context --> Guardrails["Strict Anti-Hallucination Prompt"]
        UserQuery --> Guardrails
        Guardrails --> LLM["Gemini 1.5 Pro (temperature=0.0)"]
        LLM --> Output["Grounded Response with [Page X] Citations"]
        Output --> UI["Streamlit UI with Collapsible Source Inspectability"]
        Context --> UI
    end
```

---

## 💡 Key AI Engineering Decisions

| Dimension | Implementation | Technical Justification |
| :--- | :--- | :--- |
| **Chunking Strategy** | `RecursiveCharacterTextSplitter` (`size=1000`, `overlap=200`) | Prioritizes double newlines (`\n\n`), single newlines, and spaces to preserve natural paragraph semantics. Overlap guarantees concepts spanning chunk seams are never split from their context. |
| **Embeddings** | `models/text-embedding-004` (768 dim) | Google's state-of-the-art embedding architecture delivering industry-leading MTEB retrieval performance across multilingual and technical domains with generous free-tier quotas. |
| **Vector Store** | `ChromaDB` | Serverless, disk-persisted vector database supporting native metadata filtering (`file_name`, `page`, `chunk_index`). |
| **Inference Model** | `gemini-1.5-pro` (`temperature=0.0`) | Zero temperature guarantees deterministic, factual generation strictly tethered to retrieved vector chunks. Long-context architecture excels at synthesized cross-page reasoning. |
| **Hallucination Safeguards** | Negative constraint prompting + strict fallback | Explicit system mandate: If retrieved context is insufficient, explicitly state *"I cannot find this information in the uploaded document(s)"* rather than extrapolating. |
| **Inspectability** | Collapsible UI Chunk Inspector | Displays raw text snippets, page numbers, and chunk IDs for every generated citation to enable instant auditing. |
| **Agent Interoperability** | Model Context Protocol (FastMCP) | Standardized tools (`search_documents`, `list_indexed_documents`) allowing external AI coding agents (Cursor, Claude Desktop) to query Vidya's vector base. |

---

## 🚀 Quickstart & Local Setup

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Google Gemini API Key ([Get a free API key at Google AI Studio](https://aistudio.google.com/))

### 2. Clone Repository & Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/your-username/vidya.git
cd vidya

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Create local .env from example template
# On Linux/macOS:
cp .env.example .env
# On Windows (PowerShell):
copy .env.example .env
```

Edit `.env` and add your Google Gemini API key:
```ini
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 5. Launch the Streamlit App

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
Vidya/
├── .env.example              # Environment variables template
├── .gitignore                # Security filters (blocks .env, venv, and vector stores)
├── requirements.txt          # Pinned production dependencies
├── README.md                 # System documentation & architectural spec
├── PROGRESS.md               # Engineering roadmap, milestones, and changelog
├── AGENTS.md                 # Operational directives for autonomous & pair agents
├── app.py                    # Streamlit frontend with inspectability & chat memory
├── test_sanity.py            # Local verification suite
├── src/
│   ├── __init__.py
│   ├── config.py             # Centralized settings, model parameters & paths
│   ├── ingestion.py          # PDF extraction, chunking, and ChromaDB persistence
│   ├── chain.py              # History-aware conversational retriever and grounded QA chain
│   └── mcp_server.py         # FastMCP server for Cursor and Claude Desktop tools
└── data/
    ├── chroma_db/            # Persistent vector database storage (git-ignored)
    └── uploads/              # Local scratch space for uploaded files
```

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

Vidya is structured for one-click deployment to **Streamlit Community Cloud**:

1. **Push to GitHub**:
   Ensure `.env` and local database directories are **not** committed (`.gitignore` handles this automatically).
   ```bash
   git add .
   git commit -m "feat: migrate to Google Gemini 1.5 Pro and text-embedding-004"
   git push origin main
   ```
2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/).
   - Select your repository and point the main file path to `app.py`.
3. **Configure Secrets**:
   - In the Streamlit Cloud dashboard, click **Advanced Settings** -> **Secrets**.
   - Add your Gemini API key:
     ```toml
     GEMINI_API_KEY = "AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
     ```
   - Click **Deploy**. Your app is live!

---

## 💼 Resume & Interview Bullet Points

Use these ready-to-use bullet points on your resume and LinkedIn:

- **Architected "Vidya"**, an enterprise conversational RAG platform integrating **LangChain**, **Google Gemini 1.5 Pro**, and **ChromaDB** for semantic PDF document intelligence.
- **Engineered an ingestion pipeline** leveraging `RecursiveCharacterTextSplitter` (1000-char chunks, 200-char overlap) and Google's `text-embedding-004` (768 dim), achieving contextual continuity across chunk boundaries.
- **Eliminated hallucinations** by deploying a deterministic multi-stage conversational chain with history-aware query reformulation, zero-temperature inference, and strict fallback constraints.
- **Integrated Model Context Protocol (MCP)** via FastMCP, exposing semantic search and document inspection tools to external AI agents (Cursor, Claude Desktop).
- **Implemented citation auditability** with page-level metadata preservation and a real-time collapsible source inspector in Streamlit, enabling instant fact-checking of model outputs.



