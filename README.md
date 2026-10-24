# 📖 Vidya (विद्या) — Document Intelligence & Grounded Conversational RAG

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-green.svg)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-orange.svg)](https://www.trychroma.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini%20%7C%20Embeddings--3-purple.svg)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Vidya** (Sanskrit for *Knowledge* or *Learning*) is an enterprise-grade Retrieval-Augmented Generation (RAG) system engineered for high-precision document intelligence. It transforms dense, unstructured PDFs (research papers, financial filings, technical documentation) into an interactive, multi-turn conversational knowledge base with **verifiable, citation-backed answers and strict anti-hallucination guardrails**.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Ingestion["1. Document Ingestion Pipeline"]
        PDF["Uploaded PDF(s)"] --> Loader["PyPDFLoader (Page-level Extraction)"]
        Loader --> Splitter["RecursiveCharacterTextSplitter\n(chunk_size=1000, overlap=200)"]
        Splitter --> Embedder["OpenAI text-embedding-3-small\n(1536 Dimensions)"]
        Embedder --> Chroma[("ChromaDB Vector Store\n(Persistent Collections)")]
    end

    subgraph Retrieval["2. Grounded Conversational Retrieval"]
        UserQuery["User Query"] --> History["Chat History / Session Memory"]
        History --> Reformulator["History-Aware Query Reformulator\n(GPT-4o-mini)"]
        Reformulator --> Standalone["Standalone Search Query"]
        Standalone --> Retriever["Similarity Vector Search (Top-K)"]
        Chroma --> Retriever
        Retriever --> Context["Top-K Retrieved Chunks + Page Metadata"]
    end

    subgraph Generation["3. Factual Synthesis & Verification"]
        Context --> Guardrails["Strict Anti-Hallucination Prompt"]
        UserQuery --> Guardrails
        Guardrails --> LLM["GPT-4o-mini (temperature=0.0)"]
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
| **Embeddings** | `text-embedding-3-small` (1536 dim) | Superior MTEB benchmark performance over `text-embedding-ada-002` while reducing inference costs by ~5x and token footprint. |
| **Vector Store** | `ChromaDB` | Serverless, disk-persisted vector database supporting native metadata filtering (`file_name`, `page`, `chunk_index`). |
| **Inference Model** | `gpt-4o-mini` (`temperature=0.0`) | Zero temperature guarantees deterministic, non-creative generation strictly tethered to the retrieved vector chunks. |
| **Hallucination Safeguards** | Negative constraint prompting + strict fallback | Explicit system mandate: If retrieved context is insufficient, explicitly state *"I cannot find this information in the uploaded document(s)"* rather than extrapolating. |
| **Inspectability** | Collapsible UI Chunk Inspector | Displays raw text snippets, page numbers, and chunk IDs for every generated citation to enable instant auditing. |

---

## 🚀 Quickstart & Local Setup

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- OpenAI API Key ([Get an API key here](https://platform.openai.com/api-keys))

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
cp .env.example .env     # On Linux/macOS
copy .env.example .env   # On Windows
```

Edit `.env` and add your OpenAI API key:
```ini
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
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
├── app.py                    # Streamlit frontend with inspectability & chat memory
├── src/
│   ├── __init__.py
│   ├── config.py             # Centralized settings, model parameters & paths
│   ├── ingestion.py          # PDF extraction, chunking, and ChromaDB persistence
│   └── chain.py              # History-aware conversational retriever and grounded QA chain
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
   git commit -m "feat: initial Vidya production RAG system"
   git push origin main
   ```
2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/).
   - Select your repository and point the main file path to `app.py`.
3. **Configure Secrets**:
   - In the Streamlit Cloud dashboard, click **Advanced Settings** -> **Secrets**.
   - Add your OpenAI API key:
     ```toml
     OPENAI_API_KEY = "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
     ```
   - Click **Deploy**. Your app is live!

---

## 💼 Resume & Interview Bullet Points

Use these ready-to-use bullet points on your resume and LinkedIn:

- **Architected "Vidya"**, an enterprise conversational RAG platform integrating **LangChain**, **OpenAI GPT-4o-mini**, and **ChromaDB** for semantic PDF document intelligence.
- **Engineered an ingestion pipeline** leveraging `RecursiveCharacterTextSplitter` (1000-char chunks, 200-char overlap) and OpenAI's `text-embedding-3-small`, achieving contextual continuity across chunk boundaries.
- **Eliminated hallucinations** by deploying a deterministic multi-stage conversational chain with history-aware query reformulation, zero-temperature inference, and strict fallback constraints.
- **Implemented citation auditability** with page-level metadata preservation and a real-time collapsible source inspector in Streamlit, enabling instant fact-checking of model outputs.

---


