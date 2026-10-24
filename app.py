"""
Vidya: Enterprise Document Intelligence & Conversational RAG Assistant
Streamlit Application Entrypoint.
"""

import os
import tempfile
from pathlib import Path
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from src.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_RETRIEVAL_K,
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    is_api_key_configured,
    OPENAI_API_KEY,
    UPLOADS_DIR,
    CHROMA_DIR,
)
from src.ingestion import (
    load_pdf,
    chunk_documents,
    create_or_update_vectorstore,
    clear_vectorstore,
)
from src.chain import build_rag_chain

# ==============================================================================
# Page Configuration & Modern Theme Styling
# ==============================================================================
st.set_page_config(
    page_title="Vidya | Document Intelligence & RAG",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, portfolio-grade UI
st.markdown(
    """
    <style>
    /* Main container enhancements */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    
    /* Hero Title Card */
    .hero-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.98rem;
        line-height: 1.5;
        margin: 0;
    }
    
    /* Status Badge styling */
    .badge-container {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.8rem;
    }
    
    .badge {
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        background: rgba(255, 255, 255, 0.05);
        color: #cbd5e1;
    }
    
    .badge-active {
        background: rgba(34, 197, 94, 0.15);
        border-color: rgba(34, 197, 94, 0.4);
        color: #4ade80;
    }
    
    /* Source Chunk Box in expander */
    .source-chunk-card {
        background-color: rgba(15, 23, 42, 0.6);
        border-left: 3px solid #38bdf8;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.75rem;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    
    .source-meta {
        color: #38bdf8;
        font-weight: 600;
        font-size: 0.8rem;
        margin-bottom: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Session State Initialization
# ==============================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # LangChain message objects

if "messages" not in st.session_state:
    st.session_state.messages = []  # UI display dicts: [{"role": ..., "content": ..., "context": [...]}]

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "indexed_docs_info" not in st.session_state:
    st.session_state.indexed_docs_info = None


# ==============================================================================
# Sidebar: Document Ingestion, Model Controls & Telemetry
# ==============================================================================
with st.sidebar:
    st.title("⚙️ Vidya Control Panel")
    st.caption("Document Intelligence & Retrieval Config")

    # API Key Handling (Environment variable or direct input for portability)
    user_api_key = OPENAI_API_KEY
    if not is_api_key_configured():
        st.warning("⚠️ OpenAI API Key not detected in `.env`.")
        user_api_key = st.text_input(
            "Enter OpenAI API Key:",
            type="password",
            help="Your key is kept only in memory during the session.",
        )
    else:
        st.success("🔒 OpenAI API Key Loaded from `.env`", icon="✅")

    st.markdown("---")
    st.subheader("📄 Document Ingestion")

    uploaded_files = st.file_uploader(
        "Upload Technical / Research PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multi-page PDFs to test semantic extraction and grounded citations.",
    )

    with st.expander("🛠️ Token & Retrieval Hyperparameters", expanded=False):
        chunk_size = st.slider(
            "Chunk Size (Chars)",
            min_value=400,
            max_value=2000,
            value=DEFAULT_CHUNK_SIZE,
            step=100,
            help="Target character length for each recursive text split (~4 chars/token).",
        )
        chunk_overlap = st.slider(
            "Chunk Overlap (Chars)",
            min_value=50,
            max_value=400,
            value=DEFAULT_CHUNK_OVERLAP,
            step=25,
            help="Character overlap to preserve boundary context across chunks.",
        )
        retrieval_k = st.slider(
            "Top-K Retrieval",
            min_value=2,
            max_value=8,
            value=DEFAULT_RETRIEVAL_K,
            step=1,
            help="Number of relevant document chunks fetched per query.",
        )

    # Ingestion Trigger Button
    if uploaded_files:
        if st.button("🚀 Ingest & Index Document(s)", use_container_width=True, type="primary"):
            if not user_api_key:
                st.error("Please provide a valid OpenAI API Key before indexing.")
            else:
                with st.spinner("Processing PDF(s), generating embeddings, and building ChromaDB index..."):
                    try:
                        all_docs = []
                        total_pages = 0

                        # Process each uploaded file
                        for uploaded_file in uploaded_files:
                            # Save to temp file for PyPDFLoader
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                tmp_file.write(uploaded_file.read())
                                tmp_path = tmp_file.name

                            docs = load_pdf(tmp_path)
                            # Overwrite temp path with actual filename for clean citations
                            for doc in docs:
                                doc.metadata["source"] = uploaded_file.name
                                doc.metadata["file_name"] = uploaded_file.name

                            all_docs.extend(docs)
                            total_pages += len(docs)
                            # Cleanup temp file
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass

                        # Split documents into chunks
                        chunks = chunk_documents(
                            documents=all_docs,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )

                        # Persist to ChromaDB vector store
                        vectorstore = create_or_update_vectorstore(
                            chunks=chunks,
                            persist_directory=CHROMA_DIR,
                            api_key=user_api_key,
                        )

                        # Assemble conversational chain
                        rag_chain = build_rag_chain(
                            vectorstore=vectorstore,
                            retrieval_k=retrieval_k,
                            model_name=DEFAULT_LLM_MODEL,
                            api_key=user_api_key,
                        )

                        # Store in session state
                        st.session_state.vectorstore = vectorstore
                        st.session_state.rag_chain = rag_chain
                        st.session_state.indexed_docs_info = {
                            "files": [f.name for f in uploaded_files],
                            "pages": total_pages,
                            "chunks": len(chunks),
                        }

                        # Reset chat messages on new document ingestion
                        st.session_state.chat_history = []
                        st.session_state.messages = []

                        st.success(f"Indexed {len(chunks)} chunks across {total_pages} pages!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error during document ingestion: {str(e)}")

    st.markdown("---")

    # System Diagnostics & Architecture Badges
    st.subheader("📊 System Specs")
    st.markdown(
        f"""
        - **LLM Engine:** `{DEFAULT_LLM_MODEL}`
        - **Temperature:** `0.0` (Deterministic)
        - **Embedding Model:** `{DEFAULT_EMBEDDING_MODEL}`
        - **Vector Database:** `ChromaDB` (Persistent)
        - **Chunk Splitter:** `RecursiveCharacter`
        """
    )

    if st.session_state.indexed_docs_info:
        info = st.session_state.indexed_docs_info
        st.markdown(
            f"""
            **Active Knowledge Base:**
            - **Files:** {', '.join(info['files'])}
            - **Total Pages:** {info['pages']}
            - **Indexed Chunks:** {info['chunks']}
            """
        )

    # Session Reset Controls
    if st.button("🗑️ Reset Chat & Knowledge Base", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.session_state.rag_chain = None
        st.session_state.indexed_docs_info = None
        clear_vectorstore(CHROMA_DIR)
        st.info("System state and local vector store reset successfully.")
        st.rerun()


# ==============================================================================
# Main Content Area
# ==============================================================================

# Hero Header
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Vidya (विद्या)</div>
        <p class="hero-subtitle">
            Enterprise-Grade Document Intelligence & Grounded Conversational Q&A.
            Extract semantic insights with verifiable, citation-backed sources and zero hallucinations.
        </p>
        <div class="badge-container">
            <span class="badge">OpenAI text-embedding-3-small</span>
            <span class="badge">ChromaDB Vector Store</span>
            <span class="badge">GPT-4o-mini (Temp=0)</span>
            <span class="badge">Multi-Turn Memory</span>
            <span class="badge">Source Inspectability</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Display empty state instructions if no documents are indexed yet
if not st.session_state.rag_chain:
    st.info(
        "👋 **Welcome to Vidya!** To get started:\n"
        "1. Enter or verify your OpenAI API key in the sidebar.\n"
        "2. Upload one or more PDF documents (e.g., technical papers, financial reports, or legal contracts).\n"
        "3. Click **'Ingest & Index Document(s)'** to generate embeddings and initialize conversational retrieval.",
        icon="💡",
    )

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Collapsible inspectability for assistant responses
        if msg["role"] == "assistant" and msg.get("context"):
            with st.expander("🔍 Inspect Retrieved Context & Citations", expanded=False):
                st.caption(
                    f"Retrieved {len(msg['context'])} grounded chunks via similarity search to generate this answer:"
                )
                for i, doc in enumerate(msg["context"], 1):
                    file_name = doc.metadata.get("file_name", "Unknown Document")
                    page_num = doc.metadata.get("page", "N/A")
                    chunk_idx = doc.metadata.get("chunk_index", "N/A")
                    total_chunks = doc.metadata.get("total_chunks", "N/A")

                    st.markdown(
                        f"""
                        <div class="source-chunk-card">
                            <div class="source-meta">
                                📌 [Chunk {i}/{len(msg['context'])}] • Source: {file_name} • Page {page_num} (Chunk #{chunk_idx}/{total_chunks})
                            </div>
                            <div>{doc.page_content}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ==============================================================================
# Chat Input & Response Generation
# ==============================================================================
user_prompt = st.chat_input("Ask a question about your uploaded document(s)...")

if user_prompt:
    # Validate preconditions
    if not st.session_state.rag_chain:
        st.warning("Please upload and index a PDF document before asking questions.")
    else:
        # 1. Display user message in UI
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # 2. Invoke Conversational RAG chain
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document embeddings and formulating grounded answer..."):
                try:
                    response = st.session_state.rag_chain.invoke(
                        {
                            "input": user_prompt,
                            "chat_history": st.session_state.chat_history,
                        }
                    )

                    answer_text = response.get("answer", "No response generated.")
                    retrieved_docs = response.get("context", [])

                    # Render answer
                    st.markdown(answer_text)

                    # Render source inspection expander immediately
                    if retrieved_docs:
                        with st.expander("🔍 Inspect Retrieved Context & Citations", expanded=False):
                            st.caption(
                                f"Retrieved {len(retrieved_docs)} grounded chunks via similarity search to generate this answer:"
                            )
                            for i, doc in enumerate(retrieved_docs, 1):
                                file_name = doc.metadata.get("file_name", "Unknown Document")
                                page_num = doc.metadata.get("page", "N/A")
                                chunk_idx = doc.metadata.get("chunk_index", "N/A")
                                total_chunks = doc.metadata.get("total_chunks", "N/A")

                                st.markdown(
                                    f"""
                                    <div class="source-chunk-card">
                                        <div class="source-meta">
                                            📌 [Chunk {i}/{len(retrieved_docs)}] • Source: {file_name} • Page {page_num} (Chunk #{chunk_idx}/{total_chunks})
                                        </div>
                                        <div>{doc.page_content}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    # Update session state histories
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer_text,
                            "context": retrieved_docs,
                        }
                    )

                    # Update LangChain conversational memory
                    st.session_state.chat_history.append(HumanMessage(content=user_prompt))
                    st.session_state.chat_history.append(AIMessage(content=answer_text))

                except Exception as e:
                    error_msg = f"An error occurred while answering: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
