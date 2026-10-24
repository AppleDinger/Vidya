"""
Conversational Retrieval & Grounded Prompting Chain for Vidya.

Implements:
1. History-aware query reformulation for follow-up conversational turns.
2. Strict hallucination-resistant system prompting.
3. Direct source citation enforcement (document name + page numbers).
4. End-to-end LCEL retrieval chain returning both final answers and retrieved context documents.
"""

from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma

from src.config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_RETRIEVAL_K,
    OPENAI_API_KEY,
)


# ==============================================================================
# Prompt Engineering & Grounding Safeguards
# ==============================================================================

# Prompt 1: Contextualize Question (Query Rewriting for Multi-turn Continuity)
CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Prompt 2: Grounded Answering & Strict Anti-Hallucination Guardrails
QA_SYSTEM_PROMPT = """You are Vidya, an enterprise-grade Document Intelligence and Research Assistant.
Your highest imperative is absolute factual grounding and zero hallucination.

Operating Rules:
1. Grounding: Answer the question using ONLY the provided retrieved context below. Do not assume, extrapolate, or use pre-trained external knowledge not present in the context.
2. Missing Information: If the context does not contain sufficient information to answer the question, state:
   "I cannot find this information in the uploaded document(s)."
   Do not guess, apologize profusely, or fabricate plausible answers.
3. Citations: For every major claim, bullet point, or fact you provide, include an inline citation to the source document and page number in the format: [Page X] or [File: filename, Page X].
4. Tone & Clarity: Maintain a structured, professional, and analytical tone. Use markdown headings and bullet points where appropriate for readability.

Retrieved Context:
{context}"""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def get_llm(
    model_name: str = DEFAULT_LLM_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    api_key: Optional[str] = None,
) -> ChatOpenAI:
    """
    Initializes the OpenAI LLM instance.

    Temperature is strictly set to 0.0 to eliminate stochastic variance
    and enforce deterministic, grounded factual retrieval.
    """
    active_api_key = api_key or OPENAI_API_KEY
    if not active_api_key:
        raise ValueError("OpenAI API key must be provided or configured in .env")

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=active_api_key,
    )


def build_rag_chain(
    vectorstore: Chroma,
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
    model_name: str = DEFAULT_LLM_MODEL,
    api_key: Optional[str] = None,
):
    """
    Constructs a production-grade conversational retrieval chain.

    Pipeline Flow:
    User Input + Chat History
      -> [History-Aware Retriever] -> Standalone Query
      -> [ChromaDB Vector Store] -> Top-K Document Chunks (with page metadata)
      -> [Stuff Documents QA Chain with Strict Guardrails]
      -> Grounded Answer with Citations + Inspectable Source Chunks

    Returns:
        A LangChain Runnable returning a dict with keys:
        - "answer": The LLM generated response
        - "context": List of retrieved Document chunks with metadata
    """
    llm = get_llm(model_name=model_name, api_key=api_key)

    # Configure retriever with similarity search and top-k filtering
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": retrieval_k},
    )

    # 1. Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=CONTEXTUALIZE_Q_PROMPT,
    )

    # 2. Create document question-answering chain
    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=QA_PROMPT,
    )

    # 3. Create full end-to-end conversational retrieval chain
    conversational_rag_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=question_answer_chain,
    )

    return conversational_rag_chain
