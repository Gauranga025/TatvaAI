"""
core/rag_engine.py — RAG pipeline (ChromaDB + Mistral)
───────────────────────────────────────────────────────
Key design decisions
────────────────────
• EphemeralClient (in-memory ChromaDB) — no persistent disk storage,
  no cross-session collisions, Streamlit Cloud compatible.

• Singleton embeddings model via _get_embeddings() — loaded once per
  Streamlit server process via @st.cache_resource in app.py; this
  module just returns the shared instance.

• Centralised Mistral auth — uses core/mistral_client.py for key
  validation and LLM instantiation so errors are caught early.

• Graceful failure — every public function raises either MistralAuthError
  (bad key) or RuntimeError (all other failures) with a friendly message.
"""

from __future__ import annotations

import logging
import uuid

from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from core.mistral_client import MistralAuthError, classify_api_error, get_mistral_llm

logger = logging.getLogger(__name__)

# ─── Embedding model (lazy singleton) ───────────────────────────────────────────

_embeddings_instance = None


def _get_embeddings():
    """
    Return the HuggingFace sentence-transformer embedding model.

    The first call loads the model (~90 MB); subsequent calls return
    the cached instance immediately.  app.py also calls this via
    @st.cache_resource so the Streamlit process shares one copy.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: PLC0415

        logger.info("[rag_engine] Loading HuggingFace embedding model…")
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        logger.info("[rag_engine] Embedding model ready")
    return _embeddings_instance


# ─── RAG prompt ─────────────────────────────────────────────────────────────────

_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful meeting assistant. Answer the question below
using ONLY the context from the meeting transcript provided.

If the answer is not in the context, say:
  "I couldn't find that information in the meeting transcript."

Context:
{context}

Question: {question}

Answer:""",
)


# ─── Internal helpers ────────────────────────────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    """Concatenate retrieved document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


# ─── Public API ──────────────────────────────────────────────────────────────────

def build_rag_chain(transcript: str):
    """
    Index *transcript* in an in-memory ChromaDB and return a RAG chain.

    Steps:
      1. Split transcript into overlapping chunks.
      2. Embed chunks with sentence-transformers.
      3. Store in an ephemeral (in-memory) Chroma collection.
      4. Build a retrieval-augmented chain (retriever | prompt | LLM | parser).

    Args:
        transcript: Full meeting transcript text.

    Returns:
        A LangChain runnable chain that accepts {"question": str}.

    Raises:
        MistralAuthError: API key is missing / invalid.
        RuntimeError:     Indexing or chain construction failed.
    """
    if not transcript or not transcript.strip():
        raise ValueError("Cannot build RAG chain — transcript is empty.")

    logger.info("[build_rag_chain] Indexing %d-char transcript", len(transcript))

    # ── 1. Validate Mistral key before doing any expensive work ─────────────
    # get_mistral_llm() raises MistralAuthError if the key is bad.
    llm = get_mistral_llm(context="build_rag_chain")

    # ── 2. Chunk the transcript ──────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_text(transcript)
    docs   = [Document(page_content=chunk) for chunk in chunks]
    logger.info("[build_rag_chain] %d chunks created", len(docs))

    # ── 3. Embed + store in ephemeral Chroma ────────────────────────────────
    try:
        import chromadb  # noqa: PLC0415

        # Unique collection name per run so parallel Streamlit sessions
        # don't share or corrupt each other's index.
        collection_name = f"tatvaai_{uuid.uuid4().hex[:8]}"

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=_get_embeddings(),
            client=chromadb.EphemeralClient(),
            collection_name=collection_name,
        )
        logger.info("[build_rag_chain] Chroma collection '%s' ready", collection_name)
    except Exception as exc:
        msg = f"Failed to build vector index: {exc}"
        logger.error("[build_rag_chain] %s", exc)
        raise RuntimeError(msg) from exc

    # ── 4. Build retrieval chain ─────────────────────────────────────────────
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | _RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    logger.info("[build_rag_chain] RAG chain built successfully")
    return chain


def ask_question(rag_chain, question: str) -> str:
    """
    Run *question* through the RAG chain and return the answer.

    Args:
        rag_chain: Chain returned by build_rag_chain().
        question:  User's natural-language question.

    Returns:
        Answer string from Mistral.

    Raises:
        MistralAuthError: API key revoked between sessions.
        RuntimeError:     LLM call or retrieval failed.
    """
    if not question or not question.strip():
        return "Please enter a question."

    logger.info("[ask_question] Question: %s", question[:120])

    try:
        answer = rag_chain.invoke(question)
        logger.info("[ask_question] Answer length: %d chars", len(answer))
        return answer
    except MistralAuthError:
        raise
    except Exception as exc:
        friendly = classify_api_error(exc)
        logger.error("[ask_question] Failed: %s", exc)
        raise RuntimeError(friendly) from exc
