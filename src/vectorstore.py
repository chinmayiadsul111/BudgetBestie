"""
FAISS-backed retrieval over the financial SOP documents in data/sop_docs/.

Builds the index once (cached to disk at retrieval.index_path) and reuses it
on subsequent runs — rebuilding embeddings for a static doc set on every
request would be wasteful and, with API-based embeddings, costly.
"""
from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from src.config_loader import get_config, PROJECT_ROOT
from src.embeddings_provider import get_embeddings_model
from src.logging_setup import get_logger

logger = get_logger(__name__)

_VECTORSTORE_CACHE: FAISS | None = None


def _index_path() -> Path:
    cfg = get_config()
    return PROJECT_ROOT / cfg.get("retrieval.index_path", "data/faiss_index")


def build_or_load_index(force_rebuild: bool = False) -> FAISS:
    global _VECTORSTORE_CACHE
    if _VECTORSTORE_CACHE is not None and not force_rebuild:
        return _VECTORSTORE_CACHE

    cfg = get_config()
    embeddings = get_embeddings_model()
    index_path = _index_path()

    if index_path.exists() and not force_rebuild:
        logger.info("faiss_index_loaded_from_disk", extra={"path": str(index_path)})
        _VECTORSTORE_CACHE = FAISS.load_local(
            str(index_path), embeddings, allow_dangerous_deserialization=True
        )
        return _VECTORSTORE_CACHE

    docs_dir = PROJECT_ROOT / cfg.get("retrieval.sop_docs_dir", "data/sop_docs")
    loader = DirectoryLoader(str(docs_dir), glob="**/*.md", loader_cls=TextLoader)
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.get("retrieval.chunk_size", 500),
        chunk_overlap=cfg.get("retrieval.chunk_overlap", 75),
    )
    chunks = splitter.split_documents(raw_docs)

    logger.info("faiss_index_building", extra={"source_docs": len(raw_docs), "chunks": len(chunks)})
    vectorstore = FAISS.from_documents(chunks, embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    logger.info("faiss_index_saved", extra={"path": str(index_path)})

    _VECTORSTORE_CACHE = vectorstore
    return vectorstore


def retrieve(query: str) -> list[dict]:
    """
    Returns a list of {"content": str, "source": str, "score": float} dicts,
    filtered by similarity_threshold from config. An empty list is a
    meaningful signal to the Retrieval agent (low-confidence context available).
    """
    cfg = get_config()
    top_k = cfg.get("retrieval.top_k", 4)
    threshold = cfg.get("retrieval.similarity_threshold", 0.35)

    vectorstore = build_or_load_index()
    results_with_scores = vectorstore.similarity_search_with_relevance_scores(query, k=top_k)

    filtered = [
        {"content": doc.page_content, "source": doc.metadata.get("source", "unknown"), "score": score}
        for doc, score in results_with_scores
        if score >= threshold
    ]

    logger.info(
        "retrieval_completed",
        extra={"query": query, "results_returned": len(filtered), "results_raw": len(results_with_scores)},
    )
    return filtered
