from functools import lru_cache
from typing import Sequence

from langchain_core.documents import Document

from app.config import (
    RERANKER_ENABLED,
    RERANKER_MODEL,
    RERANK_TOP_N,
)


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_MODEL)


def rerank_documents(
    query: str,
    documents: Sequence[Document],
    top_n: int = RERANK_TOP_N,
) -> list[tuple[Document, float | None]]:
    if not documents:
        return []

    if not RERANKER_ENABLED:
        return [(doc, None) for doc in documents[:top_n]]

    try:
        reranker = get_reranker()
        pairs = [(query, doc.page_content) for doc in documents]
        scores = reranker.predict(pairs)
        ranked = sorted(
            zip(documents, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [(doc, float(score)) for doc, score in ranked[:top_n]]
    except Exception:
        return [(doc, None) for doc in documents[:top_n]]
