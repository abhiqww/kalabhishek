from functools import lru_cache

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    EMBEDDING_MODEL,
    TOP_K,
)
from app.models import TradeMetadata


@lru_cache(maxsize=1)
def get_vector_db() -> Chroma:
    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        return Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Chroma: {exc}") from exc


def seed_chroma() -> None:
    try:
        db = get_vector_db()
        existing = db.get(ids=["DOC001", "DOC002"])
        if existing.get("ids"):
            return

        records = [
            {
                "text": (
                    "Shipment SHP1001 has invoice INV1001. "
                    "Incoterm is DAP. Payment term is NET30. "
                    "Invoice value is 10000 USD and declared value is 10000 USD."
                ),
                "metadata": TradeMetadata(
                    doc_id="DOC001",
                    doc_type="invoice",
                    shipment_id="SHP1001",
                    invoice_number="INV1001",
                    region="EMEA",
                    incoterm="DAP",
                    payment_term="NET30",
                    invoice_value=10000,
                    declared_value=10000,
                ).model_dump(),
            },
            {
                "text": (
                    "Shipment SHP1002 has invoice INV1002. "
                    "Incoterm is UNKNOWN. Payment term is MISSING. "
                    "Invoice value is 15000 USD and declared value is 10000 USD."
                ),
                "metadata": TradeMetadata(
                    doc_id="DOC002",
                    doc_type="invoice",
                    shipment_id="SHP1002",
                    invoice_number="INV1002",
                    region="APJ",
                    incoterm="UNKNOWN",
                    payment_term="MISSING",
                    invoice_value=15000,
                    declared_value=10000,
                ).model_dump(),
            },
        ]

        db.add_texts(
            texts=[item["text"] for item in records],
            metadatas=[item["metadata"] for item in records],
            ids=[item["metadata"]["doc_id"] for item in records],
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to seed Chroma: {exc}") from exc


def retrieve_top_k(query: str, k: int = TOP_K):
    try:
        db = get_vector_db()
        return db.similarity_search_with_score(query=query, k=k)
    except Exception as exc:
        raise RuntimeError(f"Top-K retrieval failed: {exc}") from exc


def find_shipment(shipment_id: str):
    try:
        db = get_vector_db()
        return db.similarity_search(
            query=shipment_id,
            k=3,
            filter={"shipment_id": shipment_id},
        )
    except Exception as exc:
        raise RuntimeError(f"Shipment lookup failed: {exc}") from exc
