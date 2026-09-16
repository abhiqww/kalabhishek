import json

from langchain.tools import tool

from app.config import RERANK_TOP_N, TOP_K
from app.reranker import rerank_documents
from app.vector_store import find_shipment, retrieve_top_k


@tool
def search_trade_documents(query: str) -> str:
    """Search stored shipment/invoice data using Chroma Top-K plus reranking."""
    try:
        candidates_with_scores = retrieve_top_k(query=query, k=TOP_K)
        candidate_docs = [doc for doc, _ in candidates_with_scores]
        reranked = rerank_documents(
            query=query,
            documents=candidate_docs,
            top_n=RERANK_TOP_N,
        )

        results = []
        for rank, (doc, rerank_score) in enumerate(reranked, start=1):
            results.append(
                {
                    "rank": rank,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "rerank_score": rerank_score,
                }
            )

        return json.dumps(
            {
                "status": "success",
                "retrieval_top_k": TOP_K,
                "rerank_top_n": RERANK_TOP_N,
                "results": results,
            },
            default=str,
        )
    except Exception as exc:
        return json.dumps(
            {
                "status": "error",
                "message": "Trade document retrieval failed.",
                "error_type": type(exc).__name__,
            }
        )


@tool
def check_trade_compliance(shipment_id: str) -> str:
    """Check a shipment against basic deterministic trade-compliance rules."""
    try:
        docs = find_shipment(shipment_id)

        if not docs:
            return json.dumps(
                {
                    "status": "not_found",
                    "shipment_id": shipment_id,
                    "message": "Shipment was not found.",
                }
            )

        metadata = docs[0].metadata
        issues = []

        allowed_incoterms = {
            "EXW", "FCA", "CPT", "CIP", "DAP", "DPU", "DDP",
            "FAS", "FOB", "CFR", "CIF",
        }

        incoterm = str(metadata.get("incoterm", "")).upper()
        payment_term = str(metadata.get("payment_term", "")).upper()
        invoice_value = float(metadata.get("invoice_value", 0))
        declared_value = float(metadata.get("declared_value", 0))

        if incoterm not in allowed_incoterms:
            issues.append({
                "rule": "INCOTERM_RULE",
                "message": "Invalid or missing Incoterm.",
            })

        if payment_term in {"", "MISSING", "UNKNOWN"}:
            issues.append({
                "rule": "PAYMENT_TERM_RULE",
                "message": "Payment term is missing.",
            })

        if invoice_value != declared_value:
            issues.append({
                "rule": "VALUE_MATCH_RULE",
                "message": (
                    f"Invoice value {invoice_value} does not match "
                    f"declared value {declared_value}."
                ),
            })

        compliance_status = "BLOCK" if issues else "PASS"

        return json.dumps(
            {
                "status": "success",
                "shipment_id": shipment_id,
                "compliance_status": compliance_status,
                "issues": issues,
                "metadata": metadata,
            },
            default=str,
        )
    except Exception as exc:
        return json.dumps(
            {
                "status": "error",
                "shipment_id": shipment_id,
                "message": "Compliance check failed.",
                "error_type": type(exc).__name__,
            }
        )


TOOLS = [
    search_trade_documents,
    check_trade_compliance,
]
