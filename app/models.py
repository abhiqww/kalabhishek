from typing import Literal
from pydantic import BaseModel


class TradeMetadata(BaseModel):
    doc_id: str
    doc_type: Literal["shipment", "invoice", "rule"]
    shipment_id: str
    invoice_number: str
    region: str
    incoterm: str
    payment_term: str
    invoice_value: float
    declared_value: float
