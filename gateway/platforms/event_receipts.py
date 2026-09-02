"""Successful-turn receipts for durable selective process wakes.

Admission alone is not successful processing: the outbox survives until the
runner has persisted a completed turn. Unknown exits stay retryable.
"""

_GATEWAY_DELIVERY_RECEIPT_ATTR = "_gateway_durable_delivery_receipt"


def has_gateway_delivery_receipt(event) -> bool:
    return getattr(event, _GATEWAY_DELIVERY_RECEIPT_ATTR, None) is not None


def resolve_gateway_delivery_receipt(event, outcome: str) -> bool:
    receipt = getattr(event, _GATEWAY_DELIVERY_RECEIPT_ATTR, None)
    if receipt is None or receipt.done():
        return False
    receipt.set_result(outcome)
    return True
