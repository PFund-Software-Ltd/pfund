import hashlib
import hmac


def hmac_sha256(secret: str, message: str) -> str:
    return hmac.new(
        secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def add_oauth_header(headers: dict[str, str], token: str) -> None:
    """OAuth2 bearer auth: attach ``Authorization: Bearer <token>`` in place.

    Identical across the US brokers (Tradier, TradeStation, Schwab, Tradovate).
    """
    headers["Authorization"] = f"Bearer {token}"
