from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from http import HTTPMethod
    from httpx2._types import QueryParamTypes, RequestData, HeaderTypes
    from pfund.entities.accounts.account_oauth import OAuthAccount

from pfund.venues._apis.signers.base import BaseSigner


class OAuthBearerSigner(BaseSigner["OAuthAccount"]):
    """Shared signer for venues that authenticate with an OAuth2 bearer token.

    Identical across all four US brokers (Tradier, TradeStation, Schwab, Tradovate), so it
    lives in the shared signers folder rather than per-venue. No per-request signing — the
    access token is simply attached as an ``Authorization: Bearer`` header.
    """

    def sign(
        self,
        account: OAuthAccount,
        method: HTTPMethod,
        url: str,
        *,
        params: QueryParamTypes | None = None,
        json: Any | None = None,
        data: RequestData | None = None,
        headers: dict[str, str],
    ) -> None:
        headers["Authorization"] = f"Bearer {account.token}"
