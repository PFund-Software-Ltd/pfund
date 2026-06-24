from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from http import HTTPMethod
    from httpx2._types import QueryParamTypes, RequestData, HeaderTypes
    from pfund.entities.accounts.account_api import APIKeyAccount

import urllib.parse

# aliased: the `json` request kwarg below would otherwise shadow this module
from msgspec import json as msgspec_json

from pfund.venues._apis.signers.hmac import HmacSigner


class BybitSigner(HmacSigner["APIKeyAccount"]):
    # max time (ms) after the request timestamp that Bybit will still accept the request;
    # guards against replayed/delayed requests. 5000ms is Bybit's default.
    RECV_WINDOW: ClassVar[str] = "5000"  # in ms

    def sign(
        self,
        account: APIKeyAccount,
        method: HTTPMethod,
        url: str,
        *,
        params: QueryParamTypes | None = None,
        json: Any | None = None,
        data: RequestData | None = None,
        headers: dict[str, str],
    ) -> None:
        timestamp = str(self.nonce)
        # Bybit signs over the exact payload sent: raw JSON body for writes,
        # url-encoded query params for reads.
        body = json if json is not None else data
        payload_str = (
            msgspec_json.encode(body).decode()
            if body
            else urllib.parse.urlencode(params)
        )
        prehash = timestamp + account.key + self.RECV_WINDOW + payload_str
        signature = self._hmac_sha256(account.secret, prehash)
        headers.update(
            {
                "Content-Type": "application/json",
                "X-BAPI-API-KEY": account.key,
                "X-BAPI-SIGN": signature,
                "X-BAPI-SIGN-TYPE": "2",
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": self.RECV_WINDOW,
            }
        )
