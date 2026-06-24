from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic

if TYPE_CHECKING:
    from http import HTTPMethod
    from httpx2._types import QueryParamTypes, RequestData, HeaderTypes

import time
from abc import ABC, abstractmethod

from pfund.typing import AccountT


class BaseSigner(ABC, Generic[AccountT]):
    """Injects venue authentication into an outgoing request.

    A signer mutates ``params``/``json``/``data``/``headers`` in place so the signature is
    computed over the exact bytes that get sent (sign-must-match-send). Venues sign into
    different places: Bybit/OKX add headers, Binance/Aster append a ``signature`` to the
    params, Hyperliquid adds a signature object to the body.

    ``params`` is the query string; ``json`` and ``data`` are the JSON and form bodies (a
    venue uses one or the other, so the unused one is ``None``).

    Parameterized by the account type it reads auth material from: HMAC venues bind to an
    ``APIKeyAccount`` (key/secret), OAuth brokers to a token account, Hyperliquid to a wallet.
    """

    @property
    def nonce(self) -> int:
        return int(time.time() * 1000)

    @abstractmethod
    def sign(
        self,
        account: AccountT,
        method: HTTPMethod,
        url: str,
        *,
        params: QueryParamTypes | None = None,
        json: Any | None = None,
        data: RequestData | None = None,
        headers: dict[str, str],
    ) -> None: ...
