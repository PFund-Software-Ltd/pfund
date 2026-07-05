from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    TypeAlias,
    TypedDict,
    cast,
    Generic,
)

if TYPE_CHECKING:
    from httpx2 import Request, Response
    from httpx2._types import (
        QueryParamTypes,
        RequestData,
        RequestContent,
        HeaderTypes,
        CookieTypes,
    )
    from pfund.entities import BaseAccount
    from pfund.venues.adapter_base import BaseAdapter
    from pfund.venues.crypto_exchange import CryptoExchangeSigner
    from pfund.venues._apis.typing import Schema, Result, RawResponse, EndpointName

    URL: TypeAlias = str

import logging
import sys
import time
from datetime import datetime, UTC
from abc import ABC, abstractmethod
from json import JSONDecodeError
from pathlib import Path
from http import HTTPMethod

from httpx2 import AsyncClient, HTTPStatusError, RequestError
from pfund_kit.utils.yaml import load, dump

from pfund.venues.venue_base import ConfigT
from pfund.venues._apis.typing import Endpoint
from pfund.enums import Environment, TradingVenue
from pfund.errors import ResponseParseError
from pfund.venues._apis.schema_parser import SchemaParser


class BaseRestAPI(ABC, Generic[ConfigT]):
    venue: ClassVar[TradingVenue]
    _signer: ClassVar[CryptoExchangeSigner[Any]]
    VERSION: ClassVar[str | None] = None  # e.g. "v5" for str
    URLS: ClassVar[dict[Literal[Environment.PAPER, Environment.LIVE], URL]] = {}
    # NOTE: EndpointName should match with function names
    PUBLIC_ENDPOINTS: ClassVar[dict[EndpointName, Endpoint]] = {}
    PRIVATE_ENDPOINTS: ClassVar[dict[EndpointName, Endpoint]] = {}

    class _Samples:
        """Maintainer-only: record/replay raw API payloads (dev mode)."""

        FILENAME: ClassVar[str] = "rest_api_samples.yml"

        class Record(TypedDict):
            recorded_at: str  # ISO-8601 UTC, when the sample was recorded
            payload: RawResponse

        def __init__(self, venue: TradingVenue):
            self._venue = venue

        @property
        def file_path(self) -> Path:
            from pfund.config import get_config

            return Path(get_config().data_path) / self._venue / self.FILENAME

        def load(self) -> dict[EndpointName, Record]:
            return cast(
                "dict[EndpointName, BaseRestAPI._Samples.Record]",
                load(self.file_path) or {},
            )

        def get(self, endpoint_name: EndpointName) -> Record | None:
            return self.load().get(endpoint_name)

        def dump(self, endpoint_name: EndpointName, payload: RawResponse):
            samples = self.load()
            samples[endpoint_name] = {
                "recorded_at": datetime.now(UTC).isoformat(),
                "payload": payload,
            }
            dump(data=samples, file_path=self.file_path)

    def __init__(
        self,
        env: Literal[Environment.PAPER, Environment.LIVE, "PAPER", "LIVE"],
        config: ConfigT | None = None,
        read_only: bool = False,
        dev_mode: bool = False,
    ):
        self._env = Environment[env.upper()]
        if self._env.is_simulated():
            raise ValueError(f"environment {self._env} is not supported")
        self._logger = logging.getLogger(f"pfund.{self.venue.lower()}")
        self._config: ConfigT = config or cast(ConfigT, self.venue.venue_class.Config())
        self._read_only = read_only
        self._dev_mode = dev_mode
        if self._dev_mode:
            self.samples = self._Samples(self.venue)
            self._logger.warning(
                "DEV mode is enabled. Samples can be obtained by rest_api.samples.get(endpoint_name)\n"
                + "This mode is intended ONLY for internal development.\n"
                + "It should NEVER be used in production or by end users."
            )
        self._url: URL = self.URLS[self.env]
        self._client = AsyncClient()

    @property
    def env(self) -> Literal[Environment.PAPER, Environment.LIVE]:
        return cast(Literal[Environment.PAPER, Environment.LIVE], self._env)

    @abstractmethod
    def _is_success(
        self, endpoint_name: EndpointName, payload: RawResponse
    ) -> bool: ...

    @abstractmethod
    def _extract_error(
        self, endpoint_name: EndpointName, payload: RawResponse
    ) -> str: ...

    @abstractmethod
    async def get_balances(self, account: BaseAccount) -> Result: ...

    def __getstate__(self) -> dict[str, Any]:
        # NOTE: drop the unpicklable client so the object survives Ray/cloudpickle;
        # __setstate__ rebuilds a fresh one in the destination process.
        state = self.__dict__.copy()
        del state["_client"]
        return state

    def __setstate__(self, state: dict[str, Any]):
        self.__dict__.update(state)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        self._client = AsyncClient()

    @property
    def nonce(self) -> int:
        return int(time.time() * 1000)

    @property
    def adapter(self) -> BaseAdapter:
        VenueClass = self.venue.venue_class
        return VenueClass.adapter

    @staticmethod
    def _convert_ms_to_seconds(ms: int | str) -> float:
        return int(ms) / 1000

    def _build_request(
        self,
        endpoint: Endpoint,
        *,
        account: BaseAccount | None = None,
        params: QueryParamTypes | None = None,
        json: Any | None = None,
        data: RequestData | None = None,
        content: RequestContent | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
    ) -> Request:
        method = endpoint.method
        if self._read_only and method != HTTPMethod.GET:
            raise RuntimeError(
                f"{self.venue} REST API is read-only — {method} requests are disabled"
            )
        url = self._url + endpoint.path
        # NOTE: the signer mutates params/json/data/headers in place, so signed requests
        # need a concrete dict (the broad HeaderTypes forms are only for unsigned requests).
        if account is not None:
            if headers is None:
                headers = {}
            elif not isinstance(headers, dict):
                raise TypeError(
                    f"signed requests require headers as a dict[str, str], got {type(headers).__name__}"
                )
            # the isinstance check narrows to dict but can't inspect key/value types;
            # we require str keys/values for signing, so assert that to the type checker.
            headers = cast("dict[str, str]", headers)
            self._signer.sign_rest_api(
                account=account,
                method=method,
                url=url,
                params=params,
                json=json,
                data=data,
                headers=headers,
            )
        return self._client.build_request(
            method=method,
            url=url,
            params=params,
            json=json,
            data=data,
            content=content,
            headers=headers,
            cookies=cookies,
        )

    def get_endpoint(self, endpoint_name: EndpointName) -> Endpoint:
        if endpoint_name in self.PUBLIC_ENDPOINTS:
            endpoint = self.PUBLIC_ENDPOINTS[endpoint_name]
        elif endpoint_name in self.PRIVATE_ENDPOINTS:
            endpoint = self.PRIVATE_ENDPOINTS[endpoint_name]
        else:
            raise ValueError(
                f'"{endpoint_name}" is not a registered endpoint; it must match a key in '
                + "PUBLIC_ENDPOINTS or PRIVATE_ENDPOINTS"
            )
        return endpoint

    def _create_result(
        self,
        endpoint_name: str,
        request: Request,
        *,
        account: BaseAccount | None = None,
    ) -> Result:
        return {
            "success": False,
            "error": "",
            "venue": self.venue,
            "account": account.name if account else None,
            "response": {"data": None},
            "request": {
                "endpoint_name": endpoint_name,
                "url": str(request.url),
                "status_code": None,
                "params": dict(request.url.params) or None,
            },
            "raw_response": None,
        }

    async def request(
        self,
        method: HTTPMethod | str,
        endpoint_path: str,
        *,
        schema: Schema | None = None,
        account: BaseAccount | None = None,
        params: QueryParamTypes | None = None,
        json: Any | None = None,
        data: RequestData | None = None,
        content: RequestContent | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
    ) -> Result | Response:
        """Call an arbitrary venue endpoint not registered in PUBLIC_ENDPOINTS/PRIVATE_ENDPOINTS.

        The named API methods (e.g. ``place_order``, ``get_positions``) resolve their
        ``Endpoint`` by looking it up in the endpoint registries. This is the dynamic
        escape hatch for endpoints that don't have a first-class method yet: the
        ``Endpoint`` is built on the fly from ``method`` + ``endpoint_path`` instead.

        Two modes, selected by ``schema``:
          - schema given: routed through ``_request``, parsed/validated against the
            schema and returned as a structured ``Result`` (same path as named methods).
          - schema omitted: the raw httpx ``Response`` is returned untouched for the
            caller to handle.

        Args:
            method: HTTP method, as an ``HTTPMethod`` or a case-insensitive string.
            endpoint_path: URL path appended to the venue base URL (e.g. "/v5/order/create").
            schema: parse the response into a ``Result`` when provided; otherwise return
                the raw ``Response``.
            account: when provided, the request is signed for that account (private endpoint).
            params, json, data, content, headers, cookies: forwarded to the underlying request.
        """
        endpoint = Endpoint(method=HTTPMethod[method.upper()], path=endpoint_path)
        if schema:
            result: Result = await self._request(
                schema=schema,
                endpoint=endpoint,
                account=account,
                params=params,
                json=json,
                data=data,
                content=content,
                headers=headers,
                cookies=cookies,
            )
            return result
        else:
            request: Request = self._build_request(
                endpoint=endpoint,
                account=account,
                params=params,
                json=json,
                data=data,
                content=content,
                headers=headers,
                cookies=cookies,
            )
            response: Response = await self._client.send(request)
            return response

    async def _request(
        self,
        schema: Schema,
        *,
        endpoint: Endpoint | None = None,
        account: BaseAccount | None = None,
        params: QueryParamTypes | None = None,
        json: Any | None = None,
        data: RequestData | None = None,
        content: RequestContent | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
    ) -> Result:
        """Build, send, and parse a request into a structured ``Result``.

        ``endpoint`` resolution has two modes:
          - endpoint is None (named-method path): the caller is a named API method
            (e.g. ``place_order``) that doesn't pass an endpoint. We recover its name
            from the calling frame (``sys._getframe(1).f_code.co_name``) and look it up
            in the registries via ``get_endpoint``. This relies on the convention that
            an endpoint's registry key matches the method name (see PUBLIC/PRIVATE_ENDPOINTS).
            NOTE: this is frame-name coupling — an intervening frame (decorator, wrapper)
            between the named method and this call would break the lookup.
          - endpoint given (dynamic path): callers like ``request`` build the ``Endpoint``
            themselves for endpoints with no first-class method, bypassing the registry.
            ``endpoint_name`` is set to the path here since there's no registry key; it is
            used for logging/samples only, not for control flow.
        """
        if account:
            if account.env != self._env:
                raise ValueError(
                    f"Account environment {account.env} does not match RestAPI environment {self._env}"
                )
            if account.venue != self.venue:
                raise ValueError(
                    f"Account venue {account.venue} does not match RestAPI venue {self.venue}"
                )
        if endpoint is None:
            endpoint_name: EndpointName = sys._getframe(1).f_code.co_name
            endpoint = self.get_endpoint(endpoint_name)
        else:
            endpoint_name = endpoint.path
        request: Request = self._build_request(
            endpoint=endpoint,
            account=account,
            params=params,
            json=json,
            data=data,
            content=content,
            headers=headers,
            cookies=cookies,
        )
        result: Result = self._create_result(endpoint_name, request, account=account)
        try:
            response: Response = await self._client.send(request)
            payload: RawResponse = response.json()
            result["request"]["status_code"] = response.status_code
            result["raw_response"] = payload
            is_response_success = response.is_success
            result["success"] = is_response_success and self._is_success(
                endpoint_name, payload
            )
            if result["success"]:
                result["response"] = SchemaParser.convert(payload, schema)
            else:
                error = self._extract_error(endpoint_name, payload)
                if not is_response_success:
                    http_error = f"HTTP {response.status_code} {response.reason_phrase}"
                    error = f"{error} ({http_error})" if error else http_error
                result["error"] = error
        except ResponseParseError:
            result["success"] = False
            result["error"] = "Response parse error"
            self._logger.exception(f'REST API "{endpoint_name}" parse error:')
        except Exception as exc:
            expected = (
                JSONDecodeError,
                RequestError,
                HTTPStatusError,
            )
            if isinstance(exc, expected):
                self._logger.exception(f'REST API "{endpoint_name}" error:')
            else:
                self._logger.exception(f'Unhandled REST API "{endpoint_name}" error:')
        finally:
            self._logger.debug(
                f'REST API "{endpoint_name}" raw response: {result["raw_response"]}'
            )
            if self._dev_mode and result["raw_response"] is not None:
                self.samples.dump(endpoint_name, result["raw_response"])
        return result
