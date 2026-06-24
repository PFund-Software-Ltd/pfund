from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias, TypedDict, cast

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
    from pfund.venues._apis.signers.base import BaseSigner
    from pfund.venues._apis.typing import Schema, Result, RawPayload, EndpointName

    URL: TypeAlias = str

from pfund.venues._apis.typing import Endpoint

import logging
import sys
import time
from datetime import datetime, UTC
from abc import ABC, abstractmethod
from json import JSONDecodeError
from pathlib import Path

from httpx2 import AsyncClient, HTTPStatusError, RequestError
from pfund_kit.utils.yaml import load, dump

from pfund.enums import Environment, TradingVenue
from pfund.errors import (
    AccountInSimulatedEnvDuringAPICallError,
    ParseAPIResponseError,
    PrivateAPICallInSandboxEnvError,
)
from pfund.venues._apis.schema_parser import SchemaParser


class BaseRESTfulAPI(ABC):
    venue: ClassVar[TradingVenue]
    adapter: ClassVar[BaseAdapter]
    _signer: ClassVar[BaseSigner[Any]]
    VERSION: ClassVar[str | None] = None  # e.g. "v5" for str
    URLS: ClassVar[
        dict[Literal[Environment.SANDBOX, Environment.PAPER, Environment.LIVE], URL]
    ] = {}
    PUBLIC_ENDPOINTS: ClassVar[dict[EndpointName, Endpoint]] = {}
    PRIVATE_ENDPOINTS: ClassVar[dict[EndpointName, Endpoint]] = {}

    class _Samples:
        """Maintainer-only: record/replay raw API payloads (dev mode)."""

        FILENAME: ClassVar[str] = "rest_api_samples.yml"

        class Record(TypedDict):
            recorded_at: str  # ISO-8601 UTC, when the sample was recorded
            payload: RawPayload

        def __init__(self, venue: TradingVenue):
            self._venue = venue

        @property
        def file_path(self) -> Path:
            from pfund.config import get_config

            return Path(get_config().data_path) / self._venue / self.FILENAME

        def load(self) -> dict[EndpointName, Record]:
            return cast(
                "dict[EndpointName, BaseRESTfulAPI._Samples.Record]",
                load(self.file_path) or {},
            )

        def get(self, endpoint_name: EndpointName) -> Record | None:
            return self.load().get(endpoint_name)

        def dump(self, endpoint_name: EndpointName, payload: RawPayload):
            samples = self.load()
            samples[endpoint_name] = {
                "recorded_at": datetime.now(UTC).isoformat(),
                "payload": payload,
            }
            dump(data=samples, file_path=self.file_path)

    def __init__(
        self,
        env: Literal[Environment.SANDBOX, Environment.PAPER, Environment.LIVE],
        dev_mode: bool = False,
    ):
        self._env = Environment[env.upper()]
        self._logger = logging.getLogger(f"pfund.{self.venue.lower()}")
        if self._env == Environment.SANDBOX:
            self._logger.warning(
                f"{self._env} environment will be using LIVE data for public endpoints"
            )
        elif self._env == Environment.BACKTEST:
            raise ValueError("BACKTEST environment is not supported")
        self._dev_mode = dev_mode
        if self._dev_mode:
            self.samples = self._Samples(self.venue)
            self._logger.warning(
                "DEV mode is enabled. Samples can be obtained by rest_api.samples.get(endpoint_name)\n"
                + "This mode is intended ONLY for internal development.\n"
                + "It should NEVER be used in production or by end users."
            )
        self._url: URL = self.URLS[self._env]
        self._client = AsyncClient()

    @abstractmethod
    def _is_success(self, endpoint_name: EndpointName, payload: RawPayload) -> bool: ...

    @abstractmethod
    def _extract_error(
        self, endpoint_name: EndpointName, payload: RawPayload
    ) -> str: ...

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
            self._signer.sign(
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
            if self._env == Environment.SANDBOX:
                raise PrivateAPICallInSandboxEnvError(
                    f'"{endpoint_name}" is a private endpoint and cannot be called in SANDBOX environment'
                )
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
            "data": None,
            "request": {
                "endpoint_name": endpoint_name,
                "url": str(request.url),
                "status_code": None,
                "params": dict(request.url.params) or None,
            },
            "raw_payload": None,
        }

    async def _request(
        self,
        schema: Schema,
        *,
        account: BaseAccount | None = None,
        params: QueryParamTypes | None = None,
        json: Any | None = None,
        data: RequestData | None = None,
        content: RequestContent | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
    ) -> Result:
        # the endpoint name is the calling method (e.g. "get_markets"), one frame up.
        # endpoint methods are named to match their PUBLIC/PRIVATE_ENDPOINTS key, so we
        # read it off the stack instead of making every endpoint pass it in by hand.
        if self._env.is_simulated() and account is not None:
            raise AccountInSimulatedEnvDuringAPICallError(
                f"Simulated environment {self._env} can only access public endpoints, account should NOT be provided"
            )
        endpoint_name: EndpointName = sys._getframe(1).f_code.co_name
        endpoint = self.get_endpoint(endpoint_name)
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
            payload: RawPayload = response.json()
            result["request"]["status_code"] = response.status_code
            result["raw_payload"] = payload
            result["success"] = response.is_success and self._is_success(
                endpoint_name, payload
            )
            if result["success"]:
                result["data"] = SchemaParser.convert(payload, schema)
            else:
                result["error"] = self._extract_error(endpoint_name, payload)
        except Exception as exc:
            expected = (
                ParseAPIResponseError,
                JSONDecodeError,
                RequestError,
                HTTPStatusError,
            )
            if isinstance(exc, expected):
                self._logger.exception(f'REST API "{endpoint_name}" error:')
            else:
                self._logger.exception(f'Unhandled REST API "{endpoint_name}" error:')
        finally:
            if self._dev_mode and result["raw_payload"] is not None:
                self.samples.dump(endpoint_name, result["raw_payload"])
        return result
