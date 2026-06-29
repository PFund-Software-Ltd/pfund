from typing import Any, NamedTuple, Required, NotRequired, TypeAlias, Callable
from typing_extensions import TypedDict  # need it to use "extra_items"

from http import HTTPMethod
from collections.abc import Sequence

from pfund.typing import AccountName
from pfund.enums import TradingVenue


EndpointName: TypeAlias = str
RawPayload: TypeAlias = dict[str, Any] | list[dict[str, Any]]
ParsingSequence: TypeAlias = Sequence[str | Callable[..., Any]]


class Endpoint(NamedTuple):
    method: HTTPMethod
    path: str


Schema = TypedDict(
    "Schema",
    {
        "ts": NotRequired[ParsingSequence],
        "@data": NotRequired[ParsingSequence],
        "data": Required[dict[str, Any]],
    },
    extra_items=Any,
)


class RequestData(TypedDict, total=True):
    endpoint_name: str
    url: str
    status_code: int | None
    params: dict[str, Any] | None


class ResponseData(TypedDict, total=False, extra_items=Any):
    # "data" is the key defined in schema
    ts: NotRequired[float]
    channel: NotRequired[str]  # for streaming only  (e.g. websocket channel)
    data: Required[dict[str, Any] | list[dict[str, Any]] | None]


class Result(TypedDict, total=True):
    success: bool
    error: str
    venue: TradingVenue
    request: RequestData
    account: AccountName | None
    response: ResponseData
    raw_payload: RawPayload | None
