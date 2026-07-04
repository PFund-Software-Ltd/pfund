class PFundError(Exception):
    pass


class ResponseParseError(PFundError):
    """Raised when parsing raw result from REST API or WebSocket API fails"""

    pass


class WebSocketTimeoutError(PFundError):
    """Raised when a WebSocket connection times out"""

    pass


class NotSupportedByVenueError(PFundError):
    """Raised when a venue does not support the requested feature"""

    pass


class InteractiveBrokersError(PFundError):
    """Raised when Interactive Brokers reports an error for a pending API request.

    ibapi never raises on the request/response path — failures arrive through the
    EWrapper.error(reqId, errorCode, ...) callback (and ibapi.errors only defines
    CodeMsgPair constants, not exceptions) — so this is the exception surfaced to
    the awaiter when a request fails.
    """

    pass


class MissingSymbolError(PFundError):
    """Raised when a symbol must be provided but is missing for product"""

    pass
