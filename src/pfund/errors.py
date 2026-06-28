class PFundError(Exception):
    pass


class ResponseParseError(PFundError):
    """Raised when parsing raw result from REST API or WebSocket API fails"""

    pass


class WebSocketTimeoutError(PFundError):
    """Raised when a WebSocket connection times out"""

    pass
