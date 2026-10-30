class PFundError(Exception):
    pass


class WebSocketTimeoutError(PFundError):
    """Raised when a WebSocket connection times out"""

    pass


class NotSupportedByVenueError(PFundError):
    """Raised when a venue does not support the requested feature"""

    pass


class MissingSymbolError(PFundError):
    """Raised when a symbol must be provided but is missing for product"""

    pass
