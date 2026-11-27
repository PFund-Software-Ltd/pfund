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


class StaleArtifactError(PFundError):
    """Raised when a stored artifact was written by a different version of the component's code"""

    pass
