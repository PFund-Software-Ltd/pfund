class PFundError(Exception):
    pass


class ParseRawResultError(PFundError):
    """Raised when parsing raw result from REST API or WebSocket API fails"""
    pass