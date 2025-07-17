class PFundError(Exception):
    pass


class ParseApiResponseError(PFundError):
    """Raised when parsing raw result from REST API or WebSocket API fails"""
    pass