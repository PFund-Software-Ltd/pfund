class PFundError(Exception):
    pass


class ParseAPIResponseError(PFundError):
    """Raised when parsing raw result from REST API or WebSocket API fails"""

    pass


class AccountInSimulatedEnvDuringAPICallError(PFundError):
    """Raised when account is provided in simulated environment during an API call"""

    pass


class PrivateAPICallInSandboxEnvError(PFundError):
    """Raised when a private API call is made in sandbox environment"""

    pass
