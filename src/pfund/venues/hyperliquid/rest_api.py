from pfund.venues._apis.rest_api_base import BaseRestAPI


class HyperliquidRestAPI(BaseRestAPI):
    PUBLIC_ENDPOINTS = {
        # "get_markets":   (RequestMethod.POST, "/info", {"type": "meta"}),
        # "get_positions": (RequestMethod.POST, "/info", {"type": "clearinghouseState"}),
    }
    PRIVATE_ENDPOINTS = {}
