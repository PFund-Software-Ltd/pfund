"""
Supports Binance's options trading
"""

from pfund.venues._apis.rest_api_base import BaseRestAPI


# TODO: it doesn't have testnet, can't test it
class BinanceRestAPIOption(BaseRestAPI):
    URLS = {
        "PAPER": None,
        "LIVE": "https://eapi.binance.com",
    }
