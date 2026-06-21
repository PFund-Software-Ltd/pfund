"""
Supports Binance's options trading
"""

from pfund._apis.ws_api_base import BaseWebSocketAPI


# TODO: it doesn't have testnet, can't test it
class BinanceOptionWebSocketAPI(BaseWebSocketAPI):
    URLS = {"PAPER": None, "LIVE": "wss://nbstream.binance.com/eoptions"}
