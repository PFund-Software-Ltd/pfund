"""
Inverse (IPERP, IFUT) in PFund = COIN-M Futures in Binance
"""

from pfund._apis.ws_api_base import BaseWebSocketAPI


class BinanceInverseWebSocketAPI(BaseWebSocketAPI):
    URLS = {
        "PAPER": "wss://dstream.binancefuture.com",
        "LIVE": "wss://dstream.binance.com",
    }
