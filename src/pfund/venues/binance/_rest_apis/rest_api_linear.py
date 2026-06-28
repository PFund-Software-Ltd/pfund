"""
Linear (PERP, FUT) in PFund = USDT-M Futures in Binance
"""

from pfund.venues._apis.rest_api_base import BaseRestAPI


class BinanceRestAPILinear(BaseRestAPI):
    URLS = {
        "PAPER": "https://testnet.binancefuture.com",
        "LIVE": "https://fapi.binance.com",
    }
