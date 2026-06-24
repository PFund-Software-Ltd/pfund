"""
Linear (PERP, FUT) in PFund = USDT-M Futures in Binance
"""

from pfund.venues._apis.rest_api_base import BaseRESTfulAPI


class BinanceRESTfulAPILinear(BaseRESTfulAPI):
    URLS = {
        "PAPER": "https://testnet.binancefuture.com",
        "LIVE": "https://fapi.binance.com",
    }
