"""
Inverse (IPERP, IFUT) in PFund = COIN-M Futures in Binance
"""

from pfund.venues._apis.rest_api_base import BaseRESTfulAPI


class BinanceRESTfulAPIInverse(BaseRESTfulAPI):
    URLS = {
        "PAPER": "https://testnet.binancefuture.com",
        "LIVE": "https://dapi.binance.com",
    }
