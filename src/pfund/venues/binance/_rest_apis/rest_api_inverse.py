"""
Inverse (IPERP, IFUT) in PFund = COIN-M Futures in Binance
"""

from pfund.venues._apis.rest_api_base import BaseRestAPI


class BinanceRestAPIInverse(BaseRestAPI):
    URLS = {
        "PAPER": "https://testnet.binancefuture.com",
        "LIVE": "https://dapi.binance.com",
    }
