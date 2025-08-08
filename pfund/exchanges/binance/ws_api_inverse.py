'''
Inverse (IPERP, IFUT) in PFund = COIN-M Futures in Binance
'''

from pfund.exchanges.ws_api_base import BaseWebSocketAPI


class InverseWebSocketAPI(BaseWebSocketAPI):
    URLS = {
        'PAPER': 'wss://dstream.binancefuture.com',
        'LIVE': 'wss://dstream.binance.com',
    }