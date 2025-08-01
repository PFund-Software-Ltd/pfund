'''
Inverse (IPERP, IFUT) in PFund = COIN-M Futures in Binance
'''

from pfund.exchanges.ws_api_base import BaseWebSocketAPI


class WebSocketAPIInverse(BaseWebSocketAPI):
    URLS = {
        'PAPER': 'wss://dstream.binancefuture.com',
        'LIVE': 'wss://dstream.binance.com',
    }