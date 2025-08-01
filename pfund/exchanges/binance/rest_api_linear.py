'''
Linear (PERP, FUT) in PFund = USDT-M Futures in Binance
'''
from pfund.exchanges.rest_api_base import BaseRESTfulAPI


class RESTfulAPILinear(BaseRESTfulAPI):
    URLS = {
        'PAPER': 'https://testnet.binancefuture.com',
        'LIVE': 'https://fapi.binance.com',
    }