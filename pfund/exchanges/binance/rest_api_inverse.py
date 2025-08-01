'''
Inverse (IPERP, IFUT) in PFund = COIN-M Futures in Binance
'''

from pfund.exchanges.rest_api_base import BaseRESTfulAPI


class RESTfulAPIInverse(BaseRESTfulAPI):
    URLS = {
        'PAPER': 'https://testnet.binancefuture.com',
        'LIVE': 'https://dapi.binance.com',
    }