from pathlib import Path

from pfund.exchanges.rest_api_base import BaseRestApi


# TODO:
class RestApi(BaseRestApi):
    URLS = {}
    PUBLIC_ENDPOINTS = {}
    PRIVATE_ENDPOINTS = {}
    
    def __init__(self, env):
        exch = Path(__file__).parent.name
        super().__init__(env, exch)
        