from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund._typing import tEnvironment
    from pfund.products.product_base import BaseProduct

from pathlib import Path

from pfund.exchanges.exchange_base import BaseExchange
        
        
class Exchange(BaseExchange):
    SUPPORT_PLACE_BATCH_ORDERS = True
    SUPPORT_CANCEL_BATCH_ORDERS = True

    USE_WS_PLACE_ORDER = True
    USE_WS_CANCEL_ORDER = True

    # TODO
    # MAX_NUM_OF_PLACE_BATCH_ORDERS = ...
    # MAX_NUM_OF_CANCEL_BATCH_ORDERS = ...
    
    def __init__(self, env: tEnvironment, fetch_market_configs=False):
        exch = Path(__file__).parent.name
        super().__init__(env, exch, fetch_market_configs=fetch_market_configs)
    
    def create_external_product_name(self, product: BaseProduct) -> str:
        pass

        