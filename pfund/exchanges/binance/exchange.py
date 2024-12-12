from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.types.literals import tENVIRONMENT
    from pfund.products import BaseProduct

from pathlib import Path

from pfund.exchanges.exchange_base import BaseExchange
        
        
class Exchange(BaseExchange):
    SUPPORT_PLACE_BATCH_ORDERS = True
    SUPPORT_CANCEL_BATCH_ORDERS = True

    USE_WS_PLACE_ORDER = True
    USE_WS_CANCEL_ORDER = True

    # TODO
    # _MAX_NUM_OF_PLACE_BATCH_ORDERS = ...
    # _MAX_NUM_OF_CANEL_BATCH_ORDERS = ...
    
    def __init__(self, env: tENVIRONMENT, refetch_market_configs=False):
        exch = Path(__file__).parent.name
        super().__init__(env, exch, refetch_market_configs=refetch_market_configs)
    
    def create_external_product_name(self, product: BaseProduct) -> str:
        pass

        