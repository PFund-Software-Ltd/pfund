from __future__ import annotations
from typing import Any, Literal

import os
from decimal import Decimal

from pfund.enums import CryptoExchange, Broker
from pfund.products.product_base import BaseProduct


class CryptoProduct(BaseProduct):
    broker: Broker = Broker.CRYPTO
    exchange: CryptoExchange
    category: str | None = None
    
    # EXTEND: may add taker_fee, maker_fee, multiplier
    # but the current problem is these values can't be obtained from apis consistently across exchanges,
    # and for fees, they can be different for different accounts,
    # so for now let users set them manually in e.g. strategy's config
    
    def model_post_init(self, __context: Any):
        super().model_post_init(__context)
        # FIXME
        # self._load_config()
    
    def _load_config(self):
        from pfund.config import get_config
        from pfund import print_warning
        from pfund.utils.utils import load_yaml_file
        config = get_config()
        file_path = f'{config.cache_path}/{self.exchange.lower()}/market_configs.yml'
        if not os.path.exists(file_path):
            return
        config = load_yaml_file(file_path)[self.category]
        if str(self) not in config:
            print_warning(
                f'{self} not found in {self.exchange} market configs,\n'
                f'configs such as tick_size and lot_size are not loaded.\n'
                f'Try to clear your market configs by running command:\n'
                f'    pfund clear cache --exch {self.exchange.lower()}\n'
            )
        else:
            self.tick_size = Decimal(config[str(self)]['tick_size'])
            self.lot_size = Decimal(config[str(self)]['lot_size'])

    # FIXME
    def get_fee(self, fee_type: Literal['taker', 'maker'], in_bps=False):
        if fee_type == 'taker':
            fee = self.taker_fee
        elif fee_type == 'maker':
            fee = self.maker_fee
        if not in_bps:
            fee /= 10000
        return fee
