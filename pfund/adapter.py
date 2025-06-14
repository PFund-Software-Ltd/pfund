from typing import Literal, TypeAlias, Any

from collections import defaultdict
from pathlib import Path

from pfund.enums import TradingVenue
from pfund.utils.utils import load_yaml_file


# NOTE: DynamicGroup can be used to specify a group that is not defined in adapter.yml
# e.g. for Bybit, it uses product category ('spot', 'linear', 'inverse', 'option', etc.) for grouping
# to achieve converting e.g. TODO ... -> BTCUSDH25
DynamicGroup: TypeAlias = str

tADAPTER_GROUP = DynamicGroup | Literal[
    # defined in adapter.yml
    'asset',
    'asset_type',
    'option_type',
    'order_type',
    'side',
    'tif',
    'order_status',
    'offset',
    'price_direction',
    'channel',
    'resolution',
] 


class Adapter:
    FILENAME = 'adapter.yml'
    
    def __init__(self, trading_venue: str, is_strict: bool=False):
        '''
        Args:
            is_strict: if False, it will search for the same key in other groups if group is not specified
        '''
        self._trading_venue = TradingVenue[trading_venue.upper()]
        self._adapter = defaultdict(dict)
        self._is_strict = is_strict
        self._load_config(self.get_file_path())
    
    def __str__(self):
        import json
        # only show (key: value) (one-sided), no need to show (value: key)
        one_sided_mappings = {}
        for group, mappings in self._adapter.items():
            if group not in one_sided_mappings:
                one_sided_mappings[group] = {}
            for k, v in mappings.items():
                if k not in one_sided_mappings[group].values():
                    one_sided_mappings[group][k] = v
        return json.dumps(one_sided_mappings, indent=4)
    
    @property
    def groups(self) -> list[str]:
        return list(self._adapter.keys())
    
    def get_file_path(self) -> Path:
        '''Gets the file path of the adapter.yml'''
        from pfund.const.paths import PROJ_PATH
        from pfund.enums import CryptoExchange
        tv_type = 'exchanges' if self._trading_venue in CryptoExchange.__members__ else 'brokers'
        return PROJ_PATH / tv_type / self._trading_venue.value.lower() / self.FILENAME
    
    def _load_config(self, file_path: Path):
        '''Loads adapter.yml'''
        config: dict = load_yaml_file(file_path)
        for group in config:
            group = group.lower()
            for k, v in config[group].items():
                self._add_mapping(group, k, v)

    def _add_mapping(self, group: tADAPTER_GROUP, k: str, v: str):
        group = group.lower()
        self._adapter[group][k] = v
        self._adapter[group][v] = k
        
    def load_all_product_mappings(self):
        '''
        Load all product mappings from market configs.
        Useful when e.g. pfeed needs to download all products and hence needs to know all product mappings.
        '''
        import importlib
        from pfund.enums import CryptoExchange
        if self._trading_venue not in CryptoExchange.__members__:
            raise ValueError(f'load_all_product_mappings is supported for crypto exchanges only, {self._trading_venue} is not a valid crypto exchange')
        exch = self._trading_venue.value
        Exchange = getattr(importlib.import_module(f'pfund.exchanges.{exch.lower()}.exchange'), 'Exchange')
        market_configs_file_path = Exchange.get_file_path(Exchange.MARKET_CONFIGS_FILENAME)
        market_configs: dict[str, dict] = load_yaml_file(market_configs_file_path)
        for category in market_configs:
            for pdt, product_configs in market_configs[category].items():
                epdt = product_configs['symbol']
                self._add_mapping(category, pdt, epdt)
    
    def __len__(self):
        '''
        Returns the number of mappings in the adapter, only count one-sided mappings.
        e.g. a: b, b: a -> counted as 1 mapping
        '''
        return sum(len(mappings) for mappings in self._adapter.values()) // 2
    
    def __contains__(self, item: Any):
        for mappings in self._adapter.values():
            if item in mappings:
                return True
        return False

    def __call__(self, key: str, group: tADAPTER_GROUP='') -> str | tuple:
        group = group.lower()
        if self._is_strict:
            assert group, '"group" cannot be empty when strict=True'
            groups = [group]
        else:
            groups = [group] if group else list(self._adapter.keys())

        for group in groups:
            if group not in self._adapter:
                continue
            if key in self._adapter[group]:
                return self._adapter[group][key]
        return key
