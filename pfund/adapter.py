from typing import Literal, TypeAlias

from collections import defaultdict
from pathlib import Path

from pfund.enums import TradingVenue


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
    def __init__(self, trading_venue: str, is_strict: bool=False):
        '''
        Args:
            is_strict: if False, it will search for the same key in other groups if group is not specified
        '''
        trading_venue = TradingVenue[trading_venue.upper()]
        self._adapter = defaultdict(dict)
        self._is_strict = is_strict
        self._load_config(self._get_file_path(trading_venue))
    
    @property
    def groups(self) -> list[str]:
        return list(self._adapter.keys())
    
    def print_groups(self):
        from pprint import pprint
        pprint(self.groups)
    
    # TODO: __str__
    
    @staticmethod
    def _get_file_path(trading_venue: TradingVenue) -> Path:
        from pfund.const.paths import PROJ_PATH
        from pfund.enums import CryptoExchange
        filename = 'adapter.yml'
        tv_type = 'exchanges' if trading_venue in CryptoExchange.__members__ else 'brokers'
        return PROJ_PATH / tv_type / trading_venue.value.lower() / filename
    
    def _load_config(self, file_path: Path):
        from pfund.utils.utils import load_yaml_file
        config: dict = load_yaml_file(file_path)
        for group in config:
            group = group.lower()
            for k, v in config[group].items():
                self._add_mapping(group, k, v)

    def _add_mapping(self, group: tADAPTER_GROUP, k: str, v: str):
        group = group.lower()
        self._adapter[group][k] = v
        self._adapter[group][v] = k

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
