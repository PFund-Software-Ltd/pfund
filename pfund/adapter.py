from typing import Any

from pathlib import Path

from pfund.enums import TradingVenue
from pfund.utils.utils import load_yaml_file


class Adapter:
    FILENAME = 'adapter.yml'
    
    def __init__(self, trading_venue: str):
        self._trading_venue = TradingVenue[trading_venue.upper()]
        self._adapter = {}
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
        return PROJ_PATH / tv_type / self._trading_venue.lower() / self.FILENAME
    
    def _load_config(self, file_path: Path):
        '''Loads adapter.yml'''
        config: dict = load_yaml_file(file_path)
        for group in config:
            if config[group]:
                for k, v in config[group].items():
                    self.add_mapping(group, k, v)
            else:
                self._adapter[group.lower()] = {}

    def add_mapping(self, group: str, key: Any, value: Any):
        group = group.lower()
        if group not in self._adapter:
            self._adapter[group] = {}
        self._adapter[group][key] = value
        self._adapter[group][value] = key
    
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

    def __call__(self, key: Any, *, group: str, strict: bool=False) -> Any:
        '''
        Args:
            strict: if True, raise KeyError if key is not found
        '''
        if strict:
            return self._adapter[group.lower()][key]
        else:
            return self._adapter[group.lower()].get(key, key)
