from typing import Literal

from collections import defaultdict

from pfund.utils.utils import load_yaml_file


# NOTE: strings could be product categories, 
# e.g. 'spot', 'linear', 'inverse', 'option', depends on exchange
tADAPTER_GROUP = str | Literal[
    # defined in adapter.yml
    'asset',
    'product_type',
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
    def __init__(self, file_path: str):
        self._adapter = defaultdict(dict)
        self._load_config(file_path)
    
    @property
    def groups(self) -> list[str]:
        return list(self._adapter.keys())
    
    def _load_config(self, file_path: str):
        config: dict = load_yaml_file(file_path)
        for group in config:
            group = group.lower()
            for k, v in config[group].items():
                self.add_mapping(group, k, v)

    def add_mapping(self, group: tADAPTER_GROUP, k: str, v: str):
        group = group.lower()
        self._adapter[group][k] = v
        self._adapter[group][v] = k

    def __call__(self, key: str, group: tADAPTER_GROUP='', strict: bool=False) -> str | tuple:
        '''
        Args:
            strict: if False, it will search for the same key in other groups if group is not specified
        '''
        group = group.lower()
        if strict:
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
