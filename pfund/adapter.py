import yaml
import os

from pfund.const.paths import PROJ_CONFIG_PATH


class Adapter:
    def __init__(self, trading_venue, adapter_dict):
        self._trading_venue = trading_venue
        self._adapter_dict = adapter_dict
        self._adapter = {}
        self._ref_keys = []
        for adapt_type in adapter_dict:
            is_one_way = True if '>' in adapt_type else False
            # need to handle tifs individually because it may collide with ccy names, e.g. GTC can be a coin
            ref_key = adapt_type if adapt_type in ['tifs', 'sides', 'ptypes'] else ''
            if ref_key:
                self._ref_keys.append(ref_key)
            for k,v in adapter_dict[adapt_type].items():
                self.update(k, v, ref_key=ref_key, is_one_way=is_one_way)
        self.load_pdt_matchings()

    def build_internal_pdt_format(self, bccy, qccy, ptype):
        pdt = '_'.join([bccy, qccy, ptype])
        return pdt

    def load_pdt_matchings(self):
        file_path = f'{PROJ_CONFIG_PATH}/{self._trading_venue.lower()}'
        config_name = 'pdt_matchings'
        for file_name in os.listdir(file_path):
            if not file_name.startswith(config_name):
                continue
            file_splits = file_name.split('_')
            if len(file_splits) > 2:
                category = file_splits[-1].split('.')[0]
            else:
                category = ''
            with open(file_path + '/' + file_name, 'r') as f:
                if pdt_macthings := yaml.safe_load(f):
                    for pdt, epdt in pdt_macthings.items():
                        self.update(pdt, epdt, ref_key=category)
                        if category:
                            self._ref_keys.append(category)

    def update(self, k, v, ref_key='', is_one_way=False):
        if not ref_key:
            self._adapter[k] = v
            if not is_one_way:
                self._adapter[v] = k
        else:
            if ref_key not in self._adapter:
                self._adapter[ref_key] = {}
            self._adapter[ref_key][k] = v
            if not is_one_way:
                self._adapter[ref_key][v] = k

    def __call__(self, *keys, ref_key='') -> str | tuple:
        if not ref_key:
            adapted_keys = tuple(self._adapter.get(key, key) for key in keys)
        else:
            adapted_keys = tuple(self._adapter[ref_key].get(key, key) if ref_key in self._adapter else key for key in keys)
        if len(keys) == 1:
            adapted_keys = adapted_keys[0]
        return adapted_keys

    def __str__(self):
        return str(self._adapter)
