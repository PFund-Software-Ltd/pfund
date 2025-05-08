from __future__ import annotations  
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.typing import tENVIRONMENT, tCRYPTO_EXCHANGE
    from pfund.datas.data_base import BaseData
    from pfund.enums import PublicDataChannel

import os
import datetime
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import reduce
import importlib

from numpy import sign
import yaml

from pfund.managers.order_manager import OrderUpdateSource
from pfund.adapter import Adapter
from pfund.products.product_crypto_cefi import get_CeFiCryptoProduct
from pfund.products.product_base import BaseProduct
from pfund.accounts.account_crypto import CryptoAccount
from pfund.enums import (
    Environment, 
    CeFiProductType, 
    PrivateDataChannel, 
    DataChannelType,
)
from pfund.const.paths import PROJ_PATH
from pfund.utils.utils import get_last_modified_time, load_yaml_file
from pfund.config import get_config


class BaseExchange(ABC):
    SUPPORT_PLACE_BATCH_ORDERS = False
    SUPPORT_CANCEL_BATCH_ORDERS = False 
    
    USE_WS_PLACE_ORDER = False
    USE_WS_CANCEL_ORDER = False
    
    SUPPORTED_PRIVATE_CHANNELS = ['trade', 'balance', 'position', 'order']

    def __init__(self, env: tENVIRONMENT, name: tCRYPTO_EXCHANGE, refetch_market_configs=False):
        '''
        Args:
            refetch_market_configs: 
                if True, refetch markets (e.g. tick sizes, lot sizes, listed markets) from exchange 
                even if the config files exist.
                if False, markets will be automatically refetched on a weekly basis
        '''
        self.env = Environment[env.upper()]
        self.bkr = 'CRYPTO'
        self.name = self.exch = name.upper()
        self.logger = logging.getLogger(self.name.lower())
        self.adapter = Adapter(f'{PROJ_PATH}/exchanges/{self.exch.lower()}/adapter.yml')
        self._products = {}
        self._accounts = {}
        
        # APIs
        RestApi = getattr(importlib.import_module(f'pfund.exchanges.{self.exch.lower()}.rest_api'), 'RestApi')
        self._rest_api = RestApi(self.env, self.adapter)
        WebsocketApi = getattr(importlib.import_module(f'pfund.exchanges.{self.exch.lower()}.ws_api'), 'WebsocketApi')
        self._ws_api = WebsocketApi(self.env, self.adapter)
        # FIXME: remove it
        self._add_default_private_channels()

        # used for REST API to send back results in threads to engine
        self._zmq = None
        
        self._check_if_refetch_market_configs(refetch_market_configs)
            
    def _check_if_refetch_market_configs(self, refetch_market_configs: bool):
        '''
        Fetch market information from exchange, including:
        - Tick sizes (minimum price increments)
        - Lot sizes (minimum quantity increments) 
        - Listed markets/trading pairs
        and then save them to the cache.
        '''
        filename = 'market_configs.yml'
        config = get_config()
        file_path = f'{config.cache_path}/{self.exch.lower()}/{filename}'
        def _check_if_market_configs_outdated() -> bool:
            '''
            Check if the market configs are outdated or do not exist.
            If outdated or do not exist, return True.
            '''
            if not os.path.exists(file_path):
                self.logger.debug(f'{self.exch} {filename} does not exist, fetching data...')
                return True
            last_modified_time = get_last_modified_time(file_path)
            renew_every_x_days = 7
            is_outdated = last_modified_time + datetime.timedelta(days=renew_every_x_days) < datetime.datetime.now(tz=datetime.timezone.utc)
            if is_outdated:
                os.remove(file_path)
                self.logger.info(f'{self.exch} {filename} is outdated, refetching data...')
            return is_outdated
        if not (refetch_market_configs or _check_if_market_configs_outdated()):
            return
        if (markets_per_category := self.get_markets()) is None:
            return
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(yaml.dump(markets_per_category))

    def load_market_configs(self):
        '''
        Load market configs from cache.
        The file can contain thousands of markets, so it's not loaded into memory by default,
        '''
        filename = 'market_configs.yml'
        config = get_config()
        file_path = f'{config.cache_path}/{self.exch.lower()}/{filename}'
        return load_yaml_file(file_path)
    
    def load_all_product_mappings(self):
        '''
        Load all product mappings from market configs.
        '''
        market_configs: dict[str, dict] = self.load_market_configs()
        for category in market_configs:
            for pdt, product_configs in market_configs[category].items():
                epdt = product_configs['symbol']
                self.adapter.add_mapping(category, pdt, epdt)
    
    @staticmethod
    @abstractmethod
    def _derive_product_category(product_type: str) -> str:
        pass
    
    @property
    def products(self):
        return self._products
    
    @property
    def accounts(self):
        return self._accounts
    
    def create_product(self, product_basis: str, product_alias: str='', **product_specs) -> BaseProduct:
        base_asset, quote_asset, ptype = product_basis.split('_')
        ptype = CeFiProductType[ptype]
        CeFiCryptoProduct = get_CeFiCryptoProduct(product_basis)
        category = self._derive_product_category(ptype)
        # symbol = epdt = external product, e.g. BTC_USDT_PERP -> BTCUSDT
        symbol = self._map_internal_to_external_product_name(
            base_asset.upper(), 
            quote_asset.upper(), 
            ptype,
            specs=product_specs,
        )
        product = CeFiCryptoProduct(
            bkr='CRYPTO',
            exch=self.exch,
            symbol=symbol,
            base_asset=base_asset,
            quote_asset=quote_asset,
            type=ptype,
            category=category,
            alias=product_alias,
            **product_specs,
        )
            
        return product

    def get_product(self, pdt: str) -> BaseProduct | None:
        return self._products.get(pdt.upper(), None)

    def add_product(self, product: BaseProduct):
        self._products[product.name] = product
        self._rest_api.add_category(product.category)
        self.adapter.add_mapping(product.category, product.name, product.symbol)
        # TODO: check if the product is listed in the markets
        self.logger.debug(f'added product {product.name}')

    def get_account(self, acc: str) -> CryptoAccount | None:
        return self._accounts.get(acc.upper(), None)
    
    def add_account(self, account: CryptoAccount):
        self._accounts[account.name] = account
        self._ws_api.add_account(account)
        self.logger.debug(f'added account {account.name}')
    
    def _add_default_private_channels(self):
        for channel in self.SUPPORTED_PRIVATE_CHANNELS:
            channel = PrivateDataChannel[channel.lower()]
            self.add_channel(channel, channel_type=DataChannelType.private)
            
    def add_channel(
        self,
        channel: PublicDataChannel | PrivateDataChannel | str,
        channel_type: DataChannelType,
        data: BaseData | None=None
    ):
        self._ws_api.add_channel(channel, channel_type, data=data)
        
    def remove_channel(
        self, 
        channel: PublicDataChannel | PrivateDataChannel | str,
        channel_type: DataChannelType,
        data: BaseData | None=None
    ):
        self._ws_api.remove_channel(channel, channel_type, data=data)

    def use_separate_private_ws_url(self) -> bool:
        return self._ws_api._use_separate_private_ws_url
    
    def get_ws_servers(self):
        return self._ws_api._servers

    # REVIEW
    @staticmethod
    def _combine_trades(trades: list):
        """
        Combines trades with the same eoid from trade history 
        because some exchanges separate trades for the same order
        """
        trades_per_eoid = defaultdict(list)
        trades_combined = []
        for trade in trades:
            eoid = trade['eoid']
            trades_per_eoid[eoid].append(trade)
        for trades in trades_per_eoid.values():
            # if multiple trades for the same order, combine them
            if len(trades) > 1:
                avg_px = filled_qty = 0.0
                trade_ts = 0.0
                for trade in trades:
                    last_traded_px, last_traded_qty = trade['ltp'], trade['ltq']
                    avg_px += last_traded_px * last_traded_qty
                    filled_qty += last_traded_qty
                    trade_ts = max(trade_ts, trade['trade_ts']) 
                avg_px /= filled_qty
                trade_adj = trades[-1]
                trade_adj['avg_px'] = avg_px
                trade_adj['filled_qty'] = filled_qty
            else:
                trade_adj = trades[0]
                trade_adj['avg_px'] = trade_adj['ltp']
                trade_adj['filled_qty'] = trade_adj['ltq']
            trades_combined.append(trade_adj)
        return trades_combined
    
    def start(self):
        from pfund.engines.trade_engine import TradeEngine
        zmq_ports = TradeEngine.settings['zmq_ports']
        self._zmq = ZeroMQ(self.exch+'_'+'rest_api')
        self._zmq.start(
            logger=self.logger,
            send_port=zmq_ports[self.exch]['rest_api'],
            # only used to send out returns from REST API
            # shouldn't receive any msgs, so recv_ports is empty
            recv_ports=[],
        )

    def stop(self):
        self._zmq.stop()


    '''
    ************************************************
    API Calls
    ************************************************
    '''
    def get_markets(self, category: str) -> dict | None:
        return self._rest_api.get_markets(category)

    # TODO: update to get rid of step_into()
    def get_orders(self, account: CryptoAccount, schema, params=None, **kwargs) -> dict | None:
        orders = {'ts': None, 'data': defaultdict(list), 'source': OrderUpdateSource.GOO}
        ret = self._rest_api.get_orders(account, params=params)
        res = self._parse_return(ret, schema['result'], default_result=False)
        res_type = type(res)
        if res:
            if 'ts' in schema:
                orders['ts'] = float(step_into(ret, schema['ts'])) * schema['ts_adj']
            if res_type is list:
                for order in res:
                    epdt = step_into(order, schema['pdt'])
                    category = params.get('category', '')
                    pdt = self.adapter(epdt, group=category)
                    update = {}
                    for k, (ek, *sequence) in schema['data'].items():
                        group = k + 's' if k in ['tif', 'side'] else ''
                        initial_value = self.adapter(step_into(order, ek), group=group)
                        v = reduce(lambda v, f: f(v) if v else v, sequence, initial_value)
                        update[k] = v
                    orders['data'][pdt].append(update)
                    eoid = update['eoid']
            else:
                raise Exception(f'{self.exch} unhandled {res_type=}')
        if self._zmq and orders['data']:
            zmq_msg = (2, 1, (self.bkr, self.exch, account.acc, orders))
            self._zmq.send(*zmq_msg, receiver='engine')
        return orders

    def get_balances(self, account: CryptoAccount, schema, params=None, **kwargs) -> dict | None:
        balances = {'ts': None, 'data': defaultdict(dict)}
        ret = self._rest_api.get_balances(account, params=params)
        res = self._parse_return(ret, schema['result'], default_result=False)
        res_type = type(res)
        if res:
            if 'ts' in schema:
                balances['ts'] = float(step_into(ret, schema['ts'])) * schema['ts_adj']
            if res_type is dict:
                for eccy, balance in res.items():
                    ccy = self.adapter(eccy, group='asset')
                    for k, (ek, *sequence) in schema['data'].items():
                        initial_value = self.adapter(step_into(balance, ek))
                        v = reduce(lambda v, f: f(v) if v else v, sequence, initial_value)
                        balances['data'][ccy][k] = v
            elif res_type is list:
                for balance in res:
                    eccy = step_into(balance, schema['ccy'])
                    ccy = self.adapter(eccy, group='asset')
                    for k, (ek, *sequence) in schema['data'].items():
                        initial_value = self.adapter(step_into(balance, ek))
                        v = reduce(lambda v, f: f(v) if v else v, sequence, initial_value)
                        balances['data'][ccy][k] = v
            else:
                raise Exception(f'{self.exch} unhandled {res_type=}')
        if self._zmq and balances['data']:
            zmq_msg = (3, 1, (self.bkr, self.exch, account.acc, balances,))
            self._zmq.send(*zmq_msg, receiver='engine')
        return balances

    def get_positions(self, account: CryptoAccount, schema, params=None, **kwargs) -> dict | None:
        positions = {'ts': None, 'data': defaultdict(dict)}
        ret = self._rest_api.get_positions(account, params=params)
        res = self._parse_return(ret, schema['result'], default_result=False)
        res_type = type(res)
        if res:
            if 'ts' in schema:
                positions['ts'] = float(step_into(ret, schema['ts'])) * schema['ts_adj']
            if res_type is list:
                for position in res:
                    epdt = step_into(position, schema['pdt'])
                    category = params.get('category', '')
                    pdt = self.adapter(epdt, group=category)
                    qty = float(step_into(position, schema['data']['qty'][0]))
                    if qty == 0 and pdt not in self._products:
                        continue
                    if 'side' in schema:
                        eside = step_into(position, schema['side'])
                        side = self.adapter(eside, group='side')
                    # e.g. BINANCE_USDT only returns position size (signed qty)
                    elif 'size' in schema:
                        side = sign(step_into(position, schema['size']))
                    positions['data'][pdt][side] = {}
                    for k, (ek, *sequence) in schema['data'].items():
                        initial_value = self.adapter(step_into(position, ek))
                        v = reduce(lambda v, f: f(v) if v else v, sequence, initial_value)
                        positions['data'][pdt][side][k] = v
            else:
                raise Exception(f'{self.exch} unhandled {res_type=}')
        if self._zmq and positions['data']:
            zmq_msg = (3, 2, (self.bkr, self.exch, account.acc, positions,))
            self._zmq.send(*zmq_msg, receiver='engine')
        return positions

    def get_trades(self, account: CryptoAccount, schema, params=None, **kwargs) -> dict | None:
        trades = {'ts': None, 'data': defaultdict(list), 'source': OrderUpdateSource.GTH}
        ret = self._rest_api.get_trades(account, params=params)
        res = self._parse_return(ret, schema['result'], default_result=False)
        res_type = type(res)
        if res:
            if 'ts' in schema:
                trades['ts'] = float(step_into(ret, schema['ts'])) * schema['ts_adj']
            if res_type is list:
                for trade in res:
                    epdt = step_into(trade, schema['pdt'])
                    category = params.get('category', '')
                    pdt = self.adapter(epdt, group=category)
                    update = {}
                    for k, (ek, *sequence) in schema['data'].items():
                        group = k + 's' if k in ['tif', 'side'] else ''
                        initial_value = self.adapter(step_into(trade, ek), group=group)
                        v = reduce(lambda v, f: f(v) if v else v, sequence, initial_value)
                        update[k] = v
                        if k == 'trade_ts':
                            update[k] *= schema['ts_adj']
                    trades['data'][pdt].append(update)
            else:
                raise Exception(f'{self.exch} unhandled {res_type=}')
        if self._zmq and trades['data']:
            zmq_msg = (2, 1, (self.bkr, self.exch, account.acc, trades))
            self._zmq.send(*zmq_msg, receiver='engine')
        return trades
    
    def place_order(self, account: CryptoAccount, schema: dict, params=None, **kwargs):
        order = {'ts': None, 'data': {}, 'source': OrderUpdateSource.REST}
        ret = self._rest_api.place_order(account, params=params)
        res = self._parse_return(ret, schema['result'], default_result=False)
        res_type = type(res)
        if res:
            if 'ts' in schema:
                order['ts'] = float(step_into(ret, schema['ts'])) * schema['ts_adj']
            if res_type is dict:
                for k, (ek, *sequence) in schema['data'].items():
                    group = k + 's' if k in ['tif', 'side'] else ''
                    initial_value = self.adapter(step_into(res, ek), group=group)
                    v = reduce(lambda v, f: f(v) if v else v, sequence, initial_value)
                    order[k] = v
            else:
                raise Exception(f'{self.exch} unhandled {res_type=}')
        if self._zmq and order['data']:
            zmq_msg = (2, 1, (self.bkr, self.exch, account.acc, order))
            self._zmq.send(*zmq_msg, receiver='engine')
        return order

    def cancel_order(self, account: CryptoAccount, schema: dict, params=None, **kwargs):
        order = {'ts': None, 'data': {}, 'source': OrderUpdateSource.REST}
        ret = self._rest_api.cancel_order(account, params=params)
        res = self._parse_return(ret, schema['result'], default_result=False)
        res_type = type(res)
        if res:
            if 'ts' in schema:
                order['ts'] = float(step_into(ret, schema['ts'])) * schema['ts_adj']
            if res_type is dict:
                for k, (ek, *sequence) in schema['data'].items():
                    group = k + 's' if k in ['tif', 'side'] else ''
                    initial_value = self.adapter(step_into(res, ek), group=group)
                    v = reduce(lambda v, f: f(v) if v else v, sequence, initial_value)
                    order[k] = v
            else:
                raise Exception(f'{self.exch} unhandled {res_type=}')
        if self._zmq and order['data']:
            zmq_msg = (2, 1, (self.bkr, self.exch, account.acc, order))
            self._zmq.send(*zmq_msg, receiver='engine')
        return order

    # TODO
    def place_batch_orders(self, account: CryptoAccount, scheme, params=None, **kwargs):
        pass

    # TODO
    def cancel_batch_orders(self, account: CryptoAccount, scheme, params=None, **kwargs):
        pass