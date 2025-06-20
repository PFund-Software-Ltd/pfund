from __future__ import annotations  
from typing import TYPE_CHECKING, ClassVar, TypeAlias
if TYPE_CHECKING:
    from pathlib import Path
    from pfund.exchanges.rest_api_base import Result, RawResult
    from pfund.typing import tEnvironment, ProductName, AccountName
    from pfund.products.product_crypto import CryptoProduct
    from pfund.accounts.account_crypto import CryptoAccount
    from pfund.datas.data_base import BaseData
    from pfund.engines.trade_engine_settings import TradeEngineSettings

import os
import datetime
import logging
import importlib
from functools import reduce
from collections import defaultdict
from abc import ABC, abstractmethod

from pfund.adapter import Adapter
from pfund.managers.order_manager import OrderUpdateSource
from pfund.enums import Environment, Broker, CryptoExchange, PublicDataChannel, PrivateDataChannel, DataChannelType


ProductCategory: TypeAlias = str


class BaseExchange(ABC):
    bkr = Broker.CRYPTO
    name: ClassVar[CryptoExchange]
    adapter: ClassVar[Adapter]
    MARKET_CONFIGS_FILENAME = 'market_configs.yml'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.adapter = Adapter(cls.name)
            
    def __init__(self, env: Environment | tEnvironment):
        from pfund.engines.trade_engine import TradeEngine
        
        self._env = Environment[env.upper()]
        self._logger = logging.getLogger(self.name.lower())
        self._products: dict[ProductName, CryptoProduct] = {}
        self._accounts: dict[AccountName, CryptoAccount] = {}
        self._settings: TradeEngineSettings | None = getattr(TradeEngine, "_settings", None)

        # APIs
        exchange_path = f'pfund.exchanges.{self.name.lower()}'
        RestApi = getattr(importlib.import_module(f'{exchange_path}.rest_api'), 'RestApi')
        self._rest_api = RestApi(self._env)
        WebsocketApi = getattr(importlib.import_module(f'{exchange_path}.ws_api'), 'WebsocketApi')
        self._ws_api = WebsocketApi(self._env)

        # used for REST API to send back results in threads to engine
        self._zmq = None
        
        if self._settings:
            self._check_if_refetch_market_configs()
    
    @property
    def broker(self) -> Broker:
        return self.bkr
    
    @property
    def products(self):
        return self._products
    
    @property
    def accounts(self):
        return self._accounts
    
    @classmethod
    def get_file_path(cls, filename: str) -> Path | None:
        '''
        Args:
            filename: the filename of the file to get the path for, e.g. market_configs.yml
        '''
        from pfund.config import get_config
        config = get_config()
        file_paths = {
            cls.MARKET_CONFIGS_FILENAME: config.cache_path / cls.name
        }
        if filename in file_paths:
            return file_paths[filename] / filename
        else:
            print(f'{filename} not found')
            return None
    
    def load_market_configs(self):
        from pfund.utils.utils import load_yaml_file
        market_configs_file_path = self.get_file_path(self.MARKET_CONFIGS_FILENAME)
        market_configs: dict[str, dict] = load_yaml_file(market_configs_file_path)
        return market_configs
    
    def _check_if_refetch_market_configs(self):
        '''
        Check if the market configs are outdated and need to be refetched.
        If so, refetch the market configs and return True.
        If not, return False.
        '''
        from pfund.utils.utils import get_last_modified_time

        filename = self.MARKET_CONFIGS_FILENAME
        file_path = self.get_file_path(filename)
        
        force_refetching = self._settings.refetch_market_configs
        is_exist = os.path.exists(file_path)
        is_outdated = (is_exist and (
            get_last_modified_time(file_path)
            + datetime.timedelta(days=self._settings.renew_market_configs_every_x_days) 
            < datetime.datetime.now(tz=datetime.timezone.utc)
        ))

        is_refetching = False

        if force_refetching:
            is_refetching = True
        elif not is_exist:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self._logger.debug(f'{self.name} {filename} does not exist, fetching data...')
            is_refetching = True
        elif is_outdated:
            self._logger.info(f'{self.name} {filename} is outdated, refetching data...')
            is_refetching = True
        
        if is_refetching:
            self.fetch_market_configs()
        return is_refetching
    
    def fetch_market_configs(self):
        '''
        Fetch market information from exchange, including:
        - Tick sizes (minimum price increments)
        - Lot sizes (minimum quantity increments) 
        - Listed markets/trading pairs
        and then append them to the existing market configs.
        '''
        from pfund.utils.utils import load_yaml_file, dump_yaml_file
        markets: dict[ProductCategory, Result] = self.get_markets()
        market_configs_file_path = self.get_file_path(self.MARKET_CONFIGS_FILENAME)
        existing_market_configs = load_yaml_file(market_configs_file_path) or {}
        for category, result in markets.items():
            is_success = result['is_success']
            if not is_success:
                self._logger.warning(f'failed to fetch market configs for {category}')
                continue
            configs = {config['symbol']: config for config in result['data']['message']}
            existing_market_configs[category.upper()] = configs
        dump_yaml_file(market_configs_file_path, existing_market_configs)

    @classmethod
    def create_product(cls, basis: str, name: str='', **specs) -> CryptoProduct:
        from pfund.products import ProductFactory
        Product = ProductFactory(trading_venue=cls.name, basis=basis)
        return Product(basis=basis, adapter=cls.adapter, name=name, **specs)

    def get_product(self, name: str) -> CryptoProduct:
        '''
        Args:
            name: product name (product.name)
        '''
        return self._products[name]

    def add_product(self, product: CryptoProduct) -> CryptoProduct:
        if product.name not in self._products:
            market_configs = self.load_market_configs()
            if product.symbol not in market_configs[product.category]:
                raise ValueError(
                    f"The symbol '{product.symbol}' is not found in the market configurations. "
                    f"It might be delisted, or your market configurations could be outdated. "
                    f"Please set 'refetch_market_configs=True' in TradeEngine's settings to refetch the latest market configurations."
                )
            self._products[product.name] = product
            # REVIEW: maybe use asset_type instead of category for more generic grouping?
            self.adapter._add_mapping(product.category, product.name, product.symbol)
            self._logger.debug(f'added {product=}')
        else:
            existing_product: CryptoProduct = self.get_product(product.name)
            # assert products are the same with the same name
            if existing_product == product:
                product = existing_product
            else:
                raise ValueError(f'product name {product.name} has already been used for {existing_product}')
        return product

    def get_account(self, name: str) -> CryptoAccount:
        return self._accounts[name]
    
    def add_account(self, account: CryptoAccount) -> CryptoAccount:
        if account.name not in self._accounts:
            self._accounts[account.name] = account
            self._ws_api._add_account(account)
            self._logger.debug(f'added {account=}')
        else:
            raise ValueError(f'account name {account.name} has already been added')
        return account
            
    def add_channel(
        self,
        channel: PublicDataChannel | PrivateDataChannel | str,
        channel_type: DataChannelType,
        data: BaseData | None=None
    ):
        self._ws_api._add_channel(channel, channel_type, data=data)
        
    # FIXME
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
        self._zmq = ZeroMQ(self.name+'_'+'rest_api')
        self._zmq.start(
            logger=self._logger,
            send_port=zmq_ports[self.name]['rest_api'],
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
    @abstractmethod
    async def aget_markets(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def get_markets(self, *args, **kwargs):
        pass

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
                raise Exception(f'{self.name} unhandled {res_type=}')
        if self._zmq and orders['data']:
            zmq_msg = (2, 1, (self.bkr, self.name, account.acc, orders))
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
                raise Exception(f'{self.name} unhandled {res_type=}')
        if self._zmq and balances['data']:
            zmq_msg = (3, 1, (self.bkr, self.name, account.acc, balances,))
            self._zmq.send(*zmq_msg, receiver='engine')
        return balances

    def get_positions(self, account: CryptoAccount, schema, params=None, **kwargs) -> dict | None:
        from numpy import sign

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
                raise Exception(f'{self.name} unhandled {res_type=}')
        if self._zmq and positions['data']:
            zmq_msg = (3, 2, (self.bkr, self.name, account.acc, positions,))
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
                raise Exception(f'{self.name} unhandled {res_type=}')
        if self._zmq and trades['data']:
            zmq_msg = (2, 1, (self.bkr, self.name, account.acc, trades))
            self._zmq.send(*zmq_msg, receiver='engine')
        return trades
    
    def place_order(self, account: CryptoAccount, schema: dict, params=None, expires_in=5000):
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
                raise Exception(f'{self.name} unhandled {res_type=}')
        if self._zmq and order['data']:
            zmq_msg = (2, 1, (self.bkr, self.name, account.acc, order))
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
                raise Exception(f'{self.name} unhandled {res_type=}')
        if self._zmq and order['data']:
            zmq_msg = (2, 1, (self.bkr, self.name, account.acc, order))
            self._zmq.send(*zmq_msg, receiver='engine')
        return order

    # TODO
    def place_batch_orders(self, account: CryptoAccount, scheme, params=None, **kwargs):
        pass

    # TODO
    def cancel_batch_orders(self, account: CryptoAccount, scheme, params=None, **kwargs):
        pass