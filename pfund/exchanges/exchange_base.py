import time
import logging
from collections import defaultdict
from functools import reduce
from pprint import pformat
import importlib

from numpy import sign

from pfund.managers.order_manager import OrderUpdateSource
from pfund.adapter import Adapter
from pfund.products import CryptoProduct
from pfund.accounts import CryptoAccount
from pfund.const.paths import EXCHANGE_PATH, PROJ_CONFIG_PATH
from pfund.config.configuration import Configuration
from pfund.zeromq import ZeroMQ
from pfund.utils.utils import step_into, convert_to_uppercases


class BaseExchange:
    USE_WS_PLACE_ORDER = False
    USE_WS_CANCEL_ORDER = False
    SUPPORT_PLACE_BATCH_ORDERS = False
    SUPPORT_CANCEL_BATCH_ORDERS = False 

    def __init__(self, env, name):
        self.env = env.upper()
        self.bkr = 'CRYPTO'
        self.name = self.exch = name.upper()
        self.logger = logging.getLogger(self.name.lower())
        config_path = f'{PROJ_CONFIG_PATH}/{self.exch.lower()}'
        if hasattr(self, 'category'):
            config_path += '/' + self.category
        self.configs = Configuration(config_path, 'config')
        self.adapter = Adapter(config_path, self.configs.load_config_section('adapter'))
        self._products = {}
        self._accounts = {}
        self.categories = []
        
        # APIs
        RestApi = getattr(importlib.import_module(f'pfund.exchanges.{self.exch.lower()}.rest_api'), 'RestApi')
        self._rest_api = RestApi(self.env)
        WebsocketApi = getattr(importlib.import_module(f'pfund.exchanges.{self.exch.lower()}.ws_api'), 'WebsocketApi')
        self._ws_api = WebsocketApi(self.env, self.adapter)
        self._load_settings()

        # used for REST API to send back results in threads to engine
        self._zmq = None
        if not self._is_all_configs_ready():
            self._setup_configs()

    @property
    def products(self):
        return self._products
    
    @property
    def accounts(self):
        return self._accounts
    
    def _load_settings(self):
        settings = self.configs.load_config_section('settings')
        private_channels = settings.get('private_channels', [])
        for channel in private_channels:
            self.add_channel(channel, type_='private')

    def add_category(self, category):
        if category not in self.categories:
            self.categories.append(category)

    def create_product(self, bccy, qccy, ptype, *args, **kwargs) -> CryptoProduct:
        if category := self.PTYPE_TO_CATEGORY[ptype] if hasattr(self, 'PTYPE_TO_CATEGORY') else '':
            self.add_category(category)
        product = CryptoProduct(self.exch, bccy, qccy, ptype, *args, category=category, **kwargs)
        product.load_configs(self.configs)
        return product

    def get_product(self, pdt: str) -> CryptoProduct | None:
        return self._products.get(pdt.upper(), None)

    def add_product(self, product, **kwargs):
        self._products[product.name] = product
        self.logger.debug(f'added product {product.name}')

    def get_account(self, acc: str) -> CryptoAccount | None:
        return self._accounts.get(acc.upper(), None)
    
    def add_account(self, account):
        self._accounts[account.name] = account
        self._ws_api.add_account(account)
        self.logger.debug(f'added account {account.name}')

    def is_use_private_ws_server(self):
        return self._ws_api._is_use_private_ws_server
    
    def get_ws_servers(self):
        return self._ws_api._servers

    def _is_all_configs_ready(self):
        if hasattr(self, 'SUPPORTED_CATEGORIES'):
            config_names = []
            for category in self.SUPPORTED_CATEGORIES:
                config_names.extend([
                    '_'.join(['pdt_matchings', category]), 
                    '_'.join(['tick_sizes', category]), 
                    '_'.join(['lot_sizes', category]), 
                ])
        else:
            config_names = ['pdt_matchings', 'tick_sizes', 'lot_sizes']
        return all(map(self.configs.check_if_config_exists_and_not_empty, config_names))

    def _setup_configs(self):
        # e.g. bybit has a unified API for 4 categories: linear, inverse, spot, option
        if hasattr(self, 'SUPPORTED_CATEGORIES'):
            for category in self.SUPPORTED_CATEGORIES:
                markets = []
                while not markets:
                    markets = self.get_markets(category)
                    if markets:
                        self._create_pdt_matchings_config(markets, category)
                        self.adapter.load_pdt_matchings()
                        self._create_tick_sizes_and_lot_sizes_config(markets, category)
                        break
                    else:
                        config_path = self.configs.get_config_path()
                        self.logger.warning(f'{self.exch} could not get_markets() for category={category} probably due to server issue/maintenance, keep retrying; '\
                                    f'or you can create pdt_matchings.yml, tick_sizes.yml and lot_sizes.yml inside {config_path} manually and re-run the program')
                        time.sleep(3)
        else:
            markets = []
            while not markets:
                markets = self.get_markets()
                if markets:
                    self._create_pdt_matchings_config(markets)
                    self.adapter.load_pdt_matchings()
                    self._create_tick_sizes_and_lot_sizes_config(markets)
                    break
                else:
                    config_path = self.configs.get_config_path()
                    self.logger.warning(f'{self.exch} could not get_markets() probably due to server issue/maintenance, keep retrying; '\
                                f'or you can create pdt_matchings.yml, tick_sizes.yml and lot_sizes.yml inside {config_path} manually and re-run the program')
                    time.sleep(3)
                
    def _create_pdt_matchings_config(self, schema, markets, category=''):
        pdt_macthings = {}
        for market in markets:
            ebccy = step_into(market, schema['bccy'])
            eqccy = step_into(market, schema['qccy'])
            eptype = step_into(market, schema['ptype'])
            epdt = step_into(market, schema['pdt'])
            ebccy, eqccy, epdt = convert_to_uppercases(ebccy, eqccy, epdt)
            ptype = self.adapter(eptype, ref_key='ptypes')
            bccy, qccy = self.adapter(ebccy, eqccy)
            if ptype in ['PERP', 'IPERP', 'SPOT']:
                pdt = self.adapter.build_internal_pdt_format(bccy, qccy, ptype)
                pdt_macthings[pdt] = epdt
            # EXTEND
            # e.g. epdt = 'BTC-26MAY23' for linear futures, 
            # e.g. epdt = BTC-29SEP23-80000-C
            # create the internal formats
            elif ptype in ['FUT', 'IFUT', 'OPT']:
                pass
        config_name = 'pdt_matchings'
        file_name = '_'.join([config_name, category])
        self.configs.write_config(file_name, pdt_macthings)

    def _create_tick_sizes_and_lot_sizes_config(self, schema, markets, tag=''):
        for config_name in ['tick_sizes', 'lot_sizes']:
            file_name = '_'.join([config_name, tag])
            sizes = {}
            for market in markets:
                epdt = step_into(market, schema['pdt'])
                pdt = self.adapter(epdt, ref_key=tag)
                size = step_into(market, schema[config_name[:-1]])
                sizes[pdt] = str(size)
            self.configs.write_config(file_name, sizes)

    def _parse_return(self, ret, schema, default_result=None):
        if 'error_from' not in ret:
            res = step_into(ret, schema)
            return res
        else:
            pretty_ret = pformat(ret, sort_dicts=False)
            self.logger.error(pretty_ret)
            # when default_result is not None, it means error message from exchange will not be handled
            if default_result is not None:
                return default_result
            else:
                return ret['message'] if not ret['is_exception'] else default_result

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
        from pfund import TradeEngine
        self._zmq = ZeroMQ(self.exch+'_'+'rest_api')
        self._zmq.start(
            logger=self.logger,
            send_port=TradeEngine.zmq_ports[self.exch]['rest_api'],
            # only used to send out returns from REST API
            # shouldn't receive any msgs, so recv_ports is empty
            recv_ports=[],
        )

    def stop(self):
        self._zmq.stop()

    def add_channel(self, channel, type_, product=None, **kwargs):
        if type_.lower() == 'public':
            assert product
            # need to assert using the exchange.add_product() first so that the product is loaded correctly
            assert product.pdt in self._products, f"{product.pdt} must be added in the exchange first using exchange.add_product(...)"
            if channel == 'kline':
                assert 'period' in kwargs and 'timeframe' in kwargs, 'Keyword arguments "period" or/and "timeframe" is missing'
            self._ws_api.add_product(product, **kwargs)
        self._ws_api.add_channel(channel, type_, product=product, **kwargs)


    '''
    ************************************************
    API Calls
    ************************************************
    '''
    def get_markets(self, schema, params=None, is_write_samples=False):
        ret = self._rest_api.get_markets(params=params)
        res = self._parse_return(
            ret,
            schema['result'],
            default_result=[]
        )
        # write down the sample data for later retrieval
        if is_write_samples:
            return_path = f'{EXCHANGE_PATH}/{self.exch.lower()}/rest_api_samples/get_markets_return'
            result_path = f'{EXCHANGE_PATH}/{self.exch.lower()}/rest_api_samples/get_markets_result'
            if params and 'category' in params:
                category = params['category']
                return_path += '_' + category
                result_path += '_' + category
            with open(return_path, 'w') as f:
                f.write(pformat(ret))
            with open(result_path, 'w') as f:
                f.write(pformat(res))
        markets = res
        return markets

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
                    pdt = self.adapter(epdt, ref_key=category)
                    update = {}
                    for k, (ek, *sequence) in schema['data'].items():
                        ref_key = k + 's' if k in ['tif', 'side'] else ''
                        initial_value = self.adapter(step_into(order, ek), ref_key=ref_key)
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
                    ccy = self.adapter(eccy)
                    for k, (ek, *sequence) in schema['data'].items():
                        initial_value = self.adapter(step_into(balance, ek))
                        v = reduce(lambda v, f: f(v) if v else v, sequence, initial_value)
                        balances['data'][ccy][k] = v
            elif res_type is list:
                for balance in res:
                    eccy = step_into(balance, schema['ccy'])
                    ccy = self.adapter(eccy)
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
                    pdt = self.adapter(epdt, ref_key=category)
                    qty = float(step_into(position, schema['data']['qty'][0]))
                    if qty == 0 and pdt not in self._products:
                        continue
                    if 'side' in schema:
                        eside = step_into(position, schema['side'])
                        side = self.adapter(eside, ref_key='sides')
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
                    pdt = self.adapter(epdt, ref_key=category)
                    update = {}
                    for k, (ek, *sequence) in schema['data'].items():
                        ref_key = k + 's' if k in ['tif', 'side'] else ''
                        initial_value = self.adapter(step_into(trade, ek), ref_key=ref_key)
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
                    ref_key = k + 's' if k in ['tif', 'side'] else ''
                    initial_value = self.adapter(step_into(res, ek), ref_key=ref_key)
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
                    ref_key = k + 's' if k in ['tif', 'side'] else ''
                    initial_value = self.adapter(step_into(res, ek), ref_key=ref_key)
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