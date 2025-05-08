from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.brokers.broker_base import BaseBroker
    
import time
from collections import OrderedDict, defaultdict

from typing import Union
from enum import Enum


from pfund.orders.order_statuses import *
from pfund.orders.order_crypto import CryptoOrder
from pfund.orders.order_ib import IBOrder
from pfund.enums import Event, RunMode


class OrderUpdateSource(Enum):
    GTH = 'get_trades_history'
    GOO = 'get_opened_orders'
    REST = 'rest_api'  # place_orders/amend_orders/cancel_orders from REST API
    WSO = 'websocket_api_orders'
    WST = 'websocket_api_trades'


OrderClosedReason = Union[FillOrderStatus, CancelOrderStatus, MainOrderStatus]


class OrderManager:
    _NUM_OF_CACHED_CLOSED_ORDERS = 200
    _MAX_NUM_OF_CLOSED_ORDERS = 500
    _NO_ORDER_UPDATE_TOLERANCE_IN_SECONDS = 3
    _MAX_MISSED_NUM = 3
    _MAX_RECON_NUM = 5

    def __init__(self, broker: BaseBroker):
        self._broker = broker
        self._logger = broker._logger
        # { trading_venue: { acc: OrderedDict(order_id: OrderObject) } }
        self.submitted_orders = defaultdict(lambda: defaultdict(OrderedDict))
        self.opened_orders = defaultdict(lambda: defaultdict(OrderedDict))
        self.closed_orders = defaultdict(lambda: defaultdict(OrderedDict))
        
        # { order_id: num }
        self._missed_nums = defaultdict(int)
        self._recon_nums = defaultdict(int)
        self._cxl_rej_nums = defaultdict(int)

    def push(self, order, event: Event, type_):
        if strategy := self._engine.strategy_manager.get_strategy(order.strat):
            if self._broker._run_mode == RunMode.REMOTE:
                if strategy.is_running():
                    if event == Event.order:
                        strategy._update_orders(order, type_)
                    elif event == Event.trade:
                        strategy._update_trades(order, type_)
            else:
                raise NotImplementedError('parallel strategy is not implemented')
                # TODO
                # self._broker._zmq

    def get_order(self, trading_venue, acc, oid='', eoid=''):
        assert oid or eoid, 'At least one of oid or eoid should be provided'
        submitted_orders = self.submitted_orders[trading_venue][acc]
        opened_orders = self.opened_orders[trading_venue][acc]
        closed_orders = self.closed_orders[trading_venue][acc]
        if oid:
            if oid in submitted_orders:
                return submitted_orders[oid]
            elif oid in opened_orders:
                return opened_orders[oid]
            elif oid in closed_orders:
                return closed_orders[oid]
        elif eoid:
            for orders_dict in [opened_orders, closed_orders]:
                orders = list(orders_dict.values())
                for order in orders:
                    if eoid == order.eoid:
                        return order

    def get_orders(self, trading_venue, acc, pdt='', oid='', eoid='', is_opened_only=False):
        if oid or eoid:
            return self.get_order(trading_venue, acc, oid=oid, eoid=eoid)
        else:
            orders = list(self.opened_orders[trading_venue][acc].values())
            if not is_opened_only:
                orders += list(self.submitted_orders[trading_venue][acc].values())
            return orders if not pdt else [order for order in orders if order.pdt == pdt]

    # FIXME
    def schedule_jobs(self, scheduler: BackgroundScheduler):
        scheduler.add_job(self.clear_cached_closed_orders, 'interval', seconds=10)
        scheduler.add_job(self.check_missed_opened_orders, 'interval', seconds=10)

    def reconcile(self, trading_venue, acc, data: dict):
        '''
        Remove orders that exist only internally but not externally
        '''
        for pdt, updates in data.items():
            official_eoids = [ update['eoid'] for update in updates if update['eoid'] ]
            opened_orders = self.get_orders(trading_venue, acc, pdt=pdt, is_opened_only=True)
            for o in opened_orders:
                if o.eoid and o.eoid not in official_eoids:
                    self._recon_nums[o.oid] += 1
                    self._logger.debug(f'{trading_venue} {acc=} {o.oid=} {o.eoid=} add recon_num to {self._recon_nums[o.oid]}')
                    if self._recon_nums[o.oid] >= self._MAX_RECON_NUM:
                        self._logger.warning(f'cancel order {o.oid} due to reconciliation')
                        self._broker.cancel_orders(o.account, o.product, [o])

    def clear_cached_closed_orders(self):
        for tv in self.closed_orders:
            for acc in self.closed_orders[tv]:
                orders_dict = self.closed_orders[tv][acc]
                oids = list(orders_dict)
                num_of_closed_orders = len(oids)
                if num_of_closed_orders > self._MAX_NUM_OF_CLOSED_ORDERS:
                    oids_to_be_removed = oids[:-self._NUM_OF_CACHED_CLOSED_ORDERS]
                    for k in oids_to_be_removed:
                        del orders_dict[k]
                    self._logger.debug(f'{tv} {acc=} clear closed orders cache')

    def check_missed_opened_orders(self):
        now = time.time()
        for tv in self.submitted_orders:
            for acc in self.submitted_orders[tv]:
                submitted_orders = list(self.submitted_orders[tv][acc].values())
                for o in submitted_orders:
                    submitted_ts = o.timestamps[MainOrderStatus.SUBMITTED]
                    if submitted_ts - now > self._NO_ORDER_UPDATE_TOLERANCE_IN_SECONDS:
                        self._missed_nums[o.oid] += 1
                        self._logger.debug(f'{tv} {acc=} {o.oid} add missed_num to {self._missed_nums[o.oid]}')
                        if self._missed_nums[o.oid] >= self._MAX_MISSED_NUM:
                            self._logger.warning(f'cancel order {o.oid} due to missing update')
                            self._broker.cancel_orders(o.account, o.product, [o])
                            self._on_rejected(o, ts=now, reason='MISSED')

    def handle_msgs(self, topic, info):
        if topic == 1:
            bkr, exch, acc, orders = info
            trading_venue = exch if bkr == 'CRYPTO' else bkr
            self.update_orders(trading_venue, acc, orders)
    

    '''
    Order Cycle
    '''
    def update_orders(self, trading_venue, acc, orders: dict):
        ts = orders['ts']
        data = orders['data']
        source: OrderUpdateSource = orders['source']
        for pdt, updates in data.items():
            for update in updates:
                if not (order := self.get_order(trading_venue, acc, oid=update['oid'], eoid=update['eoid'])) and source == OrderUpdateSource.GOO:
                    order = self._broker.create_order(trading_venue, acc, pdt, **update)
                else:
                    # orders from trade history could be closed already, 
                    # if they are still open, GTH can correct the trade details,
                    # if not, no need to re-open them
                    if source == OrderUpdateSource.GTH:
                        pass
                    else:
                        self._logger.error(f'Cannot find order {trading_venue=} {acc=} {pdt} {update=}')
                    return

                # NOTE: add/remove_order and on_update logic, analogous to add_position and position/balance.on_update() in portfolio_manager
                order.eoid = update['eoid']
                if 'status' in update:
                    main_status, fill_status, cancel_status, amend_status = update['status']
                else:
                    main_status = fill_status = cancel_status = amend_status = ''
            
                if main_status == 'O':
                    self._on_opened(order, ts=ts, reason=source)
                elif main_status == 'R':
                    self._on_rejected(order, ts=ts, reason=source)
                
                # msgs from both OrderUpdateSource.GTH and OrderUpdateSource.WST do not have 'status'
                # will call order.is_filled() to determine 
                if fill_status in ['P', 'F'] or source in [OrderUpdateSource.GTH, OrderUpdateSource.WST]:
                    self._on_trade(
                        order, fill_status, 
                        update['avg_px'], update['filled_qty'], 
                        update['ltp'], update['ltq'],
                        ts=ts, reason=source
                    )
                
                if cancel_status == 'C':
                    self._on_cancelled(order, ts=ts, reason=source)
                elif cancel_status == 'R':
                    self._on_cancel_rejected(order, ts=ts, reason=source)
                
                if amend_status == 'A':
                    self._on_amended(order, ts=ts, reason=source)
                elif amend_status == 'R':
                    self._on_amend_rejected(order, ts=ts, reason=source)

        if source == OrderUpdateSource.GOO:
            self.reconcile(trading_venue, acc, data)

    # main order status updates
    def on_submitted(self, order, ts=None, reason=''):
        if is_updated := order.on_status_update(status=MainOrderStatus.SUBMITTED, ts=ts, reason=reason):
            self.submitted_orders[order.tv][order.acc][order.oid] = order
            self.push(order, event=Event.order, type_='submitted')

    def _on_opened(self, order, ts=None, reason=''):
        if order.oid in self.closed_orders[order.tv][order.acc]:
            self._logger.warning(f'closed order {order} is trying to be re-opened, probably due to a delayed update')
            return
        if is_updated := order.on_status_update(status=MainOrderStatus.OPENED, ts=ts, reason=reason):
            self.opened_orders[order.tv][order.acc][order.oid] = order
            self.push(order, event=Event.order, type_='opened')
        if order.oid in self.submitted_orders[order.tv][order.acc]:
            del self.submitted_orders[order.tv][order.acc][order.oid]

    def _on_closed(self, order, ts=None, reason: OrderClosedReason | str=''):
        # open the order before closing it to complete the order cycle, except reason = rejected
        if order.oid not in self.opened_orders[order.tv][order.acc] and reason != MainOrderStatus.REJECTED:
            self._on_opened(order, ts=ts, reason='open to be closed')
        if is_updated := order.on_status_update(status=MainOrderStatus.CLOSED, ts=ts, reason=reason):
            self.closed_orders[order.tv][order.acc][order.oid] = order
            self.push(order, event=Event.order, type_='closed')
        if order.oid in self.submitted_orders[order.tv][order.acc]:
            del self.submitted_orders[order.tv][order.acc][order.oid]
        if order.oid in self.opened_orders[order.tv][order.acc]:
            del self.opened_orders[order.tv][order.acc][order.oid]

    def _on_rejected(self, order, ts=None, reason=''):
        if is_updated := order.on_status_update(status=MainOrderStatus.REJECTED, ts=ts, reason=reason):
            self._on_closed(order, reason=MainOrderStatus.REJECTED)

    def _on_trade(self, order, fill_status, avg_px, filled_qty, last_traded_px, last_traded_qty, ts=None, reason=''):
        if is_updated := order.on_trade_update(avg_px, filled_qty, last_traded_px, last_traded_qty):
            if fill_status == 'P' or not order.is_filled():
                self._on_partial(order, ts=ts, reason=reason)
                self.push(order, event=Event.trade, type_='partial')
            elif fill_status == 'F' or order.is_filled():
                self._on_filled(order, ts=ts, reason=reason)
                self.push(order, event=Event.trade, type_='filled')
                    
    def _on_partial(self, order, ts=None, reason=''):
        order.on_status_update(status=FillOrderStatus.PARTIAL, ts=ts, reason=reason)

    def _on_filled(self, order, ts=None, reason=''):
        if is_updated := order.on_status_update(status=FillOrderStatus.FILLED, ts=ts, reason=reason):
            self._on_closed(order, reason=FillOrderStatus.FILLED)

    # cancel order status updates
    def on_cancel(self, order, ts=None, reason=''):
        order.on_status_update(status=CancelOrderStatus.SUBMITTED, ts=ts, reason=reason)

    def _on_cancelled(self, order, ts=None, reason=''):
        if is_updated := order.on_status_update(status=CancelOrderStatus.CANCELLED, ts=ts, reason=reason):
            self._on_closed(order, reason=CancelOrderStatus.CANCELLED)
        
    def _on_cancel_rejected(self, order, ts=None, reason=''):
        order.on_status_update(status=CancelOrderStatus.REJECTED, ts=ts, reason=reason)
    
    # amend order status updates
    def on_amend(self, order, ts=None, reason=''):
        order.on_status_update(status=AmendOrderStatus.SUBMITTED, ts=ts, reason=reason)

    def _on_amended(self, order, ts=None, reason=''):
        if is_updated := order.on_status_update(status=AmendOrderStatus.AMENDED, ts=ts, reason=reason):
            self.push(order, event=Event.order, type_='amended')

    def _on_amend_rejected(self, order, ts=None, reason=''):
        order.on_status_update(status=AmendOrderStatus.REJECTED, ts=ts, reason=reason)
