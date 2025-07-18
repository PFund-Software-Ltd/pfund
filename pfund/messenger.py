from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.typing import ZeroMQName

import logging


class Messenger:
    '''
    Handles message flows in the trade engine.
    '''
    def __init__(self, zmq_url: str, zmq_ports: dict[ZeroMQName, int]):
        '''
        Args:
            proxy: ZeroMQ xsub-xpub proxy for messaging from trading venues -> engine -> components
            router: ZeroMQ router-pull for pulling messages from components (e.g. strategies/models) -> engine -> trading venues
            publisher: ZeroMQ publisher for broadcasting internal states to external apps
        '''
        import zmq
        from pfeed.messaging.zeromq import ZeroMQ
        
        self._logger = logging.getLogger('pfund')
        
        zmq_url = zmq_url or ZeroMQ.DEFAULT_URL
        
        proxy_name = 'proxy'
        self._proxy = ZeroMQ(
            name=proxy_name,
            url=zmq_url,
            port=zmq_ports.get(proxy_name, None),
            io_threads=2,
            receiver_socket_type=zmq.XSUB,  # msgs from trading venues -> engine
            sender_socket_type=zmq.XPUB,  # msgs from engine -> components
        )
        
        router_name = 'router'
        self._router = ZeroMQ(
            name=router_name,
            url=zmq_url,
            port=zmq_ports.get(router_name, None),
            receiver_socket_type=zmq.PULL,  # msgs (e.g. orders) from components -> engine
            # NOTE: exchange's private websocket is in the main thread, no need to use ROUTER for now
            # sender_socket_type=zmq.ROUTER,  # msgs (e.g. orders) from engine -> trading venues
        )
        
        publisher_name = 'publisher'
        self._publisher = ZeroMQ(
            name=publisher_name,
            url=zmq_url,
            port=zmq_ports.get(publisher_name, None),
            receiver_socket_type=zmq.SUB,  # subscribe to components' logger's zmq PUBHandler
            sender_socket_type=zmq.PUB,  # publish to external listeners
        )

        # NOTE: this is mutating engine's settings.zmq_ports
        self._zmq_ports_in_use: dict[ZeroMQName, int] = zmq_ports
        self._zmq_ports_in_use.update(
            {q.name: q.port for q in [self._proxy, self._router, self._publisher]}
        )

    def subscribe(self):
        from pfund.enums import TradingVenue
        self._logger.debug(f'complete ZeroMQ ports in use: {self._zmq_ports_in_use}')
        zmq_names_in_use = [q.name for q in [self._proxy, self._router, self._publisher]]
        for zmq_name, zmq_port in self._zmq_ports_in_use.items():
            # subscribe to trading venue's data
            # FIXME: proxy should not subscribe to trading venue's raw data directly
            # only subscribe to the standardized data sent from the data engine in pfeed
            # TODO: should update the zmq_ports_in_use after grouping streams in pfeed
            if zmq_name.upper() in TradingVenue.__members__:
                self._proxy.subscribe(zmq_port)
                self._logger.debug(f'{self._proxy.name} subscribed to {zmq_name} on port {zmq_port}')
            # subscribe to the component's logger's zmq PUBHandler
            elif zmq_name.endswith('_logger'):
                self._publisher.subscribe(zmq_port)
                self._logger.debug(f'{self._publisher.name} subscribed to {zmq_name} on port {zmq_port}')
            # subscribe to the component's data (e.g. orders)
            elif zmq_name not in zmq_names_in_use:
                self._router.subscribe(zmq_port)
                self._logger.debug(f'{self._router.name} subscribed to {zmq_name} on port {zmq_port}')
    
    def start(self):
        self._proxy.start()
        # TODO: create thread for proxy
        # FIXME: zmq.proxy(...)
        # self._proxy.run_proxy()
        self._router.start()
        self._publisher.start()

    def stop(self):
        self._proxy.stop()
        self._router.stop()
        self._publisher.stop()

    # TODO:
    def _ping_processes(self):
        self._zmq.send(0, 0, ('engine', 'ping',))
