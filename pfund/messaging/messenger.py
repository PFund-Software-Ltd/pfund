from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.typing import tTRADING_VENUE, EngineName, ComponentName
    from pfund.enums import RunMode
    from mtflow.typing import tZMQ_MESSENGER
    
import logging

from mtflow.enums import ZMQMessenger


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class Messenger:
    '''
    Handles message flows in the trade engine.
    '''
    
    @staticmethod
    def _is_port_in_use(port):
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    def _assign_zmq_ports(self, zmq_port: int=5557) -> dict:
        _assigned_ports = []
        def _is_port_available(_port):
            _is_port_assigned = (_port in _assigned_ports)
            if self._is_port_in_use(_port) or _is_port_assigned:
                return False
            else:
                _assigned_ports.append(_port)
                return True
        def _get_port(start_port=None):
            _port = start_port or zmq_port
            if _is_port_available(_port):
                return _port
            else:
                return _get_port(start_port=_port+1)
        self.settings['zmq_ports']['engine'] = _get_port()
        for broker in self.brokers.values():
            if broker.name == 'CRYPTO':
                for exchange in broker.exchanges.values():
                    self.settings['zmq_ports'][exchange.name] = {'rest_api': _get_port()}
                    if not exchange.use_separate_private_ws_url():
                        self.settings['zmq_ports'][exchange.name]['ws_api'] = _get_port()
                    else:
                        self.settings['zmq_ports'][exchange.name]['ws_api'] = {'public': {}, 'private': {}}
                        ws_servers = exchange.get_ws_servers()
                        for ws_server in ws_servers:
                            self.settings['zmq_ports'][exchange.name]['ws_api']['public'][ws_server] = _get_port()
                        for acc in exchange.accounts.keys():
                            self.settings['zmq_ports'][exchange.name]['ws_api']['private'][acc] = _get_port()
            else:
                self.settings['zmq_ports'][broker.name] = _get_port()
        for strategy in self.strategy_manager.strategies.values():
            # FIXME:
            if self._use_ray:
                self.settings['zmq_ports'][strategy.name] = _get_port()
        self.logger.debug(f"{self.settings['zmq_ports']=}")

    def _ping_processes(self):
        self._zmq.send(0, 0, ('engine', 'ping',))
        
    def send(self, message: str):
        pass
    
    