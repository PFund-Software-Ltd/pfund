from __future__ import annotations
from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:
    from pfund.enums.data_channel import PublicDataChannel, PrivateDataChannel
    from mtflow.enums.pfund_channel import PFundChannel
    from zmq import SocketType, SocketOption

import time
import logging

import orjson
import zmq


JSONValue = Union[dict, list, str, int, float, bool, None]


# TODO: need to tunnel zeromq connections with SSH when communicating with remote ray actors
# see https://pyzmq.readthedocs.io/en/latest/howto/ssh.html
class ZeroMQ:
    '''
    Thin wrapper of zmq to handle sockets for both sending and receiving messages with exception handling.
    '''
    DEFAULT_URL = 'tcp://localhost'

    def __init__(
        self, 
        url: str=DEFAULT_URL,
        io_threads: int=1,
        sender_socket_type: SocketType | None=None,
        receiver_socket_type: SocketType | None=None,
    ):
        self._logger = logging.getLogger('mtflow')
        self._sender_port: int | None = None
        self._receiver_ports: list[int] | None = None
        self._url = url
        self._ctx = zmq.Context(io_threads=io_threads)
        assert any([sender_socket_type, receiver_socket_type]), 'Either sender_socket_type or receiver_socket_type must be provided'
        self._sender = self._ctx.socket(sender_socket_type) if sender_socket_type else None
        self._receiver = self._ctx.socket(receiver_socket_type) if receiver_socket_type else None
        self._poller = zmq.Poller() if self._receiver else None
    
    @property
    def receiver(self) -> SocketType:
        return self._receiver
    
    @property
    def sender(self) -> SocketType:
        return self._sender
    
    def setsockopt(self, option: SocketOption, value: int | bytes | str):
        self._receiver.setsockopt(option, value)
    
    # FIXME
    def __str__(self):
        return f'ZeroMQ({self.name}_zmq): send using port {self._sender_port}, receive from ports {self._receiver_ports}'

    def start(self, sender_port: int | None=None, receiver_ports: list[int] | None=None):
        assert any([sender_port, receiver_ports]), 'Either sender_port or receiver_ports must be provided'
        self._sender_port = sender_port
        self._receiver_ports = receiver_ports
        if self._poller:
            self._poller.register(self._receiver, zmq.POLLIN)
        if self._sender:
            self._sender.bind(f"{self._url}:{self._sender_port}")
        for port in self._receiver_ports:
            self._receiver.connect(f"{self._url}:{port}")
        time.sleep(1)  # give zmq some prep time, e.g. allow subscription propagation in pub-sub case

    def stop(self):
        if self._poller:
            self._poller.unregister(self._receiver)

        # terminate sender
        self._sender.unbind(f"{self._url}:{self._sender_port}")
        self._sender.close()
        time.sleep(0.5)  # give zmq some time to clean up

        # terminate receiver
        for port in self._receiver_ports:
            self._receiver.disconnect(f"{self._url}:{port}")
        self._receiver.close()
        time.sleep(0.5)  
        
        # terminate context
        self._ctx.term()

    def send(self, channel: PFundChannel | PublicDataChannel | PrivateDataChannel, data: JSONValue, msg_key: str='') -> None:
        '''
        Sends message to receivers
        Args:
            data: A JSON serializable object.
            msg_key: A message key used to group messages within a channel.
        '''
        send_ts = time.time()
        msg = [channel.encode(), msg_key.encode(), orjson.dumps(data), f"{send_ts}".encode()]
        self._sender.send_multipart(msg)
        # TODO: handle exception:
        # try:
        # except zmq.ZMQError as e:
        # if e.errno == zmq.ETERM:
        #     break           # Interrupted
        # else:
        #     raise

    def recv(self) -> tuple[PFundChannel | PublicDataChannel | PrivateDataChannel, str, JSONValue, float]:
        events = self._poller.poll(0)  # 0 sec timeout
        if events:
            msg = self._receiver.recv_multipart(zmq.DONTWAIT)
            channel, msg_key, data, pub_ts = msg
            channel, msg_key, pub_ts = channel.decode(), msg_key.decode(), float(pub_ts.decode())
            data = orjson.loads(data)
            return channel, msg_key, data, pub_ts
        # TODO: handle exception:
        # try:
        # except zmq.error.Again:  # no message available, will be raised when using zmq.DONTWAIT
        #     pass
        # # TODO
        # # except zmq.error.ContextTerminated:
        # #     pass
        # except KeyboardInterrupt:
        #     # need to close the sockets and terminate the contexts
        #     # otherwise, zmq will probably be stuck at somewhere in poll()
        #     # and can't exit the program
        #     self.stop()
        #     raise KeyboardInterrupt
        # except:
        #     self._logger.exception(f'{self.name}_zmq recv exception:')
    