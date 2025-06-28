from __future__ import annotations
from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:
    from zmq import SocketType, SocketOption

import time

import orjson
import zmq

from pfund.enums import PublicDataChannel, PrivateDataChannel, PFundDataChannel, PFundDataTopic


JSONValue = Union[dict, list, str, int, float, bool, None]
DataChannel = Union[PFundDataChannel, PublicDataChannel, PrivateDataChannel]


# TODO: need to tunnel zeromq connections with SSH when communicating with remote ray actors
# see https://pyzmq.readthedocs.io/en/latest/howto/ssh.html
class ZeroMQ:
    '''
    Thin wrapper of zmq to handle sockets for both sending and receiving messages with exception handling.
    '''
    DEFAULT_URL = 'tcp://localhost'

    def __init__(
        self,
        name: str,
        url: str=DEFAULT_URL,
        io_threads: int=1,
        port: int | None=None,
        sender_socket_type: SocketType | None=None,
        receiver_socket_type: SocketType | None=None,
    ):
        '''
        Args:
            port: If not provided, a random port will be assigned.
        '''
        self._name = name
        self._sender_port: int | None = port
        self._receiver_ports: list[int] = []
        self._url = url
        self._ctx = zmq.Context(io_threads=io_threads)
        assert any([sender_socket_type, receiver_socket_type]), 'Either sender_socket_type or receiver_socket_type must be provided'
        self._sender = self._ctx.socket(sender_socket_type) if sender_socket_type else None
        if self._sender:
            if not self._sender_port:
                self._sender_port: int = self._sender.bind_to_random_port(self._url)
            else:
                self._sender.bind(f"{self._url}:{self._sender_port}")
        self._receiver = self._ctx.socket(receiver_socket_type) if receiver_socket_type else None
        self._poller = zmq.Poller() if self._receiver else None
        
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def port(self) -> int | None:
        return self._sender_port
    
    def subscribe(self, port: int, channel: str=''):
        if self._receiver.socket_type == zmq.SUB:
            option: SocketOption = zmq.SUBSCRIBE
            value: int | bytes | str = channel.encode()
            self._receiver.setsockopt(option, value)
        elif self._receiver.socket_type in [zmq.PULL, zmq.XSUB]:
            assert not channel, 'channel is only supported for SUB socket'
        else:
            raise ValueError(f'{self._receiver.socket_type=} is not supported')
        if port not in self._receiver_ports:
            self._receiver_ports.append(port)
        else:
            raise ValueError(f'{port=} is already subscribed')
    
    def __str__(self):
        return f'ZeroMQ({self.name}_zmq): send using port {self._sender_port}, receive from ports {self._receiver_ports}'

    def start(self):
        if self._poller:
            self._poller.register(self._receiver, zmq.POLLIN)
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

    def send(self, channel: DataChannel, topic: PFundDataTopic | str, data: JSONValue) -> None:
        '''
        Sends message to receivers
        Args:
            data: A JSON serializable object.
            topic: A message key used to group messages within a channel.
        '''
        send_ts = time.time()
        msg = [channel.encode(), topic.encode(), orjson.dumps(data), f"{send_ts}".encode()]
        self._sender.send_multipart(msg)
        # TODO: handle exception:
        # try:
        # except zmq.ZMQError as e:
        # if e.errno == zmq.ETERM:
        #     break           # Interrupted
        # else:
        #     raise

    def recv(self) -> tuple[DataChannel, PFundDataTopic | str, JSONValue, float]:
        events = self._poller.poll(0)  # 0 sec timeout, non-blocking
        if events:
            msg = self._receiver.recv_multipart(zmq.DONTWAIT)
            channel, topic, data, pub_ts = msg
            channel, topic, pub_ts = channel.decode(), topic.decode(), float(pub_ts.decode())
            data = orjson.loads(data)
            return channel, topic, data, pub_ts
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
        # TODO: should catch the exception at a higher level where the logger is available
        #     self._logger.exception(f'{self.name}_zmq recv exception:')
    