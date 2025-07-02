from __future__ import annotations
from typing import TYPE_CHECKING, Union, Literal
if TYPE_CHECKING:
    from zmq import SocketType, SocketOption

import time
from enum import StrEnum
from collections import defaultdict

import zmq
import orjson

from pfund.enums import PublicDataChannel, PrivateDataChannel, PFundDataChannel, PFundDataTopic


JSONValue = Union[dict, list, str, int, float, bool, None]
DataChannel = Union[PFundDataChannel, PublicDataChannel, PrivateDataChannel]
class SocketMethod(StrEnum):
    bind = "bind"
    connect = "connect"


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
        *,
        sender_method: SocketMethod | Literal['bind', 'connect']='bind',
        receiver_method: SocketMethod | Literal['bind', 'connect']='connect',
        sender_type: SocketType | None=None,
        receiver_type: SocketType | None=None,
    ):
        '''
        Args:
            port: If not provided, a random port will be assigned.
        '''
        assert any([sender_type, receiver_type]), 'Either sender_type or receiver_type must be provided'
        self._name = name
        self._url = url
        self._ctx = zmq.Context(io_threads=io_threads)
        self._socket_methods: dict[zmq.Socket, SocketMethod] = {}
        self._socket_ports: defaultdict[zmq.Socket, list[int]] = defaultdict(list)
        self._sender: zmq.Socket | None = None
        self._receiver: zmq.Socket | None = None
        self._poller: zmq.Poller | None = None
        
        if sender_type:
            self._sender = self._ctx.socket(sender_type)
            self._socket_methods[self._sender] = SocketMethod[sender_method.lower()]
            # only queue outgoing messages once the remote peer is fully connected—otherwise block (or error) instead of buffering endlessly.
            self._sender.setsockopt(zmq.IMMEDIATE, 1)
        if receiver_type:
            self._receiver = self._ctx.socket(receiver_type)
            self._socket_methods[self._receiver] = SocketMethod[receiver_method.lower()]
            self._poller = zmq.Poller()
            self._poller.register(self._receiver, zmq.POLLIN)
        
    @property
    def name(self) -> str:
        return self._name
    
    def bind(self, socket: zmq.Socket, port: int | None=None):
        '''Binds a socket which uses bind method to a port.'''
        assert socket in self._socket_methods, f'{socket=} has not been initialized'
        assert self._socket_methods[socket] == SocketMethod.bind, f'{socket=} is not a socket used for binding'
        if port is None:
            port: int = socket.bind_to_random_port(self._url)
        else:
            socket.bind(f"{self._url}:{port}")
        if port not in self._socket_ports[socket]:
            self._socket_ports[socket].append(port)
        else:
            raise ValueError(f'{port=} is already bound')
    
    def subscribe(self, socket: zmq.Socket, port: int, channel: str=''):
        '''Subscribes to a port which uses connect method.'''
        assert socket in self._socket_methods, f'{socket=} has not been initialized'
        assert self._socket_methods[socket] == SocketMethod.connect, f'{socket=} is not a socket used for connecting'
        socket.connect(f"{self._url}:{port}")
        if socket.socket_type in [zmq.SUB, zmq.XSUB]:
            option: SocketOption = zmq.SUBSCRIBE
            value: int | bytes | str = channel.encode()
            socket.setsockopt(option, value)
        elif socket.socket_type == zmq.PULL:
            assert not channel, 'channel is only supported for SUB socket'
        else:
            raise ValueError(f'{socket.socket_type=} is not supported for subscription')
        if port not in self._socket_ports[socket]:
            self._socket_ports[socket].append(port)
        else:
            raise ValueError(f'{port=} is already subscribed')
    
    def terminate(self):
        if self._poller and self._receiver:
            self._poller.unregister(self._receiver)

        # terminate sockets
        for socket in [self._sender, self._receiver]:
            if socket is None:
                continue
            for port in self._socket_ports[socket]:
                if self._socket_methods[socket] == SocketMethod.bind:
                    socket.unbind(f"{self._url}:{port}")
                else:
                    socket.disconnect(f"{self._url}:{port}")
            socket.setsockopt(zmq.LINGER, 5000)  # wait up to 5 seconds, then close anyway
            socket.close()
        time.sleep(0.5)  # give zmq some time to clean up

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
        # TODO handle zmq.error.Again
        self._sender.send_multipart(msg, zmq.NOBLOCK)
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
        else:
            # avoids busy-waiting
            # REVIEW: not sure if this is enough, monitor CPU usage; if not enough, sleep 1ms
            time.sleep(0)
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
    