import time

from typing import Any

try:
    import zmq
except ImportError:
    pass


class ZeroMQ:
    def __init__(self, name: str):
        self.name = name.lower()
        self._logger = None
        self._send_port = None
        self._recv_ports = []
        self._sender_context = zmq.Context()
        self._sender = self._sender_context.socket(zmq.PUB)
        self._receiver_context = zmq.Context()
        self._receiver = self._receiver_context.socket(zmq.SUB)
        self._receiver.setsockopt(zmq.SUBSCRIBE, b'')
        self._poller = zmq.Poller()
    
    def __str__(self):
        return f'ZeroMQ({self.name}_zmq): send using port {self._send_port}, receive from ports {self._recv_ports}'

    def _set_logger(self, logger):
        self._logger = logger

    def start(self, logger, send_port: int, recv_ports: list[int]):
        self._set_logger(logger)
        self._poller.register(self._receiver, zmq.POLLIN)
        self._send_port = send_port
        self._recv_ports = recv_ports
        self._logger.debug(f'{self.name}_zmq set send_port to {self._send_port}, recv ports to {self._recv_ports}')
        if self._send_port:
            send_address = f"tcp://127.0.0.1:{self._send_port}"
            self._sender.bind(send_address)
            self._logger.debug(f'{self.name}_zmq sender binded to {send_address}')
        for port in self._recv_ports:
            recv_address = f"tcp://127.0.0.1:{port}"
            self._receiver.connect(recv_address)
            self._logger.debug(f'{self.name}_zmq receiver connected to {recv_address}')
        self._logger.debug(f'{self.name}_zmq started')
        time.sleep(1)  # give zmq some prep time, prevent it from missing messages in the beginning

    def stop(self):
        self._poller.unregister(self._receiver)

        # terminate sender
        send_address = f"tcp://127.0.0.1:{self._send_port}"
        self._sender.unbind(send_address)
        self._logger.debug(f'{self.name}_zmq sender unbinded from {send_address}')
        self._sender.close()
        time.sleep(0.1)
        self._sender_context.term()

        # terminate receiver
        for port in self._recv_ports:
            recv_address = f"tcp://127.0.0.1:{port}"
            self._receiver.disconnect(recv_address)
            self._logger.debug(f'{self.name}_zmq receiver disconnected from {recv_address}')
        self._receiver.close()
        time.sleep(0.1)
        self._receiver_context.term()
        self._logger.debug(f'{self.name}_zmq stopped')
        self._logger = None

    def send(self, channel: int, topic: int, info: Any, receiver: str='') -> None:
        """Sends message to receivers/subscribers
        Args:
            receiver:
                If not empty, sends messages to a specific receiver;
                Otherwise, sends to all.
        """
        if not self._sender.closed:
            msg = (time.time(), receiver, channel, topic, info)
            self._sender.send_pyobj(msg)
            self._logger.debug(f'{self.name}_zmq sent {msg}')
        else:
            self._logger.debug(f'{self.name}_zmq _sender is closed, cannot not send out msg {msg}')

    def recv(self):
        try:
            events = self._poller.poll(0)  # 0 sec timeout
            if events and not self._receiver.closed:
                msg = self._receiver.recv_pyobj()
                ts, receiver, channel, topic, info = msg
                # TODO monitor latency using ts
                if not receiver or receiver.lower() == self.name:
                    self._logger.debug(f'{self.name}_zmq recv {msg}')
                    return channel, topic, info
        except KeyboardInterrupt:
            # need to close the sockets and terminate the contexts
            # otherwise, zmq will probably be stuck at somewhere in poll()
            # and can't exit the program
            self.stop()
            raise KeyboardInterrupt
        except:
            self._logger.exception(f'{self.name}_zmq recv exception:')
    
    # TODO, monitor zmq latency
    def _monitor(self):
        pass