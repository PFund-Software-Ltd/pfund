# pyright: reportMissingTypeArgument=false, reportUnknownParameterType=false, reportUnknownMemberType=false
import logging
import time

import zmq
from msgspec import json
# from zmq.log.handlers import PUBHandler

from pfund.enums import PFundDataChannel


class ZMQPushHandler(logging.Handler):
    '''Custom logging handler that sends log messages over a ZMQ PUSH socket.
    Messages follow the internal [channel, topic, data, ts] format.
    '''
    def __init__(self, interface_or_socket: str | zmq.Socket, context: zmq.Context | None = None):
        super().__init__()
        if isinstance(interface_or_socket, zmq.Socket):
            self.socket = interface_or_socket
            self.ctx = self.socket.context
        else:
            self.ctx = context or zmq.Context()
            self.socket = self.ctx.socket(zmq.PUSH)
            self.socket.connect(interface_or_socket)
        # buffer to cache log messages until the zmq receiver is ready
        self._buffer: list[tuple[bytes, bytes, bytes, bytes]] = []
        self._is_receiver_ready = False

    def emit(self, record: logging.LogRecord) -> None:
        try:
            channel = PFundDataChannel.logging
            topic = record.levelname
            text = self.format(record)
            data = json.encode(text)
            ts = time.time()
            msg = (
                channel.encode(),
                topic.encode(),
                data,
                f"{ts}".encode()
            )
            if not self._is_receiver_ready:
                self._buffer.append(msg)
            else:
                self.socket.send_multipart(msg, zmq.NOBLOCK)
        except Exception:
            self.handleError(record)

    def set_receiver_ready(self) -> None:
        # drain the backlog exactly once
        for msg in self._buffer:
            self.socket.send_multipart(msg, zmq.NOBLOCK)
        self._buffer.clear()
        self._is_receiver_ready = True

    def close(self) -> None:
        self.socket.close(linger=5000)  # wait up to 5 seconds, then close anyway
        super().close()
