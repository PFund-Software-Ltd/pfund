import logging
import time

import orjson
from zmq.log.handlers import PUBHandler

from pfund.enums import PFundDataChannel


class ZMQPubHandler(PUBHandler):
    '''Custom PUBHandler that sends log messages that follow the internal format to ZMQ'''
    def __init__(self, interface_or_socket, **kwargs):
        super().__init__(interface_or_socket, **kwargs)
        # buffer to cache log messages until the zmq receiver is ready
        self._buffer: list[tuple[bytes, bytes, bytes, bytes]] = []
        self._is_receiver_ready = False
        
    def emit(self, record: logging.LogRecord):
        try:
            channel = PFundDataChannel.logging
            topic = record.levelname
            text = self.format(record)
            data = orjson.dumps(text)
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
                self.socket.send_multipart(msg)
        except Exception:
            self.handleError(record)
            
    def set_receiver_ready(self):
        # drain the backlog exactly once
        for msg in self._buffer:
            self.socket.send_multipart(msg)
        self._buffer.clear()
        self._is_receiver_ready = True