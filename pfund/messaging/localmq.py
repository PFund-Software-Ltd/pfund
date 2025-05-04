import time
import logging
from queue import Queue, Empty
from collections import deque

from pfund.messaging.basemq import BaseMQ


'''
# TEMP
def __init__(self):
    self.subscribers = []

def subscribe(self):
    q = queue.Queue()
    self.subscribers.append(q)
    return q

def publish(self, message):
    for q in self.subscribers:
        q.put(message)
'''
class LocalMQ(BaseMQ):
    '''
    Local message queue
    '''
    def __init__(self, name: str, logger: logging.Logger):
        self.name = name.lower()
        self._logger = logger
        self._queue = Queue()
    
    def start(self, send_port=None, recv_ports=None):
        # LocalMQ doesn’t bind to real ports — nothing to do here.
        self._logger.debug(f"{self.name}_localmq started (in‑proc queue)")

    def stop(self):
        # nothing to tear down
        self._logger.debug(f"{self.name}_localmq stopped")

    def send(self, channel: int, topic: int, info: any, receiver: str = '') -> None:
        """Mimic ZeroMQ.send: pkg up the same tuple and put on the queue."""
        msg = (time.time(), receiver, channel, topic, info)
        self._queue.put(msg)
        self._logger.debug(f"{self.name}_localmq sent {msg}")

    def recv(self):
        """Mimic ZeroMQ.recv: non‑blocking get + same filtering logic."""
        try:
            ts, receiver, channel, topic, info = self._queue.get_nowait()
        except Empty:
            return None

        # same “receiver” filtering as ZeroMQ
        if not receiver or receiver.lower() == self.name:
            self._logger.debug(f"{self.name}_localmq recv {(ts, receiver, channel, topic, info)}")
            return channel, topic, info
        # otherwise drop it
        return None