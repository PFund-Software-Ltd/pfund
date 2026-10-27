from __future__ import annotations
from typing import NamedTuple, Literal

from pfund.utils.zmq_graph import show_zmq_graph


__all__ = [
    "Timestamp",
    "show_zmq_graph",
]


# used to tell if a timestamp is from the venue or created by pfund internally
class Timestamp(NamedTuple):
    value: float
    source: Literal["venue", "pfund"]
