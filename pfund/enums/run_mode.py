from enum import StrEnum


class RunMode(StrEnum):
    LOCAL = 'LOCAL'  # = no ray + zeromq
    REMOTE = 'REMOTE'  # = ray actor + zeromq
    WASM = 'WASM'  # = no ray + no zeromq
