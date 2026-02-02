from enum import StrEnum


class RunMode(StrEnum):
    '''
    Three aspects in the trade engine's runtime:
    1. Ray actors
        - Ray actors are used for running strategies/models
    2. ZeroMQ messaging
        - ZeroMQ is used for messaging b/w exchanges -> engine, and engine -> strategies (if using ray)
    '''
    LOCAL = 'LOCAL'  # no ray, but still use zeromq for sending messages from exchanges' websockets to engine
    REMOTE = 'REMOTE'  # ray + zeromq
