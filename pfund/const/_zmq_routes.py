"""Defines ZeroMQ channels and topics across the system

Since the messaging mechanism is very latency-sensitive,
direct usage of the following variables are not recommended,
but used as a reference instead.
"""


# how engine receives msgs:
ENGINE_ROUTES = {
    'strategy_manager': {
        'channel': 1,
        'topics': {
            'pong': 0,
        }
    },
    'connection_manager': {
        'channel': 2,
        'topics': {
            'pong': 0,
            'pid': 1,
            'on_connected': 2,
            'on_disconnected': 3,
        }
    },
    'data_manager': {
        'channel': 3,
        'topics': {
            'orderbook': 1,
            'tradebook': 2,
            'kline': 3,
        }
    },
    'portfolio_manager': {
        'channel': 4,
        'topic': {
            'balances': 1,
            'positions': 2,
        }
    }
}


# how apis in connection manager receives msgs:
API_ROUTES = {
    'monitor': {
        'channel': 0,
        'topics': {
            'ping': 0
        }
    },
    'orders': {
        'channel': 1,
        'topics': {
            'place_o': 1,
            'cancel_o': 2,
            'amend_o': 3,
        }
    },
}


# how strategies in strategy manager receives msgs:
STRATEGY_ROUTES = {
    'monitor': {
        'channel': 0,
        'topics': {
            'ping': 0
        }
    },
    'orders': {
        'channel': 1,
        'topics': {
            # FIXME
            'opened_orders': 1
        }
    },
    'positions': {
        'channel': 2,
        'topics': {}
    },
    'balances': {
        'channel': 3,
        'topics': {}
    },
    'data': {
        'channel': 4,
        'topics': {}
    },
}