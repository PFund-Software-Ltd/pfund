from pfund.engines.trade_engine import TradeEngine
from pfund.config_handler import ConfigHandler


class TestEngine(TradeEngine):
    def __new__(cls, *, zmq_port=5557, config: ConfigHandler | None=None, **settings):
        return super().__new__(cls, env='TEST', zmq_port=zmq_port, config=config, **settings)
    
    def __init__(self, *, zmq_port=5557, config: ConfigHandler | None=None, **settings):
        super().__init__(env='TEST', zmq_port=zmq_port)
        # avoid re-initialization to implement singleton class correctly
        # if not hasattr(self, '_initialized'):
        #     pass
