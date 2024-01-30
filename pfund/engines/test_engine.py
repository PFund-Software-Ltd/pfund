from pfund.engines.trade_engine import TradeEngine


class TestEngine(TradeEngine):
    def __new__(cls, *, zmq_port=5557, **configs):
        return super().__new__(cls, env='TEST', zmq_port=zmq_port, **configs)        
    
    def __init__(self, *, zmq_port=5557, **configs):
        super().__init__(env='TEST', zmq_port=zmq_port, **configs)
        # avoid re-initialization to implement singleton class correctly
        if not hasattr(self, '_initialized'):
            pass
