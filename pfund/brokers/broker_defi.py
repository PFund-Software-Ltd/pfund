from pfund.brokers.broker_live import LiveBroker


# TODO
class DefiBroker(LiveBroker):
    def __init__(self, env: str):
        super().__init__(env, 'DEFI')
        