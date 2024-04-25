from pfund.exchanges.binance.exchange import Exchange


class ExchangeLinear(Exchange):
    def __init__(self, env: str, ptype: str):
        super().__init__(env, ptype)

