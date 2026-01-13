from pfund_kit import AliasRegistry


ALIASES = AliasRegistry({
    'proj': 'project',
    'config': 'configuration',
    'const': 'constant',
    'dir': 'directory',
    'env': 'environment',

    'multi': 'multiplier',
    'tfee': 'taker_fee',
    'mfee': 'maker_fee',
    'bps': 'basis_points',
    'lsize': 'lot_size',
    'tsize': 'tick_size',
    
    'bkr': 'broker',
    'exch': 'exchange',
    'strat': 'strategy',
    'pdt': 'product',
    'epdt': 'external_product',
    'bccy': 'base_currency',
    'qccy': 'quote_currency',
    'ccy': 'currency',
    'eccy': 'external_currency',
    'px': 'price',
    'qty': 'quantity',
    'ptype': 'product_type',
    'eptype': 'external_product_type',
    'otype': 'order_type',
    'acc': 'account',
    'mdl': 'model',
    'feat': 'feature',

    'o': 'order',
    'ltp': 'last_traded_price',
    'ltq': 'last_traded_quantity',
    'lts': 'last_traded_size',
    'ltt': 'last_traded_time',

    'CW': 'current_week',
    'NW': 'next_week',
    'CM': 'current_month',
    'NM': 'next_month',
    'CQ': 'current_quarter',
    'NQ': 'next_quarter',

    'ret': 'return',
    'res': 'result',
    'resp': 'response',
    'req': 'request',

    'prev': 'previous',
    'thd': 'thread',
    'cb': 'callback',
    'om': 'order_manager',

    'ack': 'acknowledge',
    'acked': 'acknowledged',
    'auth': 'authentication',
    'authed': 'authenticated',

    'ws': 'websocket',
    'rest': 'rest_api',

    'sub': 'subscribe',
    'ttl': 'total',
    'freq': 'frequency',
    'num': 'number',
    'tf': 'timeframe',
    'agg': 'aggregate',
    'resol': 'resolution',

    'IBKR': 'Interactive Brokers',

    'bg': 'background',
    'boa': 'bid_or_ask',

    'tv': 'trading_venue',
})
