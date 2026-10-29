# trading venue plugins, each module runs in its own folder with its own pixi env
# e.g. `just bybit` opens a shell in pfund-bybit's default env, `exit` goes back to pfund
#      (don't `cd ../..`, it keeps pfund-bybit's env active)
#      `just bybit test` runs pfund-bybit's tests
mod bybit 'venues/pfund-bybit'
mod hyperliquid 'venues/pfund-hyperliquid'
mod hl 'venues/pfund-hyperliquid'
mod okx 'venues/pfund-okx'
mod binance 'venues/pfund-binance'
mod ba 'venues/pfund-binance'
mod alpaca 'venues/pfund-alpaca'
mod ibkr 'venues/pfund-ibkr'
mod ib 'venues/pfund-ibkr'
