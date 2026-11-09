# trading venue plugins, each module runs in its own folder with its own pixi env
# e.g. `just bybit` opens a shell in pfund-bybit's default env, `exit` goes back to pfund
#      (don't `cd ../..`, it keeps pfund-bybit's env active)
#      `just bybit test` runs pfund-bybit's tests
mod? bybit 'venues/pfund-bybit'
mod? hyperliquid 'venues/pfund-hyperliquid'
mod? hl 'venues/pfund-hyperliquid'
mod? okx 'venues/pfund-okx'
mod? binance 'venues/pfund-binance'
mod? alpaca 'venues/pfund-alpaca'
mod? ibkr 'venues/pfund-ibkr'


# upgrade one package in uv.lock and pixi.lock, bypassing cached PyPI metadata
# (pixi has no per-package refresh, so its whole PyPI cache is cleaned to see a new release)
lock package:
    uv lock --refresh-package {{package}} --upgrade-package {{package}}
    pixi clean cache --pypi -y
    pixi update {{package}}
