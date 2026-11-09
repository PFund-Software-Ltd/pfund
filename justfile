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

# afterpython/README.py is the marimo notebook for the project website's landing page, exported to WASM.
# wasm/serve are ONLY for testing the WASM locally (`just wasm && just serve`) before running the
# afterpython build; the real website export is done by `ap build` (to afterpython/_build/readme_py/),
# not by these recipes. Nothing here is used by the website.
# Output goes to build/ (gitignored), not afterpython/_build/: `ap build` wipes afterpython/_build/
readme_out := justfile_directory() / "build" / "README_py"

# Export afterpython/README.py to WASM, extra args go to marimo: `just wasm --mode edit`
[group('marimo')]
wasm *args:
    marimo export html-wasm --sandbox --mode run --show-code -f -o {{readme_out}} {{args}} {{justfile_directory()}}/afterpython/README.py

# Serve the WASM build and open it in the browser (it can't run from file://): `just serve --port 9000`
[group('marimo')]
[arg("port", long="port")]
serve port="8000":
    (sleep 1 && python3 -m webbrowser -t http://localhost:{{port}}) &
    python3 -m http.server {{port}} -d {{readme_out}}