# Sourced by pixi on activation (pixi run / pixi shell), see [target.unix.activation] in pixi.toml.
# ty and afterpython's hooks (e.g. `ap check`) look for the project environment in .venv/,
# so point .venv at pixi's default environment, which has the dev/test packages.
# A real .venv directory (e.g. created by `uv sync`) is left alone.
if [ "$PIXI_ENVIRONMENT_NAME" = "default" ] && { [ ! -d "$PIXI_PROJECT_ROOT/.venv" ] || [ -L "$PIXI_PROJECT_ROOT/.venv" ]; }; then
    # relative, so the link survives moving the repo
    ln -sfn .pixi/envs/default "$PIXI_PROJECT_ROOT/.venv"
fi
