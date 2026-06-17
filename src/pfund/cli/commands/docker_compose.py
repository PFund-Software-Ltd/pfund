import click

from pfund_kit.cli.commands.docker_compose import docker_compose as _kit_docker_compose


# docker compose subcommands that (re)create containers — only these inject the
# ${IBKR_*} credentials/ports from the env file, so only these need a trading env
# chosen deliberately. Everything else (down, ps, logs, stop, pull, ...) just
# operates on the existing `pfund` project and passes straight through.
_ENV_FILE_REQUIRED_SUBCOMMANDS = {"up", "create", "run"}


@click.command(
    name="docker-compose",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.pass_context
def docker_compose(ctx):
    """Run docker compose for pfund (up/create/run require an explicit --env-file).

    Overrides pfund-kit's shared docker-compose command: pfund talks to one
    trading venue at a time and the trading env (paper/live) must be chosen
    deliberately, so an --env-file is mandatory for the subcommands that launch
    containers:

        pfund compose --env-file .env.paper up
        pfund compose --env-file .env.live up

    This guards against starting the gateway with the wrong (or no) trading-env
    credentials. The flag is passed straight through to `docker compose`, which
    uses it to interpolate ${IBKR_*} variables in compose.yml. Lifecycle commands
    that don't (re)create containers (down, ps, logs, stop, ...) need no env file.
    """
    launches_containers = any(arg in _ENV_FILE_REQUIRED_SUBCOMMANDS for arg in ctx.args)
    if launches_containers:
        has_env_file = any(
            arg == "--env-file" or arg.startswith("--env-file=") for arg in ctx.args
        )
        if not has_env_file:
            raise click.UsageError(
                "pfund compose up/create/run requires an explicit --env-file "
                "selecting the trading env, e.g. `pfund compose --env-file .env.paper up`"
            )
    # delegate to the shared pfund-kit command (handles --file, docker checks, run).
    # call its callback within the CURRENT context so our ctx.args (the user's
    # --env-file and compose subcommand) is preserved; ctx.forward/ctx.invoke would
    # build a fresh child context with empty args and silently drop them.
    _kit_docker_compose.callback()
