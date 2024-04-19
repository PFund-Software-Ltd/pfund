import atexit

from pfund.cli import pfund_group


def exit_cli():
    """Application Exitpoint."""
    print("Cleanup actions here...")


def run_cli() -> None:
    """Application Entrypoint."""
    # atexit.register(exit_cli)
    pfund_group(obj={})


if __name__ == '__main__':
    run_cli()