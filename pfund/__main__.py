def run_cli() -> None:
    """Application Entrypoint."""
    from pfund.cli import pfund_group
    pfund_group(obj={})


if __name__ == '__main__':
    run_cli()
