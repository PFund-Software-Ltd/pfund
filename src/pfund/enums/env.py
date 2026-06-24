from enum import StrEnum


class Environment(StrEnum):
    BACKTEST = "BACKTEST"
    SANDBOX = "SANDBOX"
    PAPER = "PAPER"
    LIVE = "LIVE"

    def is_simulated(self):
        return self in [Environment.BACKTEST, Environment.SANDBOX]

    @property
    def _color(self):
        from pfund_kit.style import RichColor, TextStyle

        return {
            Environment.BACKTEST: TextStyle.BOLD + RichColor.BLUE,
            Environment.SANDBOX: TextStyle.BOLD + RichColor.GREY0,
            Environment.PAPER: TextStyle.BOLD + RichColor.RED,
            Environment.LIVE: TextStyle.BOLD + RichColor.GREEN,
        }[self]
