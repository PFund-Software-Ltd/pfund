import logging


class ColoredFormatter(logging.Formatter):
    # NOTE: see ANSI_codes.py for more options
    COLORS = {
        'WARNING': '\033[1;93m',   # Bold Yellow
        'ERROR': '\033[1;91m',     # Bold Red
        'CRITICAL': '\033[1;95m',  # Bold Magenta
        'ENDC': '\033[0m',         # End of formatting
    }

    def format(self, record):
        log_message = super().format(record)
        color_code = self.COLORS.get(record.levelname, '')
        end_code = self.COLORS['ENDC'] if color_code else '' 
        return f"{color_code}{log_message}{end_code}"