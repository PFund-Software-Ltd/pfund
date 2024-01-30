from logging import Filter


class FullPathFilter(Filter):
    def filter(self, record):
        if 'site-packages/' in record.pathname:
            record.shortpath = record.pathname.split('site-packages/')[-1]
        elif 'externals/' in record.pathname:
            record.shortpath = record.pathname.split('externals/')[-1]
        else:
            record.shortpath = record.pathname
        return True