import os
import time
import gzip
from logging import Handler
from logging.handlers import TimedRotatingFileHandler

import telegram


class CompressedTimedRotatingFileHandler(TimedRotatingFileHandler):
    def gzip_logs(self, file_path):
        with open(file_path, 'rb') as f_in, gzip.open(f"{file_path}.gz", 'wb') as f_out:
            f_out.writelines(f_in)
        os.remove(file_path)
        
    def doRollover(self):
        # get the time that this sequence started at and make it a TimeTuple
        currentTime = int(time.time())
        dstNow = time.localtime(currentTime)[-1]
        t = self.rolloverAt - self.interval
        if self.utc:
            timeTuple = time.gmtime(t)
        else:
            timeTuple = time.localtime(t)
            dstThen = timeTuple[-1]
            if dstNow != dstThen:
                if dstNow:
                    addend = 3600
                else:
                    addend = -3600
                timeTuple = time.localtime(t + addend)
        dfn = self.rotation_filename(self.baseFilename + "." +
                                     time.strftime(self.suffix, timeTuple))
        super().doRollover()
        self.gzip_logs(dfn)
        
        
class TelegramHandler(Handler):
    def __init__(self, token, chat_id):
        super().__init__()
        self.token = token
        self.chat_id = chat_id

    def emit(self, record):
        bot = telegram.Bot(token=self.token)
        bot.send_message(
            self.chat_id,
            text=self.format(record)
        )