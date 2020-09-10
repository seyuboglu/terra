import sys
from datetime import datetime


def init_logging(path="process.log"):
    logger = Logger(path=path)
    sys.stdout = logger
    return logger


class Logger(object):
    """Logger that writes all of stdout and stderr to the file passed in as `path`.
    Known limitations:
    - Doesn't log stderr in Jupyter Notebooks
    - Writes stderr to stdout
    """

    def __init__(self, path="process.log"):
        self.stdout = sys.stdout
        self.log = open(path, "w")

    def write(self, message):
        self.stdout.write(message)

        if message.isspace():
            self.log.write(message)
        else:
            self.log.write(f"[{str(datetime.now())}]: " + message)
        self.log.flush()

    def __getattr__(self, attr):
        return getattr(self.stdout, attr)