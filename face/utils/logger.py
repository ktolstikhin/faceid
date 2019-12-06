import sys
import logging


class LevelFilter(logging.Filter):

    def filter(self, record):
        return record.levelno in (logging.DEBUG, logging.INFO, logging.WARNING)


def init_logger(name=None, verbose=False):
    logger = logging.getLogger(name)

    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    stdout_h = logging.StreamHandler(sys.stdout)
    stdout_h.setLevel(level)
    stdout_h.addFilter(LevelFilter())
    stdout_h.flush = sys.stdout.flush

    stderr_h = logging.StreamHandler(sys.stderr)
    stderr_h.setLevel(logging.ERROR)
    stderr_h.flush = sys.stderr.flush

    formatter = logging.Formatter(
        '%(asctime)s %(threadName)s %(process)d %(levelname)s > %(message)s')

    for h in (stdout_h, stderr_h):
        h.setFormatter(formatter)
        logger.addHandler(h)

    return logger

