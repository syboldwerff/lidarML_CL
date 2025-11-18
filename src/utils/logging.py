"""Simple logging utility.

Provides a lightweight wrapper around Python's standard logging
module to produce consistent log messages across the pipeline.  You
can configure log level and format here or replace this with your
preferred logging framework.
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with a preset format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger