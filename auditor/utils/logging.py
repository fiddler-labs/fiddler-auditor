import logging
from typing import Optional
import sys


FORMATTER = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
DEFAULT_LEVEL = logging.INFO


def get_module_name() -> str:
    return __name__.split('.')[0]


def get_logger(
    name: Optional[str]
) -> logging.Logger:
    """
    Creates and returns a loggin object

    Args:
        name (Optional[str]): name of the logger

    Returns:
        logging.Logger: logging object
    """
    if name is None:
        name = get_module_name()
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        _configure_logger(logger)
    return logger


def get_module_logger() -> logging.Logger:
    return get_logger(get_module_name())


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def _configure_logger(
    logger: logging.Logger,
) -> None:
    console_handler = get_console_handler()
    logger.addHandler(console_handler)
    logger.setLevel(DEFAULT_LEVEL)
    logger.propagate = False
    return
