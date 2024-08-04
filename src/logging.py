import logging
from src.variables import LOG_LEVEL


def get_logger(class_name: str, log_level: str = LOG_LEVEL):
    logging.basicConfig()
    logger = logging.getLogger(class_name)
    logger.setLevel(log_level)
    return logger
