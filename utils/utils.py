import logging

def get_logger(name, verbosity=2):
    logger = logging.getLogger(name)
    logger.setLevel(verbosity)
    return logger

