import logging
import boto3
from datetime import datetime


def setup_logging(name):
    """
    Sets up a logger with a specified name, configuring it to log messages to a file.

    This function creates a logger with the given name, sets up a file handler to write log messages
    to 'app.log', and formats the log messages with the date, logger name, log level, and message.
    The logger's log level is set to INFO.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    handler = logging.FileHandler('app.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
