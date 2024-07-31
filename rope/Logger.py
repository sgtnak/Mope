import logging

logger = logging.getLogger('mope')
logger.setLevel(logging.DEBUG)  # Set the logging level
_console_handler = logging.StreamHandler()
_file_handler = logging.FileHandler('log.txt')
_console_handler.setLevel(logging.DEBUG)
_file_handler.setLevel(logging.DEBUG)

_console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

_console_handler.setFormatter(_console_formatter)
_file_handler.setFormatter(_file_formatter)

logger.addHandler(_console_handler)
logger.addHandler(_file_handler)

def get_logger():
    return logger