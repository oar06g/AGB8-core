import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from .settings import LOG_FILE, LOG_LEVEL

log_path = Path(LOG_FILE).parent
log_path.mkdir(parents=True, exist_ok=True)


logger = logging.getLogger("agb8_core")
logger.setLevel(LOG_LEVEL)


console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)
console_formatter = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")
console_handler.setFormatter(console_formatter)


file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5)
file_handler.setLevel(LOG_LEVEL)
file_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
file_handler.setFormatter(file_formatter)


logger.addHandler(console_handler)
logger.addHandler(file_handler)

def log_info(message: str):
    logger.info(message)

def log_error(message: str):
    logger.error(message)

def log_debug(message: str):
    logger.debug(message)
