import logging
import os
from logging.handlers import RotatingFileHandler


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    logs_dir = os.getenv("LOGS_DIR", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    file_handler = RotatingFileHandler(
        os.path.join(logs_dir, f"{name}.log"), maxBytes=5_000_000, backupCount=3
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger