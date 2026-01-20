"""
Logging module for Medical AI Diagnosis System v2.0

Provides structured logging for training, evaluation, and inference.
"""
import logging
import sys
from pathlib import Path
from typing import Optional

from utils.config import LOG_LEVEL, LOG_FORMAT, LOG_FILE


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Setup structured logger for the application.

    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, uses config default)

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level or LOG_LEVEL)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level or LOG_LEVEL)
    console_formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file or LOG_FILE:
        file_path = log_file or LOG_FILE
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level or LOG_LEVEL)
        file_formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


# Global loggers for different modules
data_logger = setup_logger("medical_ai.data")
model_logger = setup_logger("medical_ai.models")
api_logger = setup_logger("medical_ai.api")
