import os
import logging
import sys
from typing import Optional


def get_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None):
    """
    Get a logger with a specific name and level.
    
    Args:
        name: Name of the logger
        level: Logging level (default: INFO)
        log_file: Optional file path to log to
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_logger(log_dir: str, name: str = "XLSR-Transducer"):
    """
    Set up a logger with both console and file handlers.
    
    Args:
        log_dir: Directory to write log files
        name: Name of the logger (default: "XLSR-Transducer")
        
    Returns:
        Logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file path
    log_file = os.path.join(log_dir, f"{name}.log")
    
    # Get logger
    logger = get_logger(name, log_file=log_file)
    
    return logger 