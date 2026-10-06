"""
Logging utilities.

Provides rich formatted logging with optional file output.
"""

import sys
import logging
from typing import Optional

from rich.logging import RichHandler

from ..config import DEFAULTS


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure logging with rich formatting and optional file output.
    
    Also redirects stdout to the log file so all print() calls are captured.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional path to log file
    
    Returns:
        Configured logger instance
    """
    handlers = [RichHandler(rich_tracebacks=True, markup=True)]
    
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        handlers.append(file_handler)
        
        # Also redirect stdout to capture print() calls
        sys.stdout = Logger(log_file)
    
    logging.basicConfig(
        level=log_level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers
    )
    
    return logging.getLogger("gtp")


def log(
    message: str,
    level: str = "info",
    verbose_key: str = "verbose",
    rank: Optional[int] = None
) -> None:
    """
    Conditional logging based on verbosity settings.
    
    Args:
        message: String to log
        level: Log level ('debug', 'info', 'warning', 'error')
        verbose_key: Key in DEFAULTS dict
        rank: DDP rank (None = log always, 0 = only main process)
    """
    if rank is not None and rank != 0:
        return
    
    if not (DEFAULTS.get(verbose_key, False) or DEFAULTS.get("verbose", False)):
        return
    
    logger = logging.getLogger("gtp")
    getattr(logger, level.lower())(message)


class Logger:
    """Redirect stdout to both console and log file."""
    
    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=1)
    
    def write(self, message: str) -> None:
        """Write to both terminal and log file."""
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self) -> None:
        """Flush both outputs."""
        self.terminal.flush()
        self.log.flush()
    
    def close(self) -> None:
        """Close log file."""
        self.log.close()
