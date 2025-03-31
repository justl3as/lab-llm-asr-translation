import logging
import sys


def setup_logger(name=None, level=logging.INFO):
    """
    Set up and configure a logger

    Args:
        name: Logger name (defaults to root logger if None)
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if handlers aren't already set up
    if not logger.handlers:
        logger.setLevel(level)

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Create formatter with date and time
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(console_handler)

        # Prevent propagation to root logger if this is a named logger
        if name:
            logger.propagate = False

    return logger
