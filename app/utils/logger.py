import logging
import sys

def setup_logger():
    """
    Configure a centralized console logger for the FastAPI application
    """
    logger = logging.getLogger("devbattle-ai")
    
    # Avoid duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    
    logger.addHandler(stdout_handler)
    logger.propagate = False
    return logger

logger = setup_logger()
