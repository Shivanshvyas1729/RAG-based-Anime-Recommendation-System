import logging
import sys
from datetime import datetime
from config.config import LOGS_DIR

LOG_FILE = LOGS_DIR / f"app_{datetime.now().strftime('%Y-%m-%d')}.log"

# Create root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger instance with the specified name."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
