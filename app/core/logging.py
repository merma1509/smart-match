# Set up the structured logging for the entire application
from loguru import logger
import os

os.makedirs("logs", exist_ok=True)

logger.add(
    "logs/smart_match.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO"
)