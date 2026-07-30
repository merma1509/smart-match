"""Set up structured logging for the entire application."""

import os

from loguru import logger

from app.core.config import settings

os.makedirs("logs", exist_ok=True)

# Remove default handler
logger.remove()

# Console format - split for readability
console_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# Add console handler
logger.add(
    sink=lambda msg: print(msg, end=""),
    level=settings.log_level,
    format=console_format,
    colorize=True,
)

# Add file handler with rotation
logger.add(
    "logs/smart_match.log",
    rotation=settings.log_rotation,
    retention=settings.log_retention,
    level=settings.log_level,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
)
