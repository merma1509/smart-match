# Centralizes all configuration settings in one place
import os
from pathlib import Path
from pydantic import BaseModel


class Settings(BaseModel):
    # App Name, Version, API host/port
    app_name: str = "Smart Match"
    app_version: str = "1.1.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Input and output directories
    input_dir: str = "data/input"
    output_dir: str = "data/output"

    # OCR model (Russian-optimized)
    ocr_model_name: str = "taiga75/ru-trocr-1700s"
    ocr_handwritten_model: str = "taiga75/ru-trocr-1700s"
    ocr_printed_model: str = "taiga75/ru-trocr-1700s"
    ocr_confidence_threshold: float = 0.7
    tesseract_lang: str = "rus"

    # File upload limits and allowed extensions
    max_file_size_mb: int = 50
    allowed_extensions: list[str] = [".jpg", ".jpeg", ".png"]

    # Preprocessing defaults
    max_image_dimension: int = 3000
    default_language: str = "ru"

    # Log settings
    log_level: str = "INFO"
    log_rotation: str = "10 MB"
    log_retention: str = "7 days"


settings = Settings()


def validate_config():
    """Validate configuration on startup."""
    errors = []

    # Create directories if they don't exist
    for dir_path in [settings.input_dir, settings.output_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Check allowed extensions
    if not settings.allowed_extensions:
        errors.append("No allowed file extensions configured")

    # Validate port range
    if not (0 < settings.api_port < 65536):
        errors.append(f"Invalid port: {settings.api_port}")

    if errors:
        raise RuntimeError(f"Configuration errors: {'; '.join(errors)}")
    return True
