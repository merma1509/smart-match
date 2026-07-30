# Centralizes all configuration settings in one place
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    # App Name, Version, API host/port
    app_name: str = "Smart Match"
    app_version: str = "1.1.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    environment: str = "development"  # "development" | "production"

    # Input and output directories — centralized
    upload_dir: str = "uploads"  # Загружаемые файлы
    input_dir: str = "uploads"  # Synced with upload_dir for backward compat
    output_dir: str = "results"  # Результаты экстракции
    joined_data_dir: str = "joined_data"  # Тестовые данные (синхронизировано)

    # CORS — ограниченный в production
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # OCR model (Russian-optimized)
    ocr_model_name: str = "taiga75/ru-trocr-1700s"
    ocr_handwritten_model: str = "taiga75/ru-trocr-1700s"
    ocr_printed_model: str = "taiga75/ru-trocr-1700s"
    ocr_confidence_threshold: float = 0.7
    tesseract_lang: str = "rus"
    ocr_use_gpu: bool = False  # GPU/CUDA support for EasyOCR

    # Layout
    layout_use_yolo: bool = False  # Использовать YOLOv8 для layout detection
    layout_yolo_model_path: str = "app/models/layout/yolov8n.pt"
    layout_yolo_confidence: float = 0.25

    # LLM
    llm_provider: str = "ollama"  # "ollama" | "openai" | "huggingface"
    llm_model_name: str = "llama3.1:8b"
    llm_api_base_url: str = "http://localhost:11434"  # Для Ollama / OpenAI-compatible
    llm_api_key: str = ""  # Для OpenAI
    llm_confidence_threshold: float = 0.7

    # File upload limits and allowed extensions
    max_file_size_mb: int = 50
    allowed_extensions: list[str] = [".jpg", ".jpeg", ".png"]

    # Preprocessing defaults
    max_image_dimension: int = 3000
    default_language: str = "ru"
    border_removal_width: int = 10  # Настраиваемая ширина удаления рамок

    # Log settings
    log_level: str = "INFO"
    log_rotation: str = "10 MB"
    log_retention: str = "7 days"

    def model_post_init(self, __context):
        """Синхронизация путей после инициализации."""
        # input_dir = upload_dir для консистентности
        if self.input_dir != self.upload_dir:
            self.input_dir = self.upload_dir

    @property
    def effective_cors_origins(self) -> list[str]:
        """Возвращает CORS origins в зависимости от окружения."""
        if self.environment == "production" and self.cors_origins == ["*"]:
            # В production нельзя использовать "*" — возвращаем пустой список
            # (пользователь должен явно указать разрешённые origins)
            return []
        return self.cors_origins


settings = Settings()


def validate_config():
    """Validate configuration on startup."""
    errors = []

    # Create directories if they don't exist
    for dir_path in [settings.upload_dir, settings.output_dir, settings.joined_data_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Валидация CORS в production
    if settings.environment == "production" and settings.cors_origins == ["*"]:
        errors.append(
            "Production environment requires explicit CORS origins. "
            "Set `cors_origins` in config instead of ['*']"
        )

    # Check allowed extensions
    if not settings.allowed_extensions:
        errors.append("No allowed file extensions configured")

    # Validate port range
    if not (0 < settings.api_port < 65536):
        errors.append(f"Invalid port: {settings.api_port}")

    # Validate LLM config
    if settings.llm_provider == "openai" and not settings.llm_api_key:
        errors.append("LLM provider is 'openai' but no API key provided")

    # Validate YOLO model path
    if settings.layout_use_yolo:
        model_path = Path(settings.layout_yolo_model_path)
        if not model_path.exists():
            errors.append(f"YOLO model not found at {settings.layout_yolo_model_path}")

    if errors:
        raise RuntimeError(f"Configuration errors: {'; '.join(errors)}")
    return True
