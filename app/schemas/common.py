# Defines shared data types used across all schemas
from pydantic import BaseModel, Field


class FieldPrediction(BaseModel):
    """A single extracted field with confidence score."""
    value: str
    confidence: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"value": "Иван Петров", "confidence": 0.95},
                {"value": "15 января 1890", "confidence": 0.88},
                {"value": "Unknown", "confidence": 0.0},
            ]
        }
    }


class NameResolution(BaseModel):
    """Normalized name with variants."""
    original: str
    canonical: str
    variants: list[str] = []
    first_name: str = ""
    last_name: str = ""
    confidence: float = 0.0

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "original": "Иванъ Петровъ",
                    "canonical": "Иван Петров",
                    "variants": ["Ivan Petrov", "Ioann Petrov"],
                    "first_name": "Иван",
                    "last_name": "Петров",
                    "confidence": 0.85,
                }
            ]
        }
    }


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process."""
    average_confidence: float = 0.0
    source_length: int = 0
    method: str | None = None
    language: str = "ru"

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "average_confidence": 0.67,
                    "source_length": 245,
                    "method": "heuristic+llm",
                    "language": "ru",
                }
            ]
        }
    }
