# Defines shared data types used across all schemas
from pydantic import BaseModel
from typing import Optional


class FieldPrediction(BaseModel):
    """A single extracted field with confidence score."""
    value: str           # the extracted text (Russian)
    confidence: float    # confidence score (0.0 to 1.0)


class NameResolution(BaseModel):
    """Normalized name with variants."""
    original: str
    canonical: str
    variants: list[str] = []
    first_name: str = ""
    last_name: str = ""
    confidence: float = 0.0


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process."""
    average_confidence: float = 0.0
    source_length: int = 0
    method: Optional[str] = None
    language: str = "ru"