# Defines the birth record output structure
from pydantic import BaseModel, Field

from app.schemas.common import ExtractionMetadata, FieldPrediction, NameResolution


class BirthRecord(BaseModel):
    record_type: str = "birth"
    child_name: FieldPrediction
    birth_date: FieldPrediction
    baptism_date: FieldPrediction
    father_name: FieldPrediction
    mother_name: FieldPrediction
    needs_review: bool = False

    # Optional resolved entities
    child_name_resolved: NameResolution | None = None
    father_name_resolved: NameResolution | None = None
    mother_name_resolved: NameResolution | None = None

    # Optional computed fields
    age_computed: dict | None = None

    # Extraction metadata
    extraction_meta: ExtractionMetadata | None = Field(None, alias="_extraction")

    model_config = {
        "json_schema_extra": {
            "example": {
                "record_type": "birth",
                "child_name": {"value": "Иван Петров", "confidence": 0.7},
                "birth_date": {"value": "15 января 1890", "confidence": 0.88},
                "baptism_date": {"value": "16 января 1890", "confidence": 0.75},
                "father_name": {"value": "Пётр Иванов", "confidence": 0.82},
                "mother_name": {"value": "Мария Иванова", "confidence": 0.65},
                "needs_review": True,
                "child_name_resolved": {
                    "original": "Иванъ Петровъ",
                    "canonical": "Иван Петров",
                    "variants": [],
                    "first_name": "Иван",
                    "last_name": "Петров",
                    "confidence": 0.85,
                },
                "extraction_meta": {
                    "average_confidence": 0.67,
                    "source_length": 245,
                    "method": "heuristic+llm",
                    "language": "ru",
                },
            }
        }
    }
