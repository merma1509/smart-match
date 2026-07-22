# Defines the death record output structure
from pydantic import BaseModel, Field

from app.schemas.common import ExtractionMetadata, FieldPrediction, NameResolution


class DeathRecord(BaseModel):
    record_type: str = "death"
    deceased_name: FieldPrediction
    death_date: FieldPrediction
    burial_date: FieldPrediction
    age: FieldPrediction | None = None
    needs_review: bool = False

    # Optional resolved entities
    deceased_name_resolved: NameResolution | None = None

    # Optional computed fields
    age_computed: dict | None = None
    age_validation: dict | None = None

    # Extraction metadata
    extraction_meta: ExtractionMetadata | None = Field(None, alias="_extraction")

    model_config = {
        "json_schema_extra": {
            "example": {
                "record_type": "death",
                "deceased_name": {"value": "Анна Петрова", "confidence": 0.73},
                "death_date": {"value": "3 марта 1892", "confidence": 0.91},
                "burial_date": {"value": "5 марта 1892", "confidence": 0.85},
                "age": {"value": "45 лет", "confidence": 0.6},
                "needs_review": False,
                "extraction_meta": {
                    "average_confidence": 0.72,
                    "source_length": 180,
                    "method": "heuristic+llm",
                    "language": "ru",
                },
            }
        }
    }
