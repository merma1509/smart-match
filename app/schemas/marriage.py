# Defines the marriage record output structure
from pydantic import BaseModel, Field

from app.schemas.common import ExtractionMetadata, FieldPrediction, NameResolution


class MarriageRecord(BaseModel):
    record_type: str = "marriage"
    groom_name: FieldPrediction
    bride_name: FieldPrediction
    marriage_date: FieldPrediction
    needs_review: bool = False

    # Optional resolved entities
    groom_name_resolved: NameResolution | None = None
    bride_name_resolved: NameResolution | None = None

    # Extraction metadata
    extraction_meta: ExtractionMetadata | None = Field(None, alias="_extraction")

    model_config = {
        "json_schema_extra": {
            "example": {
                "record_type": "marriage",
                "groom_name": {"value": "Сергей Николаев", "confidence": 0.8},
                "bride_name": {"value": "Елена Дмитриева", "confidence": 0.75},
                "marriage_date": {"value": "12 июня 1891", "confidence": 0.92},
                "needs_review": False,
                "extraction_meta": {
                    "average_confidence": 0.78,
                    "source_length": 120,
                    "method": "heuristic+llm",
                    "language": "ru",
                },
            }
        }
    }
