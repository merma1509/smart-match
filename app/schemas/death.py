# Defines the death record output structure

from pydantic import BaseModel

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

    # Metadata
    _extraction: ExtractionMetadata | None = None
