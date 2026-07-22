# Defines the birth record output structure

from pydantic import BaseModel

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

    # Metadata
    _extraction: ExtractionMetadata | None = None
