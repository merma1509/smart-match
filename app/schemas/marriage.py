# Defines the marriage record output structure

from pydantic import BaseModel

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

    # Metadata
    _extraction: ExtractionMetadata | None = None
