# Defines the marriage record output structure
from pydantic import BaseModel
from typing import Optional
from app.schemas.common import FieldPrediction, NameResolution, ExtractionMetadata


class MarriageRecord(BaseModel):
    record_type: str = "marriage"
    groom_name: FieldPrediction
    bride_name: FieldPrediction
    marriage_date: FieldPrediction
    needs_review: bool = False
    
    # Optional resolved entities
    groom_name_resolved: Optional[NameResolution] = None
    bride_name_resolved: Optional[NameResolution] = None
    
    # Metadata
    _extraction: Optional[ExtractionMetadata] = None