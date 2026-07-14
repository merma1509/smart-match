# Defines the death record output structure
from pydantic import BaseModel
from typing import Optional
from app.schemas.common import FieldPrediction, NameResolution, ExtractionMetadata


class DeathRecord(BaseModel):
    record_type: str = "death"
    deceased_name: FieldPrediction
    death_date: FieldPrediction
    burial_date: FieldPrediction
    age: Optional[FieldPrediction] = None
    needs_review: bool = False
    
    # Optional resolved entities
    deceased_name_resolved: Optional[NameResolution] = None
    
    # Optional computed fields
    age_computed: Optional[dict] = None
    age_validation: Optional[dict] = None
    
    # Metadata
    _extraction: Optional[ExtractionMetadata] = None