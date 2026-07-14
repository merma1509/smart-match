# Defines the birth record output structure
from pydantic import BaseModel
from typing import Optional
from app.schemas.common import FieldPrediction, NameResolution, ExtractionMetadata


class BirthRecord(BaseModel):
    record_type: str = "birth"
    child_name: FieldPrediction
    birth_date: FieldPrediction
    baptism_date: FieldPrediction
    father_name: FieldPrediction
    mother_name: FieldPrediction
    needs_review: bool = False
    
    # Optional resolved entities
    child_name_resolved: Optional[NameResolution] = None
    father_name_resolved: Optional[NameResolution] = None
    mother_name_resolved: Optional[NameResolution] = None
    
    # Optional computed fields
    age_computed: Optional[dict] = None
    
    # Metadata
    _extraction: Optional[ExtractionMetadata] = None