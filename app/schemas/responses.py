"""Pydantic schemas for API responses."""

from pydantic import BaseModel, Field


class QualityMetrics(BaseModel):
    """Image quality metrics."""

    contrast: float = 0.0
    sharpness: float = 0.0
    entropy: float = 0.0


class BeforeAfterMetrics(BaseModel):
    """Before and after quality metrics."""

    before: QualityMetrics
    after: QualityMetrics
    steps: list[str] = []


class FieldPrediction(BaseModel):
    """A single extracted field with confidence score."""

    value: str = "Unknown"
    confidence: float = 0.0


class NameResolution(BaseModel):
    """Normalized name with variants."""

    original: str = ""
    canonical: str = ""
    variants: list[str] = []
    first_name: str = ""
    last_name: str = ""
    confidence: float = 0.0


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process."""

    average_confidence: float = 0.0
    source_length: int = 0
    method: str | None = None
    language: str = "ru"
    record_type_confidence: float | None = None
    found_fields: list[str] = []
    llm_used: bool = False


class ExtractionResult(BaseModel):
    """Base extraction result with common fields."""

    record_type: str = "unknown"
    needs_review: bool = True
    extraction_meta: ExtractionMetadata | None = Field(None, alias="_extraction")

    model_config = {"populate_by_name": True}


class BirthExtractionResult(ExtractionResult):
    """Birth record extraction result."""

    record_type: str = "birth"
    child_name: FieldPrediction = Field(default_factory=lambda: FieldPrediction())
    birth_date: FieldPrediction = Field(default_factory=lambda: FieldPrediction())
    baptism_date: FieldPrediction = Field(default_factory=lambda: FieldPrediction())
    father_name: FieldPrediction = Field(default_factory=lambda: FieldPrediction())
    mother_name: FieldPrediction = Field(default_factory=lambda: FieldPrediction())
    child_name_resolved: NameResolution | None = None
    father_name_resolved: NameResolution | None = None
    mother_name_resolved: NameResolution | None = None
    age_computed: dict | None = None


class DeathExtractionResult(ExtractionResult):
    """Death record extraction result."""

    record_type: str = "death"
    deceased_name: FieldPrediction = Field(default_factory=lambda: FieldPrediction())
    death_date: FieldPrediction = Field(default_factory=lambda: FieldPrediction())
    burial_date: FieldPrediction = Field(default_factory=lambda: FieldPrediction())
    age: FieldPrediction | None = None
    deceased_name_resolved: NameResolution | None = None
    age_computed: dict | None = None
    age_validation: dict | None = None


class MarriageExtractionResult(ExtractionResult):
    """Marriage record extraction result."""

    record_type: str = "marriage"
    groom_name: FieldPrediction = Field(default_factory=lambda: FieldPrediction())
    bride_name: FieldPrediction = Field(default_factory=lambda: FieldPrediction())
    marriage_date: FieldPrediction = Field(default_factory=lambda: FieldPrediction())
    groom_name_resolved: NameResolution | None = None
    bride_name_resolved: NameResolution | None = None


class SingleExtractionResponse(BaseModel):
    """Response for single image extraction."""

    request_id: str
    file: str
    processing_time_seconds: float
    image_size: str
    quality_metrics: BeforeAfterMetrics
    page_type: str
    num_elements: int
    extracted_data: dict
    resolved_data: dict
    raw_text_preview: str
    needs_review: bool


class BatchExtractionResponse(BaseModel):
    """Response for batch extraction."""

    total: int
    success: int
    failed: int
    rollback: bool
    results: list[dict]
    errors: list[dict]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str
    issues: list[str] | None = None


class RootResponse(BaseModel):
    """Root endpoint response."""

    service: str
    version: str
    status: str
    endpoints: dict


class ResultItem(BaseModel):
    """Single result item in the list."""

    id: str
    file: str
    type: str
    needs_review: bool
    time: float


class ResultsListResponse(BaseModel):
    """Response for listing results."""

    total: int
    limit: int
    offset: int
    results: list[ResultItem]


class DeleteResultResponse(BaseModel):
    """Response for deleting a result."""

    status: str
    id: str
