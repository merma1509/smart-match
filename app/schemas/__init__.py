"""Schemas for API responses and data models."""

from app.schemas.common import ExtractionMetadata, FieldPrediction, NameResolution
from app.schemas.responses import (
    BatchExtractionResponse,
    BeforeAfterMetrics,
    BirthExtractionResult,
    DeathExtractionResult,
    DeleteResultResponse,
    ExtractionResult,
    HealthResponse,
    MarriageExtractionResult,
    QualityMetrics,
    ResultItem,
    ResultsListResponse,
    RootResponse,
    SingleExtractionResponse,
)
from app.schemas.responses import (
    FieldPrediction as ResponseFieldPrediction,
)
from app.schemas.responses import (
    NameResolution as ResponseNameResolution,
)

__all__ = [
    # Common
    "FieldPrediction",
    "NameResolution",
    "ExtractionMetadata",
    # Extraction results
    "ExtractionResult",
    "BirthExtractionResult",
    "DeathExtractionResult",
    "MarriageExtractionResult",
    # API responses
    "SingleExtractionResponse",
    "BatchExtractionResponse",
    "HealthResponse",
    "RootResponse",
    "ResultsListResponse",
    "ResultItem",
    "DeleteResultResponse",
    # Metrics
    "QualityMetrics",
    "BeforeAfterMetrics",
    # Response-specific types
    "ResponseFieldPrediction",
    "ResponseNameResolution",
]
