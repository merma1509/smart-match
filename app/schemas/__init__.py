"""Schemas for API responses and data models."""

from app.schemas.common import ExtractionMetadata, FieldPrediction, NameResolution
from app.schemas.responses import (
    BatchExtractionResponse,
    BirthExtractionResult,
    DeleteResultResponse,
    DeathExtractionResult,
    ExtractionResult,
    FieldPrediction as ResponseFieldPrediction,
    HealthResponse,
    MarriageExtractionResult,
    NameResolution as ResponseNameResolution,
    QualityMetrics,
    ResultItem,
    ResultsListResponse,
    RootResponse,
    SingleExtractionResponse,
    BeforeAfterMetrics,
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
