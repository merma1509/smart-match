"""Health check and root endpoint."""
import os
from pathlib import Path

from fastapi import APIRouter

from app.core.config import settings

router = APIRouter()

UPLOAD_DIR = Path(settings.input_dir)
RESULTS_DIR = Path(settings.output_dir)


@router.get("/")
def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "endpoints": {
            "GET /": "This info",
            "GET /health": "Health check",
            "POST /extract": "Extract genealogical data from image",
            "POST /extract/batch": "Extract from multiple images",
            "GET /results": "List all results",
            "GET /results/{id}": "Get result by ID",
            "DELETE /results/{id}": "Delete a result",
        },
    }


@router.get("/health")
def health():
    """Deep health check — verifies service and directories."""
    issues = []

    if not UPLOAD_DIR.exists():
        issues.append("input directory missing")
    if not RESULTS_DIR.exists():
        issues.append("output directory missing")

    for d in [UPLOAD_DIR, RESULTS_DIR]:
        if d.exists() and not os.access(d, os.W_OK):
            issues.append(f"{d.name} directory not writable")

    status = "healthy"
    if issues:
        status = "degraded"

    return {
        "status": status,
        "service": settings.app_name,
        "version": settings.app_version,
        "issues": issues if issues else None,
    }
