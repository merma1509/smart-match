"""Health check and root endpoint."""
import os
from pathlib import Path

from fastapi import APIRouter

router = APIRouter()

UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")


@router.get("/")
def root():
    return {
        "service": "Smart Match API",
        "version": "1.1.0",
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
        issues.append("uploads directory missing")
    if not RESULTS_DIR.exists():
        issues.append("results directory missing")

    for d in [UPLOAD_DIR, RESULTS_DIR]:
        if d.exists() and not os.access(d, os.W_OK):
            issues.append(f"{d.name} directory not writable")

    status = "healthy"
    if issues:
        status = "degraded"

    return {
        "status": status,
        "service": "smart-match",
        "version": "1.1.0",
        "issues": issues if issues else None,
    }
