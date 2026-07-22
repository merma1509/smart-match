"""Health check and root endpoint."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def root():
    return {
        "service": "Smart Match API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "GET /": "This info",
            "GET /health": "Health check",
            "POST /extract": "Extract genealogical data from image",
            "GET /results": "List all results",
            "GET /results/{id}": "Get result by ID",
        },
    }


@router.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "smart-match",
        "version": "1.0.0",
    }
