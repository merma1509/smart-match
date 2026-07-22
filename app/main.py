"""Smart Match API — Main entry point."""

from pathlib import Path

from fastapi import FastAPI
from loguru import logger

from app.api.routes.extract import router as extract_router
from app.api.routes.health import router as health_router
from app.api.routes.results import router as results_router
from app.services.ocr import preload_models

app = FastAPI(
    title="Smart Match API",
    description="AI-powered document intelligence for historical Russian metrical books",
    version="1.0.0",
)

# Create directories
Path("uploads").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

# Register routers
app.include_router(health_router)  # GET /, GET /health
app.include_router(extract_router)  # POST /extract
app.include_router(results_router)  # GET /results, GET /results/{id}


@app.on_event("startup")
async def startup():
    logger.info("Starting Smart Match API...")
    preload_models()
    logger.info("Smart Match API ready")
