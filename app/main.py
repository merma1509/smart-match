"""FastAPI application entry point."""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.extract import router as extract_router
from app.api.routes.health import router as health_router
from app.api.routes.results import router as results_router
from app.core.config import settings, validate_config
from app.core.logging import logger

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Genealogical data extraction from metrical books",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health_router)
app.include_router(extract_router)
app.include_router(results_router)


@app.on_event("startup")
async def startup():
    logger.info(f"Starting {settings.app_name} API...")
    try:
        validate_config()
        logger.info(f"Configuration validated: {settings.app_version}")
    except RuntimeError as e:
        logger.error(f"Configuration error: {e}")
        raise

    # Preload heavy models
    from app.services.ocr import preload_models
    preload_models()

    logger.info(f"{settings.app_name} API ready")


@app.on_event("shutdown")
async def shutdown():
    logger.info(f"{settings.app_name} API shutting down")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
