# Entry point of the application. It creates and configures the FastAPI server
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes.health import router as health_router
from app.api.routes.extract import router as extract_router

# FastAPI server instance
app = FastAPI(title="Smart Match", version="0.1.0")

# Register the routers
app.include_router(health_router)
app.include_router(extract_router)

@app.get("/")
def root():
   return {
      "service": "Smart Match",
      "status": "running"
      }

# Global exception handler that catches all unhandled errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )



