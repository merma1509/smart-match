# Define the health check endpoint 
from fastapi import FastAPI
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "smart-match"
    }