"""
Health check endpoints for load balancer probes and monitoring.
"""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"


@router.get("/health", response_model=HealthResponse)
async def health():
    """Liveness probe – returns 200 if the service is running."""
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=HealthResponse)
async def readiness():
    """
    Readiness probe – checks downstream dependencies.
    Extend this to ping Pinecone, Redis, and Postgres in production.
    """
    return HealthResponse(status="ready")
