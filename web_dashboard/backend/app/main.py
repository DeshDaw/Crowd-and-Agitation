"""
FastAPI application for Crowd Surveillance Dashboard.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .routers import files_router, health_router, runs_router
from .services.storage import RUNS_BASE_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Crowd Surveillance Dashboard API")

    # Ensure runs directory exists
    RUNS_BASE_DIR.mkdir(parents=True, exist_ok=True)

    yield

    logger.info("Shutting down Crowd Surveillance Dashboard API")


app = FastAPI(
    title="Crowd Surveillance Dashboard API",
    description="API for Abnormal Crowd Motion Detection research prototype",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - allow frontend development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:4173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, prefix="/api")
app.include_router(runs_router, prefix="/api")
app.include_router(files_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint redirects to docs."""
    return {
        "message": "Crowd Surveillance Dashboard API",
        "docs": "/docs",
        "version": "1.0.0",
    }
