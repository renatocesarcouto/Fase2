"""
FastAPI main application for Medical AI Diagnosis System v2.0

Entry point for the REST API server.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.endpoints import router, initialize_predictor
from utils.logger import api_logger
from utils.config import (
    PROJECT_NAME,
    MODEL_VERSION,
    CORS_ORIGINS,
    DEFAULT_MODEL,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup: Load model
    api_logger.info("Starting Medical AI Diagnosis API")
    try:
        initialize_predictor(model_type=DEFAULT_MODEL)
        api_logger.info("✅ API startup complete")
    except Exception as e:
        api_logger.error(f"❌ API startup failed: {e}")
        raise

    yield

    # Shutdown
    api_logger.info("Shutting down Medical AI Diagnosis API")


# Create FastAPI app
app = FastAPI(
    title=PROJECT_NAME,
    description=(
        "REST API for breast cancer diagnosis prediction using Machine Learning. "
        "**This is a medical support tool, not a replacement for professional diagnosis.**"
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, tags=["Prediction"])


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors.
    """
    api_logger.error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "detail": str(exc),
    }


if __name__ == "__main__":
    import uvicorn
    from utils.config import API_HOST, API_PORT, API_RELOAD

    api_logger.info(f"Starting server on {API_HOST}:{API_PORT}")
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
        log_level="info",
    )
