"""
Emotional Risk ML Microservice
FastAPI application exposing model inference endpoints.
"""
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers import predict, health
from app.core.model_registry import ModelRegistry
from app.core.config import settings
from app.core.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


# ── Lifespan: load models once at startup ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML models...")
    registry = ModelRegistry()
    registry.load_all()
    app.state.model_registry = registry
    logger.info("All models loaded. Service ready.")
    yield
    logger.info("Shutting down ML service.")


# ── Application factory ────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title="Emotional Risk ML Service",
        version="1.0.0",
        description="Privacy-first behavioral risk inference microservice",
        docs_url="/docs" if settings.ENV != "production" else None,
        redoc_url=None,
        lifespan=lifespan,
    )

    # Internal-only CORS — only Node API gateway should call this service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_methods=["POST", "GET"],
        allow_headers=["Content-Type", "X-Internal-Secret", "X-Request-ID"],
    )

    # ── Internal secret validation middleware ──────────────────────────────────
    @app.middleware("http")
    async def verify_internal_secret(request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)

        secret = request.headers.get("X-Internal-Secret", "")
        if settings.INTERNAL_SECRET and secret != settings.INTERNAL_SECRET:
            return JSONResponse(status_code=403, content={"detail": "Forbidden"})

        return await call_next(request)

    # ── Request ID propagation ─────────────────────────────────────────────────
    @app.middleware("http")
    async def propagate_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", "")
        response = await call_next(request)
        if request_id:
            response.headers["X-Request-ID"] = request_id
        return response

    app.include_router(predict.router)
    app.include_router(health.router)

    return app


app = create_app()