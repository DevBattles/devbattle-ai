import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api.routes import router
from app.graph.nodes import vector_client
from app.utils.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up DevBattles AI backend server lifespan...")
    try:
        # Initialize extension and tables
        await vector_client.initialize_db()
        logger.info("Database pgvector schemas initialized successfully.")
    except Exception as e:
        logger.error(f"Critical error on database migration initialization: {e}")
    yield
    logger.info("Shutting down DevBattles AI backend lifespan...")

from app.config.config import settings

if not settings.internal_api_key:
    logger.warning(
        "INTERNAL_API_KEY is not configured. All /internal/* endpoints are UNAUTHENTICATED. "
        "Set INTERNAL_API_KEY in the environment before exposing this service publicly."
    )

app = FastAPI(
    title="DevBattles AI Backend",
    description="LangGraph, pgvector, and Playwright Vision grading service engine",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS Middleware. Credentials cannot be combined with a wildcard origin per the
# CORS spec, so only enable allow_credentials when explicit origins are configured.
_cors_origins = settings.cors_origins_list
_allow_credentials = "*" not in _cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include core routes
app.include_router(router)
