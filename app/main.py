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

app = FastAPI(
    title="DevBattles AI Backend",
    description="LangGraph, pgvector, and Playwright Vision grading service engine",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include core routes
app.include_router(router)
