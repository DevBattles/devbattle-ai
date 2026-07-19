from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text
from app.config.config import settings
from app.utils.logger import logger
from app.embeddings.models import Base, AISolutionEmbedding
from typing import List, Dict, Any
import uuid

class VectorClient:
    def __init__(self):
        logger.info("Initializing pgvector database client...")
        self.engine = create_async_engine(
            settings.async_database_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            pool_recycle=1800,
            connect_args={
                "prepared_statement_cache_size": 0,
                "command_timeout": 30
            }
        )
        self.async_session = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

    async def initialize_db(self):
        """
        Register vector extension and construct schemas
        """
        try:
            async with self.engine.begin() as conn:
                logger.info("Configuring pgvector extension and table models in database...")
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                await conn.run_sync(Base.metadata.create_all)
        except Exception as e:
            logger.error(f"Error during database initialization: {e}")
            raise e

    async def save_solutions(self, question_id: str, version: int, solutions: List[Dict[str, Any]]):
        """
        Store generated solutions and their corresponding embedding vectors
        """
        q_uuid = uuid.UUID(question_id)
        async with self.async_session() as session:
            async with session.begin():
                for sol in solutions:
                    record = AISolutionEmbedding(
                        question_id=q_uuid,
                        version=version,
                        solution_code=sol["code"],
                        solution_type=sol["type"],
                        embedding=sol["embedding"]
                    )
                    session.add(record)
        logger.info(f"Saved {len(solutions)} solution embeddings for question {question_id} v{version}")

    async def get_similar_solutions(self, question_id: str, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve semantically close reference solutions from database using cosine distance
        """
        q_uuid = uuid.UUID(question_id)
        # Cosine distance operator '<=>' works directly on pgvector column types
        stmt = text("""
            SELECT solution_code, solution_type, 1 - (embedding <=> :emb) as similarity
            FROM ai_solution_embeddings
            WHERE question_id = :qid
            ORDER BY embedding <=> :emb
            LIMIT :limit
        """)

        # pgvector expects string formatted vector format e.g. '[0.1, 0.2, ...]'
        formatted_vector = f"[{','.join(map(str, query_embedding))}]"

        async with self.async_session() as session:
            result = await session.execute(stmt, {
                "qid": q_uuid,
                "emb": formatted_vector,
                "limit": limit
            })
            rows = result.fetchall()
            return [
                {
                    "code": row[0],
                    "type": row[1],
                    "similarity": float(row[2])
                }
                for row in rows
            ]
