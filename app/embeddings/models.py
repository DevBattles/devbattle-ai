from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, Text, UUID
from pgvector.sqlalchemy import Vector
import uuid

class Base(DeclarativeBase):
    pass

class AISolutionEmbedding(Base):
    __tablename__ = "ai_solution_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    solution_code: Mapped[str] = mapped_column(Text, nullable=False)
    solution_type: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding: Mapped[list] = mapped_column(Vector(768), nullable=False)  # 768 dimensions for models/text-embedding-004
