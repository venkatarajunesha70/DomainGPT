"""
Initialize the PostgreSQL schema (create all tables).
Run once before starting the API:
  python scripts/init_db.py
"""
import asyncio
from apps.api.core.database import Base, engine

# Import all models so they register with Base.metadata
from apps.api.models.user import User           # noqa: F401
from apps.api.models.document import Document, DocumentChunk  # noqa: F401
from apps.api.models.conversation import Conversation, Message  # noqa: F401


async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database tables created successfully.")


if __name__ == "__main__":
    asyncio.run(create_tables())
