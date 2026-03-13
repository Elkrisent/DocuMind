import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from models import Base
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://documind:documind_password@postgres:5432/documind"
)

async def init_db():
    """Create all database tables"""
    engine = create_async_engine(DATABASE_URL, echo=True)
    
    async with engine.begin() as conn:
        # Drop all tables (WARNING: deletes data!)
        await conn.run_sync(Base.metadata.drop_all)
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
    
    await engine.dispose()
    print("✅ Database initialized successfully!")

if __name__ == "__main__":
    asyncio.run(init_db())
