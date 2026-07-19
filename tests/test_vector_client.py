import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.embeddings.vector_client import VectorClient
from app.utils.logger import logger
from sqlalchemy import text

async def test_pgvector_connection():
    """Test pgvector connection and basic operations"""
    logger.info("Starting pgvector connection test...")
    
    vector_client = VectorClient()
    
    try:
        # Test 1: Initialize database
        logger.info("Test 1: Initializing database with pgvector extension...")
        await vector_client.initialize_db()
        logger.info("✓ Database initialized successfully")
        
        # Test 2: Check pgvector extension
        logger.info("Test 2: Checking pgvector extension...")
        async with vector_client.engine.begin() as conn:
            result = await conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))
            version = result.fetchone()
            if version:
                logger.info(f"✓ pgvector extension found, version: {version[0]}")
            else:
                logger.error("✗ pgvector extension not found")
                return False
        
        # Test 3: Test embedding save operation
        logger.info("Test 3: Testing save_solutions operation...")
        test_question_id = "00000000-0000-0000-0000-000000000001"
        test_solutions = [
            {
                "code": "def hello(): return 'Hello World'",
                "type": "python",
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 128  # 640-dimensional vector
            }
        ]
        
        await vector_client.save_solutions(test_question_id, 1, test_solutions)
        logger.info("✓ Solutions saved successfully")
        
        # Test 4: Test similarity search
        logger.info("Test 4: Testing get_similar_solutions operation...")
        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 128
        similar = await vector_client.get_similar_solutions(test_question_id, query_embedding, limit=1)
        
        if similar and len(similar) > 0:
            logger.info(f"✓ Found {len(similar)} similar solutions")
            logger.info(f"  Similarity score: {similar[0]['similarity']}")
        else:
            logger.warning("✗ No similar solutions found")
        
        # Test 5: Cleanup test data
        logger.info("Test 5: Cleaning up test data...")
        async with vector_client.engine.begin() as conn:
            await conn.execute(text("DELETE FROM ai_solution_embeddings WHERE question_id = :qid"), {"qid": test_question_id})
        logger.info("✓ Test data cleaned up")
        
        logger.info("✓ All pgvector tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ pgvector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await vector_client.engine.dispose()

if __name__ == "__main__":
    success = asyncio.run(test_pgvector_connection())
    sys.exit(0 if success else 1)
