import sys
import os
from unittest.mock import AsyncMock, MagicMock

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Setup mock environment configurations
os.environ["GEMINI_API_KEY"] = "mock_gemini_api_key_value"
os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:postgres@localhost:5432/mock_db"
os.environ["SUPABASE_URL"] = "https://mock.supabase.co"
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "mock_service_role_key_value"

from app.graph.nodes import provider, vector_client, renderer

# Mock Provider, Database Client, and Visual Browser Screenshot Renderer
provider.generate_text = AsyncMock()
provider.generate_multimodal = AsyncMock()
provider.get_embedding = AsyncMock(return_value=[0.1] * 768)

vector_client.initialize_db = AsyncMock()
vector_client.save_solutions = AsyncMock()
vector_client.get_similar_solutions = AsyncMock(return_value=[
    {"code": "<h1>Mock Solutions</h1>", "type": "semantic_html_flexbox", "similarity": 0.95}
])

renderer.capture_screenshot = AsyncMock(return_value=b"mock_png_screenshot_bytes")

# Create a mock session to prevent real db query attempts
mock_session = AsyncMock()

async def mock_execute(query, params=None):
    q_str = str(query).lower()
    mock_res = MagicMock()
    if "rubric" in q_str:
        mock_res.fetchone.return_value = (
            {
                "correctness": {"max_points": 30, "checklist": ["Requirement check 1"]},
                "responsiveness": {"max_points": 20, "checklist": ["Requirement check 1"]},
                "accessibility": {"max_points": 15, "checklist": ["Requirement check 1"]},
                "performance": {"max_points": 15, "checklist": ["Requirement check 1"]},
                "code_quality": {"max_points": 20, "checklist": ["Requirement check 1"]}
            },
        )
    else:
        # Default mock question metadata
        mock_res.fetchone.return_value = (
            "Navbar Layout",
            "Construct a responsive navbar.",
            {"index.html": {"content": "..."}},
            "Sliding header menu"
        )
    return mock_res

mock_session.execute = AsyncMock(side_effect=mock_execute)

# Async Context Manager simulation wrapper class
class AsyncContextManagerMock:
    async def __aenter__(self):
        return mock_session
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

# Force async_session queries to resolve the context manager mock in-memory
vector_client.async_session = MagicMock(return_value=AsyncContextManagerMock())

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    print("\n[Test Health Check] Response:", response.json())
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_question_generate():
    provider.generate_text.return_value = '''{
      "solutions": [
        {
          "type": "semantic_html_flexbox",
          "code": "<h1>Generated HTML</h1>",
          "explanation": "Uses standard layout"
        }
      ],
      "rubric": {
        "correctness": { "max_points": 30, "checklist": ["Requirement check 1"] },
        "responsiveness": { "max_points": 20, "checklist": ["Requirement check 1"] },
        "accessibility": { "max_points": 15, "checklist": ["Requirement check 1"] },
        "performance": { "max_points": 15, "checklist": ["Requirement check 1"] },
        "code_quality": { "max_points": 20, "checklist": ["Requirement check 1"] }
      }
    }'''

    payload = {
        "questionId": "00000000-0000-0000-0000-000000000001",
        "version": 1,
        "title": "Navbar Layout",
        "description": "Construct a responsive navbar.",
        "requirements": ["Must support flexbox"],
        "starterFiles": {"index.html": {"content": "..."}},
        "expectedOutput": "Sliding header menu"
    }

    response = client.post("/internal/questions/generate", json=payload)
    print("\n[Test Solutions Gen] Response:", response.json())
    assert response.status_code == 200
    assert response.json()["success"] is True

def test_submission_evaluate():
    # Mock text code evaluator LLM output JSON
    provider.generate_text.return_value = '''{
      "correctness": { "score": 28, "feedback": "Matches requirements" },
      "responsiveness": { "score": 18, "feedback": "Clean breakpoints" },
      "accessibility": { "score": 14, "feedback": "Has proper elements" },
      "performance": { "score": 13, "feedback": "Fast rendering" },
      "code_quality": { "score": 19, "feedback": "Clear formatting" },
      "strengths": ["Strong correctness", "Great code style"],
      "weaknesses": ["Minor structure checklist"],
      "improvements": ["Optimize inline styles"],
      "feedback": "Outstanding evaluation."
    }'''

    # Mock Vision layout LLM output JSON
    provider.generate_multimodal.return_value = '''{
      "visual_score": 92,
      "visual_feedback": "Perfect layout layout matches the design expectations.",
      "strengths": ["Visual layout match"],
      "weaknesses": ["None"]
    }'''

    payload = {
        "questionId": "00000000-0000-0000-0000-000000000001",
        "version": 1,
        "studentFiles": {"index.html": {"content": "..."}},
        "githubUrl": "https://github.com/test/repo"
    }

    response = client.post("/internal/submissions/evaluate", json=payload)
    print("\n[Test Submit Evaluation] Response:", response.json())
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert "score" in response.json()["data"]
    assert "grade" in response.json()["data"]

def test_mentor_chat():
    provider.generate_text.return_value = "Here is a hint: check the display flex rules."

    payload = {
        "currentRole": "student",
        "currentQuestion": {"title": "Navbar"},
        "currentContext": {"type": "homework", "isActive": True, "deadline": "2026-08-01T12:00:00Z"},
        "chatHistory": [],
        "message": "How do I align items in a row?"
    }

    response = client.post("/internal/mentor/chat", json=payload)
    print("\n[Test Mentor Chat] Response:", response.json())
    assert response.status_code == 200
    assert "response" in response.json()["data"]

if __name__ == "__main__":
    print("Starting AI Backend tests run...")
    test_health()
    test_question_generate()
    test_submission_evaluate()
    test_mentor_chat()
    print("\nAll AI Backend tests completed successfully!")
