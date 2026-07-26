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

# Mutable holder so individual tests can override the mocked question_versions row returned
# for "retrieve_question" queries (e.g. to simulate an HTML question with a specific starter
# file), while keeping a sane default for tests that don't care about these fields.
_default_question_meta_row = (
    "Navbar Layout",
    "Construct a responsive navbar.",
    {"index.html": {"content": "..."}},
    "Sliding header menu",
    None,   # category
    None,   # workspace_type
    None,   # evaluation_strategy
    None,   # supported_language
    False,  # preview_required
    None,   # execution_mode
    None,   # options
)
_mock_question_meta_row = {"value": _default_question_meta_row}


def set_mock_question_meta_row(row_tuple):
    _mock_question_meta_row["value"] = row_tuple


def reset_mock_question_meta_row():
    _mock_question_meta_row["value"] = _default_question_meta_row


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
        # Mock question metadata (overridable per-test via set_mock_question_meta_row)
        mock_res.fetchone.return_value = _mock_question_meta_row["value"]
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


def test_broken_html_structure_caps_score():
    """
    Regression test for: a teacher's HTML boilerplate includes an opening <body> tag, the
    student deletes it, and the AI grader still hallucinates partial credit (previously
    observed handing out e.g. 15%) despite the page being fundamentally broken. The
    deterministic structural_validation node (app/graph/structural_checks.py) must now hard-cap
    the final score regardless of what the LLM says.
    """
    starter_html = "<!DOCTYPE html><html><head><title>Navbar</title></head><body><nav>Home</nav></body></html>"
    # Student deleted the opening <body> tag (closing </body> also removed here, but even just
    # dropping the opening tag alone is enough to trigger the guardrail).
    broken_student_html = "<!DOCTYPE html><html><head><title>Navbar</title></head><nav>Home</nav></html>"

    set_mock_question_meta_row((
        "Navbar Layout",
        "Construct a responsive navbar.",
        {"index.html": {"content": starter_html}},
        "Sliding header menu",
        "HTML",             # category
        "html",             # workspace_type
        "ui_playwright",    # evaluation_strategy
        "html",             # supported_language
        False,              # preview_required (keep False so Playwright/vision isn't needed in this unit test)
        "browser",          # execution_mode
        None,               # options
    ))

    try:
        # Simulate the AI grader hallucinating a generous score despite the broken structure.
        provider.generate_text.return_value = '''{
          "score": 65,
          "strengths": ["Nice use of semantic nav element"],
          "weaknesses": ["Minor spacing issue"],
          "improvements": ["Add ARIA roles"],
          "feedback": "Solid layout overall."
        }'''

        payload = {
            "questionId": "00000000-0000-0000-0000-000000000001",
            "version": 1,
            "studentFiles": {"index.html": {"content": broken_student_html}},
            "githubUrl": None
        }

        response = client.post("/internal/submissions/evaluate", json=payload)
        print("\n[Test Broken HTML Structure] Response:", response.json())
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["score"] <= 15, f"Expected score capped at <=15 for missing <body> tag, got {data['score']}"
        assert any("<body>" in w for w in data["feedback"]["weaknesses"]), \
            "Expected the missing <body> tag issue to be surfaced in weaknesses feedback"
    finally:
        reset_mock_question_meta_row()


def test_unmodified_starter_code_caps_score():
    """An unchanged starter submission must receive zero even when the LLM is generous."""
    starter_html = "<!DOCTYPE html><html><head><title>Navbar</title></head><body><nav>Home</nav></body></html>"

    set_mock_question_meta_row((
        "Navbar Layout",
        "Construct a responsive navbar.",
        {"index.html": {"content": starter_html}},
        "Sliding header menu",
        "HTML",
        "html",
        "ui_playwright",
        "html",
        False,
        "browser",
        None,
    ))

    try:
        provider.generate_text.return_value = '''{
          "score": 90,
          "strengths": ["Good markup"],
          "weaknesses": [],
          "improvements": [],
          "feedback": "Looks complete."
        }'''

        response = client.post("/internal/submissions/evaluate", json={
            "questionId": "00000000-0000-0000-0000-000000000001",
            "version": 1,
            "studentFiles": {"index.html": {"content": starter_html}},
            "githubUrl": None,
        })
        print("\n[Test Unmodified Starter Code] Response:", response.json())
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["score"] == 0, f"Expected unchanged starter code to score 0, got {data['score']}"
        assert any("unchanged" in weakness.lower() for weakness in data["feedback"]["weaknesses"]), \
            "Expected the unchanged starter-code issue in weaknesses feedback"
    finally:
        reset_mock_question_meta_row()


if __name__ == "__main__":
    print("Starting AI Backend tests run...")
    test_health()
    test_question_generate()
    test_submission_evaluate()
    test_mentor_chat()
    test_broken_html_structure_caps_score()
    test_unmodified_starter_code_caps_score()
    print("\nAll AI Backend tests completed successfully!")
