import asyncio
import httpx

async def main():
    payload = {
        "currentRole": "student",
        "currentQuestion": {
            "title": "semantic tags",
            "description": "What are semantic HTML tags?",
            "category": "HTML",
            "difficulty": "Easy"
        },
        "currentContext": {
            "type": "practice",
            "isActive": True
        },
        "chatHistory": [],
        "message": "tell me about semantic tags"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post("http://127.0.0.1:8000/internal/mentor/chat", json=payload, timeout=60.0)
            print("Status Code:", response.status_code)
            print("Response:", response.json())
        except Exception as e:
            print("Error:", e)

asyncio.run(main())
