import sys
import os
import asyncio
from dotenv import load_dotenv
import httpx

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
load_dotenv(dotenv_path=os.path.join(project_root, ".env"))

test_prompts = [
    "What are semantic tags?",
    "Explain closures in JavaScript.",
    "What is React Fiber?",
    "Who is Virat Kohli?",
    "Write a Python binary search.",
    "Explain Docker."
]

async def main():
    async with httpx.AsyncClient(timeout=60.0) as client:
        print("--- RUNNING DIAGNOSTIC TEST FOR REQUESTED CHAT PROMPTS ---")
        for p in test_prompts:
            payload = {
                "currentRole": "student",
                "currentQuestion": {},
                "currentContext": {"type": "chat", "isActive": True},
                "chatHistory": [],
                "message": p
            }
            try:
                print(f"\nUser Query: {p}")
                res = await client.post("http://127.0.0.1:8000/internal/mentor/chat", json=payload)
                data = res.json()
                response_content = data.get("data", {}).get("response", "")
                print(f"AI Response:\n{response_content}\n")
                print("-" * 50)
            except Exception as e:
                print(f"Failed query '{p}': {e}")

if __name__ == "__main__":
    asyncio.run(main())
