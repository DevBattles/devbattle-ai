import os
import asyncio
from google import genai
from dotenv import load_dotenv

load_dotenv()

async def main():
    api_key = os.getenv("GEMINI_API_KEY")
    print(f"Testing Gemini Key: {api_key[:10]}...")
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Hello!"
        )
        print("Success! Response:", response.text)
    except Exception as e:
        print("Error:", e)

asyncio.run(main())
