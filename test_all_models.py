import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

models = ["models/gemini-2.5-pro", "models/gemini-2.5-flash", "models/gemini-2.0-flash", "models/gemini-1.5-pro"]

for m in models:
    try:
        response = client.models.generate_content(
            model=m,
            contents="Hello!"
        )
        print(f"{m}: Success! Response: {response.text[:20]}...")
    except Exception as e:
        print(f"{m}: Error: {e}")
