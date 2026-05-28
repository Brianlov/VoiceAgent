import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("Listing all available models for grounded generation:")
try:
    models = list(genai.list_models())
    for m in models:
        # Looking for models that support generateContent
        if 'generateContent' in m.supported_generation_methods:
            name = m.name.lower()
            if any(k in name for k in ["1.5", "1.0", "pro", "flash"]):
                print(f"- {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")
