import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel(model_name="gemini-flash-latest")

# Test simple query
response = model.generate_content("Hello, how are you?")

print("Response type:", type(response))
print("Response attributes:", dir(response))
print("\nHas 'parts':", hasattr(response, 'parts'))
print("Has 'text':", hasattr(response, 'text'))

try:
    print("\nTrying response.text:", response.text)
except Exception as e:
    print(f"\nError accessing response.text: {e}")

try:
    print("\nTrying response.parts:", response.parts)
    if response.parts:
        for i, part in enumerate(response.parts):
            print(f"Part {i}:", part)
            if hasattr(part, 'text'):
                print(f"Part {i} text:", part.text)
except Exception as e:
    print(f"\nError accessing response.parts: {e}")

print("\n\nFull response:")
print(response)
