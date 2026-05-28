import os
from dotenv import load_dotenv
from groq import Groq

# Load .env variables
load_dotenv()

# Try to get the key from env
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("⚠️ GROQ_API_KEY environment variable is not set!")
    
client = Groq()
try:
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
          {
            "role": "user",
            "content": "Say hello in one word."
          }
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None
    )
    print("✅ Connection Successful!")
    print("Response:", completion.choices[0].message.content)
except Exception as e:
    print("❌ Connection Failed!")
    print("Error:", e)
