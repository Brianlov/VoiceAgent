
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # Try to see if it's set as GOOGLE_API_KEY alternatively
    api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: GEMINI_API_KEY not found in environment variables.")
    exit(1)

genai.configure(api_key=api_key)

FILE_PATH = "docs/knowledge_base.txt"
DISPLAY_NAME = "Pipecat Knowledge Base"

def upload_file():
    if not os.path.exists(FILE_PATH):
        print(f"❌ Error: File {FILE_PATH} not found.")
        return

    print(f"Uploading {FILE_PATH} to Gemini...")
    
    try:
        file = genai.upload_file(
            path=FILE_PATH,
            display_name=DISPLAY_NAME
        )
        print(f"✅ File uploaded successfully!")
        print(f"File Name (ID): {file.name}")
        print(f"Display Name: {file.display_name}")
        print(f"URI: {file.uri}")
        
        # Verify state
        print("Verifying file state...")
        import time
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            file = genai.get_file(file.name)
        
        print(f"\nFinal State: {file.state.name}")
        
    except Exception as e:
        print(f"❌ Error uploading file: {e}")

if __name__ == "__main__":
    upload_file()
