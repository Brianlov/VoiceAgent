"""Quick health check for Deepgram (STT) and Cartesia (TTS).

- Tests Deepgram Speech-to-Text API
- Tests Cartesia Text-to-Speech API

Usage:
    # activate your venv first if you want
    python test_deepgram_cartesia.py

Requires env vars:
    DEEPGRAM_API_KEY
    CARTESIA_API_KEY
"""

import json
import os
import sys
from typing import Any, Dict

import requests

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
TIMEOUT = float(os.getenv("TIMEOUT", "10"))

DEEPGRAM_BASE_URL = "https://api.deepgram.com/v1"
CARTESIA_BASE_URL = "https://api.cartesia.ai/tts/stream"


def exit_error(message: str, payload: Dict[str, Any] | None = None) -> None:
    print(f"[ERROR] {message}")
    if payload:
        print(json.dumps(payload, indent=2))
    sys.exit(1)


def check_deepgram() -> None:
    """Test Deepgram STT API."""
    print("\n" + "="*60)
    print("🎤 Testing Deepgram (Speech-to-Text)...")
    print("="*60)
    
    if not DEEPGRAM_API_KEY:
        exit_error("DEEPGRAM_API_KEY not set in environment")
    
    # Test the Deepgram API by getting available models
    url = f"{DEEPGRAM_BASE_URL}/models"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "User-Agent": "Pipecat-Quickstart-Test",
    }
    
    print(f"[INFO] GET {url}")
    try:
        resp = requests.get(url, headers=headers, timeout=TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        exit_error("Deepgram request failed", {"error": str(exc)})
    
    if resp.status_code != 200:
        exit_error(
            f"Deepgram API returned {resp.status_code}",
            {"status": resp.status_code, "body": resp.text}
        )
    
    data = resp.json()
    print("[✅ OK] Deepgram is working!")
    print(f"[INFO] Available models: {len(data.get('models', []))} models")
    
    # Show first model as example
    if data.get('models'):
        first_model = data['models'][0]
        print(f"[INFO] Example: {first_model.get('name')} - UUID: {first_model.get('uuid')}")


def check_cartesia() -> None:
    """Test Cartesia TTS API."""
    print("\n" + "="*60)
    print("🔊 Testing Cartesia (Text-to-Speech)...")
    print("="*60)
    
    if not CARTESIA_API_KEY:
        exit_error("CARTESIA_API_KEY not set in environment")
    
    # Test Cartesia by making a request to get voices
    # Note: We'll use the voices endpoint to verify API key is valid
    voices_url = "https://api.cartesia.ai/voices"
    headers = {
        "X-API-Key": CARTESIA_API_KEY,
        "Cartesia-Version": "2025-04-16",
        "User-Agent": "Pipecat-Quickstart-Test",
    }
    
    print(f"[INFO] GET {voices_url}")
    try:
        resp = requests.get(voices_url, headers=headers, timeout=TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        exit_error("Cartesia request failed", {"error": str(exc)})
    
    if resp.status_code != 200:
        exit_error(
            f"Cartesia API returned {resp.status_code}",
            {"status": resp.status_code, "body": resp.text}
        )
    
    data = resp.json()
    print("[✅ OK] Cartesia is working!")
    
    # Count available voices
    voices_list = data.get('voices', [])
    print(f"[INFO] Available voices: {len(voices_list)} voices")
    
    # Show first few voice names as examples
    if voices_list:
        voice_names = [v.get('name') for v in voices_list[:3]]
        print(f"[INFO] Examples: {', '.join(voice_names)}")


def main() -> None:
    print("🚀 Testing Deepgram and Cartesia APIs...\n")
    
    # Check if API keys are set
    if not DEEPGRAM_API_KEY:
        print("[⚠️  WARNING] DEEPGRAM_API_KEY not set")
    if not CARTESIA_API_KEY:
        print("[⚠️  WARNING] CARTESIA_API_KEY not set")
    
    if not DEEPGRAM_API_KEY and not CARTESIA_API_KEY:
        exit_error("No API keys configured. Please set DEEPGRAM_API_KEY and/or CARTESIA_API_KEY")
    
    try:
        if DEEPGRAM_API_KEY:
            check_deepgram()
        if CARTESIA_API_KEY:
            check_cartesia()
        
        print("\n" + "="*60)
        print("✨ All tests passed!")
        print("="*60 + "\n")
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        exit_error(f"Unexpected error: {str(exc)}")


if __name__ == "__main__":
    main()
