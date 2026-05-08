"""Functional test for Deepgram STT and Cartesia TTS APIs.

Tests actual functionality:
- Cartesia: Convert sample text to speech (TTS)
- Deepgram: Transcribe the generated audio back to text (STT)

This validates that both APIs are working end-to-end.

Usage:
    python test_deepgram_cartesia_functional.py

Requires env vars:
    DEEPGRAM_API_KEY
    CARTESIA_API_KEY
"""

import io
import json
import os
import sys
from typing import Any, Dict

import requests

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
TIMEOUT = float(os.getenv("TIMEOUT", "30"))

DEEPGRAM_BASE_URL = "https://api.deepgram.com/v1"
CARTESIA_BASE_URL = "https://api.cartesia.ai"

# Sample text to test
SAMPLE_TEXT = "Hello! This is a test of the Deepgram and Cartesia APIs. They are working great!"


def exit_error(message: str, payload: Dict[str, Any] | None = None) -> None:
    print(f"[ERROR] {message}")
    if payload:
        print(json.dumps(payload, indent=2))
    sys.exit(1)


def test_cartesia_tts() -> bytes:
    """Test Cartesia TTS - convert text to speech."""
    print("\n" + "="*70)
    print("🔊 Testing Cartesia TTS (Text-to-Speech)...")
    print("="*70)
    
    if not CARTESIA_API_KEY:
        exit_error("CARTESIA_API_KEY not set in environment")
    
    print(f"[INFO] Input text: '{SAMPLE_TEXT}'")
    print(f"[INFO] Length: {len(SAMPLE_TEXT)} characters")
    
    url = f"{CARTESIA_BASE_URL}/api/tts/stream"
    
    payload = {
        "model_id": "sonic-english",
        "transcript": SAMPLE_TEXT,
        "voice": {
            "mode": "id",
            "id": "248be419-c900-4dc8-9e6f-2ea000d6d039",  # Prelude voice
        },
        "stream": False,  # Get full response at once
    }
    
    headers = {
        "X-API-Key": CARTESIA_API_KEY,
        "Cartesia-Version": "2025-04-16",
        "Content-Type": "application/json",
        "User-Agent": "Pipecat-Quickstart-Test",
    }
    
    print(f"[INFO] POST {url}")
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        exit_error("Cartesia TTS request failed", {"error": str(exc)})
    
    if resp.status_code != 200:
        exit_error(
            f"Cartesia TTS returned {resp.status_code}",
            {"status": resp.status_code, "body": resp.text}
        )
    
    audio_data = resp.content
    print(f"[✅ OK] Cartesia TTS successful!")
    print(f"[INFO] Generated audio: {len(audio_data)} bytes")
    
    return audio_data


def test_deepgram_stt(audio_data: bytes) -> str:
    """Test Deepgram STT - transcribe audio to text."""
    print("\n" + "="*70)
    print("🎤 Testing Deepgram STT (Speech-to-Text)...")
    print("="*70)
    
    if not DEEPGRAM_API_KEY:
        exit_error("DEEPGRAM_API_KEY not set in environment")
    
    print(f"[INFO] Input audio: {len(audio_data)} bytes")
    
    url = f"{DEEPGRAM_BASE_URL}/listen"
    
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "User-Agent": "Pipecat-Quickstart-Test",
    }
    
    params = {
        "model": "nova-2",
        "smart_format": "true",
    }
    
    print(f"[INFO] POST {url}")
    try:
        resp = requests.post(
            url,
            headers=headers,
            params=params,
            data=audio_data,
            timeout=TIMEOUT
        )
    except Exception as exc:  # noqa: BLE001
        exit_error("Deepgram STT request failed", {"error": str(exc)})
    
    if resp.status_code != 200:
        exit_error(
            f"Deepgram STT returned {resp.status_code}",
            {"status": resp.status_code, "body": resp.text}
        )
    
    data = resp.json()
    transcript = data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
    
    if not transcript:
        exit_error("No transcript returned from Deepgram", {"response": data})
    
    print(f"[✅ OK] Deepgram STT successful!")
    print(f"[INFO] Transcribed text: '{transcript}'")
    
    return transcript


def compare_texts(original: str, transcribed: str) -> None:
    """Compare original and transcribed text."""
    print("\n" + "="*70)
    print("📊 Comparing Results...")
    print("="*70)
    
    print(f"\n[ORIGINAL]    '{original}'")
    print(f"[TRANSCRIBED] '{transcribed}'")
    
    # Simple comparison - check if key words are present
    original_words = set(original.lower().split())
    transcribed_words = set(transcribed.lower().split())
    
    common_words = original_words & transcribed_words
    match_percentage = (len(common_words) / len(original_words) * 100) if original_words else 0
    
    print(f"\n[INFO] Word match: {match_percentage:.1f}%")
    print(f"[INFO] Common words: {len(common_words)} / {len(original_words)}")
    
    if match_percentage >= 80:
        print("[✅ GREAT] High fidelity transcription!")
    elif match_percentage >= 50:
        print("[⚠️  OK] Moderate fidelity transcription")
    else:
        print("[❌ WARNING] Low fidelity transcription")


def main() -> None:
    print("🚀 Functional Test: Deepgram STT + Cartesia TTS\n")
    
    # Check if API keys are set
    if not DEEPGRAM_API_KEY:
        exit_error("DEEPGRAM_API_KEY not set in environment")
    if not CARTESIA_API_KEY:
        exit_error("CARTESIA_API_KEY not set in environment")
    
    try:
        # Test Cartesia TTS
        audio_data = test_cartesia_tts()
        
        # Test Deepgram STT
        transcribed_text = test_deepgram_stt(audio_data)
        
        # Compare results
        compare_texts(SAMPLE_TEXT, transcribed_text)
        
        print("\n" + "="*70)
        print("✨ All functional tests completed!")
        print("="*70 + "\n")
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        exit_error(f"Unexpected error: {str(exc)}")


if __name__ == "__main__":
    main()
