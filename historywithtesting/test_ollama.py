"""Quick Ollama health check against localhost:11343.

- Lists models via /v1/models
- Runs a short chat completion against the specified model

Usage:
    # activate your venv first if you want
    python test_ollama.py

Override defaults with env vars:
    OLLAMA_BASE_URL=http://localhost:11343/v1
    OLLAMA_MODEL=granite3.1-dense:latest
"""

import json
import os
import sys
from typing import Any, Dict

import requests

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("OLLAMA_MODEL", "granite3.1-dense:latest")
TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "15"))


def exit_error(message: str, payload: Dict[str, Any] | None = None) -> None:
    print(f"[ERROR] {message}")
    if payload:
        print(json.dumps(payload, indent=2))
    sys.exit(1)


def check_models() -> None:
    url = f"{BASE_URL}/models"
    print(f"[INFO] GET {url}")
    try:
        resp = requests.get(url, timeout=TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        exit_error("Request failed (models)", {"error": str(exc)})
    if resp.status_code != 200:
        exit_error("Non-200 from /models", {"status": resp.status_code, "body": resp.text})
    data = resp.json()
    print("[OK] Models response:")
    print(json.dumps(data, indent=2))
    if not data.get("data"):
        exit_error("No models returned from /models")


def check_chat() -> None:
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "max_tokens": 64,
        "stream": False,
    }
    print(f"[INFO] POST {url} model={MODEL}")
    try:
        resp = requests.post(url, json=payload, timeout=TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        exit_error("Request failed (chat)", {"error": str(exc)})
    if resp.status_code != 200:
        exit_error("Non-200 from /chat/completions", {"status": resp.status_code, "body": resp.text})
    data = resp.json()
    print("[OK] Chat response:")
    print(json.dumps(data, indent=2))
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    if not content:
        exit_error("Chat response missing content", data)
    print(f"[RESULT] {content.strip()}")


def main() -> None:
    print(f"[INFO] Using BASE_URL={BASE_URL} MODEL={MODEL}")
    check_models()
    check_chat()
    print("[SUCCESS] Ollama endpoint is reachable and responded with a chat completion.")


if __name__ == "__main__":
    main()
