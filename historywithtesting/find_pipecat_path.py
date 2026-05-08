import os
import sys

# Try common pipecat paths for the turn strategy
paths = [
    "pipecat.processors.aggregators.llm_response",
    "pipecat.processors.aggregators.user_response",
    "pipecat.processors.aggregators.user_turn",
    "pipecat.audio.turn.turn_analyzer",
]

for path in paths:
    try:
        module = __import__(path, fromlist=["TranscriptionUserTurnStopStrategy"])
        if hasattr(module, "TranscriptionUserTurnStopStrategy"):
            print(f"✅ FOUND at: from {path} import TranscriptionUserTurnStopStrategy")
    except ImportError:
        continue
