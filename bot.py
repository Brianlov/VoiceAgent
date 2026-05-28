#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Quickstart Example.

The example runs a simple voice AI bot that you can connect to using your
browser and speak with it. You can also deploy this bot to Pipecat Cloud.

Required AI services:
- Deepgram (Speech-to-Text)
- OpenAI (LLM)
- Cartesia (Text-to-Speech)

Run the bot using::

    uv run bot.py
"""

import os
import sys
import logging

from dotenv import load_dotenv
from loguru import logger


print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Local Smart Turn Analyzer V3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

logger.info("✅ Local Smart Turn Analyzer V3 loaded")
logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, Frame, TextFrame
import time
from pipecat.processors.frame_processor import FrameProcessor

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams

logger.info("✅ All components loaded successfully!")

# ── Suppress pipecat/uvicorn DEBUG noise (must be AFTER pipecat imports) ─────
logger.remove()                          # remove all handlers pipecat may have added
logger.add(sys.stderr, level="INFO")     # only INFO+ for our own logs
logger.disable("pipecat")               # silence all pipecat.* loguru calls
logging.getLogger("pipecat").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
# ─────────────────────────────────────────────────────────────────────────────


# --- Import Both RAG Processors ---
from rag_context_processor import RAGContextProcessor
from graph_service import GraphRAGProcessor, AnswerLoggerProcessor

load_dotenv(override=True)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2:1.5b")
    OLLAMA_API_KEY = "ollama"

    llm = OpenAILLMService(
        api_key=OLLAMA_API_KEY,
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
    )

    messages = [
        {
            "role": "system",
            "content": "Answer ONLY using the provided [CONTEXT]. Extract the EXACT answer from the text. If the information is not present in the [CONTEXT] or the context is empty, say: 'I'm sorry, I couldn't find any specific records for that in the database.'"
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # --- A/B TESTING TOGGLE ---
    rag_mode = os.getenv("RAG_MODE", "VECTORS").upper()
    logger.info(f"🚀 INITIALIZING PIPECAT IN {rag_mode} RAG MODE")

    if rag_mode == "GRAPH":
        rag_processor = GraphRAGProcessor(
            neo4j_uri=os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "20010808"),
           # neo4j_db="squaddatasetknowledgegraph"
           #neo4j_db="squadgraphrag",
           #neo4j_db="squadv3rag"
           neo4j_db="vectorchunkgraph"

        )
        await rag_processor._initialize_graph()
    else:
        rag_processor = RAGContextProcessor()
        await rag_processor.warmup_embeddings()

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            context_aggregator.user(),  # User responses
            rag_processor,
            llm,
            AnswerLoggerProcessor(),    # Logs Answer out to graphrag_log.txt
            tts,
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @task.event_handler("on_metrics")
    async def on_metrics(task, metrics):
        # Official Pipecat metrics: STT latency, LLM TTFT, etc.
        for service, data in metrics.items():
            if "ttft" in data:
                print(f"📊 [METRIC] {service} - Time to First Token: {data['ttft']:.2f}s")
            elif "latency" in data:
                print(f"📊 [METRIC] {service} - Latency: {data['latency']:.2f}s")
            elif "processing_time" in data:
                print(f"📊 [METRIC] {service} - Processing Time: {data['processing_time']:.2f}s")
            else:
                print(f"📊 [METRIC] {service}: {data}")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Say hello and briefly introduce yourself."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
        "local": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
