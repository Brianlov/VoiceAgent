"""RAG Context Processor - Retrieves relevant context from knowledge base.

This processor enriches user messages with relevant context from the knowledge base
before passing them to the LLM.
"""

import asyncio
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import LLMContextFrame
from loguru import logger
from vector_rag_search import VectorRAGSearch

class RAGContextProcessor(FrameProcessor):
    """Adds relevant context from knowledge base to user messages."""
    
    def __init__(self):
        super().__init__()
        self.searcher = VectorRAGSearch()
        logger.info(f"RAG Context Processor initialized with vector search")
        
    async def warmup_embeddings(self):
        logger.info("Initializing VectorRAG Context Processor warmup...")
        await asyncio.to_thread(self.searcher.search, "warmup", 1)
    
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            messages = frame.context.messages
            user_messages = [msg for msg in messages if msg.get("role") == "user"]

            if user_messages:
                latest_query = user_messages[-1].get("content", "")
                if isinstance(latest_query, str) and latest_query.strip():
                    print(f"🔍 VectorRAG: Searching for: '{latest_query}'")

                    results = await asyncio.to_thread(self.searcher.search, latest_query, 2)

                    contexts = []
                    for r in results:
                        contexts.append(f"[{r['title']}]\n{r['context']}")

                    context_text = "\n\n".join(contexts) if contexts else "No relevant context found."

                    # Log to file
                    with open("rag_retrieval_log.txt", "a", encoding="utf-8") as rf:
                        rf.write(f"=== Query: {latest_query} ===\n")
                        rf.write(context_text + "\n\n")

                    print(f"📄 VectorRAG CONTEXT:\n{'='*60}\n{context_text}\n{'='*60}")

                    enriched_prompt = f"[CONTEXT]\n{context_text}\n\n[QUESTION]\n{latest_query}"
                    user_messages[-1]["content"] = enriched_prompt
                    print(f"💡 VectorRAG: Injected context into LLMContextFrame")

        await self.push_frame(frame, direction)
