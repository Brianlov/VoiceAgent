import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import asyncio
from pipecat.frames.frames import LLMMessagesFrame
from rag_context_processor import RAGContextProcessor

async def main():
    print("Initializing RAG processor...")
    processor = RAGContextProcessor()
    
    messages = [
        {"role": "user", "content": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"}
    ]
    frame = LLMMessagesFrame(messages=messages)
    
    print("Enriching with context...")
    enriched_frame = await processor._enrich_with_context(frame)
    
    print("\n--- ENRICHED MESSAGES ---")
    for msg in enriched_frame.messages:
        print(f"Role: {msg['role']}")
        print(f"Content:\n{msg['content']}\n")
        print("-" * 40)
        
    print("Done. Check if rag_retrieval_log.txt was updated.")

if __name__ == "__main__":
    asyncio.run(main())    
