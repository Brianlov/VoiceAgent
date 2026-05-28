import asyncio
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv(r"c:\Users\Brian ooi\Documents\code\VoiceAgentv1\.env")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "20010808")
NEO4J_DB = os.getenv("NEO4J_DB", "hotpotqarag")

from Graph_service_purever3 import GraphRAGProcessorV3

async def main():
    processor = GraphRAGProcessorV3(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB)
    await processor._initialize_graph()
    
    questions = [
        "Is the Armenian Gampr dog or The Chihuahua the smaller breed?",
        "Who directed a film written by the same pairing that later wrote Wild Wild West?"
    ]
    
    for q in questions:
        print(f"\n======================================")
        print(f"Testing Question: {q}")
        print(f"======================================")
        context = await processor.retrieve(q)
        print(f"\nReturned Context Length: {len(context)} characters")
        print(context)

asyncio.run(main())
