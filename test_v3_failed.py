import asyncio
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Load env variables
load_dotenv(r"c:\Users\Brian ooi\Documents\code\VoiceAgentv1\.env")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "20010808")
NEO4J_DB = os.getenv("NEO4J_DB", "hotpotqarag")

from Graph_service_purever4 import GraphRAGProcessorV4

async def main():
    processor = GraphRAGProcessorV4(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB)
    await processor._initialize_graph()
    
    df = pd.read_csv(r"c:\Users\Brian ooi\Documents\code\VoiceAgentv1\generations_hotpotqa\graphrag_generations_ollama.csv")
    sorry_rows = df[df['answer'].str.contains('sorry', na=False, case=False)]
    
    print(f"Found {len(sorry_rows)} failed questions. Testing top 3 with v3...")
    
    llm = ChatOllama(model="gemma3:12b-cloud", temperature=0.0)
    
    for i, (_, row) in enumerate(sorry_rows.head(3).iterrows()):
        q = row['question']
        print(f"\n======================================")
        print(f"Testing Failed Question #{i+1}: {q}")
        print(f"Ground Truth: {row['ground_truth']}")
        
        context = await processor.retrieve(q)
        print(f"Returned Context Length: {len(context)} characters")
        
        prompt = f"[CONTEXT]\n{context}\n\n[QUESTION]\n{q}\n\nInstructions:\n1. Read the [CONTEXT] carefully. It contains paragraphs and reasoning paths.\n2. Find the exact answer to the [QUESTION] using the reasoning paths if necessary.\n3. Provide a natural, conversational sentence as your answer.\n4. If the answer is not in the [CONTEXT], say: 'I'm sorry, I couldn't find that in the database.'\n"
        
        response = llm.invoke(prompt)
        print(f"\nv3 LLM ANSWER: {response.content}")

if __name__ == "__main__":
    # Fix for Windows loop event policy
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
