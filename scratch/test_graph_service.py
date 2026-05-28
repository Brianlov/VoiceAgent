import asyncio
from graph_service import GraphRAGProcessor
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "20010808")
    db = os.getenv("NEO4J_DB", "vectorchunkgraph")
    
    processor = GraphRAGProcessor(
        neo4j_uri=uri,
        neo4j_user=user,
        neo4j_password=pwd,
        neo4j_db=db
    )
    
    # Test a few queries
    queries = [
        "Where did Victoria spend the Christmas of 1900?",
        "In the 18th century there was one global trading hub for large diamonds, what was it?",
        "When did the Australian mandolin movement begin?",
        "What has excessive hunting contributed heavily to?"
    ]
    
    for q in queries:
        print("\n" + "="*60)
        print(f"QUERY: {q}")
        print("="*60)
        context = await processor.retrieve(q, as_list=False)
        print(f"RESULT:\n{context}")

if __name__ == "__main__":
    asyncio.run(main())
