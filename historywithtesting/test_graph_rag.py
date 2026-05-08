import os
import asyncio
from dotenv import load_dotenv

# Import the processor from your service file
from graph_service import GraphRAGProcessor

async def test_graph_rag():
    print("Loading environment variables...")
    load_dotenv(override=True)
    
    # Initialize the processor with credentials from .env
    print("Initializing GraphRAGProcessor...")
    rag_processor = GraphRAGProcessor(
        neo4j_uri=os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "20010808"),
        neo4j_db="squaddatasetknowledgegraph"
    )
    
    # Connect to the Neo4j Graph
    await rag_processor._initialize_graph()
    
    # List of test questions
    test_queries = [
        "What is Lourdes?",
        "Who is Bernadette Soubirous?",
        "Can you tell me about something that doesn't exist in the database?"
    ]
    
    print("\n" + "="*50)
    print("🧪 RUNNING GRAPHRAG RETRIEVAL TESTS")
    print("="*50)
    
    for query in test_queries:
        print(f"\n[QUERY] {query}")
        
        # Test the core retrieve mechanism
        context = await rag_processor.retrieve(query)
        
        if context:
            print(f"[DATA RETRIEVED]\n{context}")
        else:
            print("[DATA RETRIEVED]\n>>> Nothing found (returned empty string) <<<")

if __name__ == "__main__":
    asyncio.run(test_graph_rag())
