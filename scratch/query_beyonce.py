import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Ensure parent directory is in python search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

load_dotenv()
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Import pipelines
from vector_rag_search import VectorRAGSearch
from graph_service_pure import GraphRAGProcessor

UNIVERSAL_PROMPT = """
You are a helpful conversational voice assistant.

Answer the QUESTION using ONLY the CONTEXT.

If answer is not found, say:
"I'm sorry, I couldn't find that in the database."

CONTEXT:
{context}

QUESTION:
{question}
"""

async def main():
    question = "In what city and state did Beyonce  grow up? "
    print(f"❓ Querying pipelines for: '{question}'\n")

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # 1. VECTOR RAG
    print("--- 🔍 VectorRAG Pipeline ---")
    vector_search = VectorRAGSearch()
    results = vector_search.search(question, top_k=5)
    vector_contexts = [r["context"] for r in results]
    vector_context_text = "\n".join(vector_contexts)
    vector_search.close()

    prompt_vector = UNIVERSAL_PROMPT.format(context=vector_context_text, question=question)
    response_vector = llm.invoke([HumanMessage(content=prompt_vector)])
    vector_answer = response_vector.content

    print(f"\nRetrieved {len(results)} chunks.")
    print("VectorRAG Context Content:")
    print("-" * 50)
    print(vector_context_text)
    print("-" * 50)
    print(f"🤖 VectorRAG Generated Answer: {vector_answer}\n")


    # 2. GRAPH RAG (PURE)
    print("--- 🧠 GraphRAG (Pure) Pipeline ---")
    graph_processor = GraphRAGProcessor(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "20010808"),
        neo4j_db=os.getenv("NEO4J_DB", "llmknowledgegraph"),
    )
    await graph_processor._initialize_graph()
    graph_context_text = await graph_processor.retrieve(question)
    await graph_processor.driver.close()

    prompt_graph = UNIVERSAL_PROMPT.format(context=graph_context_text, question=question)
    response_graph = llm.invoke([HumanMessage(content=prompt_graph)])
    graph_answer = response_graph.content

    print("GraphRAG Context Content:")
    print("-" * 50)
    print(graph_context_text)
    print("-" * 50)
    print(f"🤖 GraphRAG Generated Answer: {graph_answer}\n")

if __name__ == "__main__":
    asyncio.run(main())
