import os
import asyncio
import pandas as pd
import textstat
from datasets import Dataset
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from ragas import evaluate
from ragas.metrics import answer_correctness, answer_relevancy
from ragas.run_config import RunConfig

# Import your local RAG processors
from vector_rag_search import VectorRAGSearch
from graph_service import GraphRAGProcessor  # Using your newest file

# The Universal Prompt to ensure a fair test between both methods
UNIVERSAL_PROMPT = """You are a helpful conversational voice assistant.
Answer the [QUESTION] using ONLY the provided [CONTEXT].

The [CONTEXT] may contain raw text paragraphs, structured data relationships, or both. 
Your task is to synthesize the context and provide the correct answer in a natural, fluent, and conversational spoken format.

Instructions:
1. Read the [CONTEXT] carefully.
2. Find the exact answer to the [QUESTION] in the [CONTEXT].
3. Provide a natural, conversational sentence as your answer. Do not read out bullet points or structured formats.
4. If the answer is not in the [CONTEXT], say: "I'm sorry, I couldn't find that in the database."

[CONTEXT]
{context}

[QUESTION]
{question}
"""

async def run_evaluation():
    print("🚀 Initializing Evaluation Stack...")

    # 1. Initialize RAG pipelines
    vector_search = VectorRAGSearch()
    
    graph_processor = GraphRAGProcessor(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "20010808"),
        neo4j_db=os.getenv("NEO4J_DB", "vectorchunkgraph") # Use the correct database that has the indexes
    )
    await graph_processor._initialize_graph()

    # 2. Initialize Cloud Judge (Google Gemini)
    try:
         print("🤖 Connecting to Cloud Judge (Google Gemini 2.5 Flash)...")
         from langchain_google_genai import ChatGoogleGenerativeAI
         local_judge = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)
         local_embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    except Exception as e:
         print(f"❌ Failed to connect to Gemini: {e}")
         return

    # 3. Load SQuAD Sample
    print("📊 Loading SQuAD dataset sample...")
    samples = [
        {"question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?", "answers": "Saint Bernadette Soubirous"},
    #     {"question": "What is in front of the Notre Dame Main Building?", "answers": "a copper statue of Christ"},
    #     {"question": "When did the Scholastic Magazine of Notre dame begin publishing?", "answers": "September 1876"},
    #     {"question": "The granting of Doctorate degrees first occurred in what year at Notre Dame?", "answers": "1924"},
    #     {"question": "The Lobund Institute was merged into the Department of Biology at Notre Dame in what year?", "answers": "1958"},
    #     {"question": "When did the first art gallery open in Washington state?", "answers": "1927"},
    #     {"question": "On what type of transportation system has Seattle begun to focus?", "answers": "mass transit"},
    #     {"question": "What can cause your memory to deterioriate or not work as well?", "answers": "Stress"},
    #     {"question": "In studies what is a relationship between sleeping and learning?", "answers": "activation patterns in the sleeping brain that mirror those recorded during the learning of tasks from the previous day"},
    #     {"question": "In the 1990s, what type of programming    changed the handling of databases?", "answers": "object-oriented"}
    ]
    df = pd.DataFrame(samples)

    results = {
        "question": [],
        "ground_truth": [],
        "vector_context": [],
        "graph_context": [],
        "vector_answer": [],
        "graph_answer": [],
        "vector_fluency_score": [],
        "graph_fluency_score": []
    }

    print("\n⚡ Generating Answers (This will take a moment)...")
    for index, row in df.iterrows():
        question = row['question']
        ground_truth = str(row.get('answers', ''))

        print(f"-> Processing Q: {question[:50]}...")
        
        # --- Vector RAG Pipeline ---
        vec_results = vector_search.search(question, top_k=5)
        vec_context = "\n".join([res['context'] for res in vec_results])
        
        # --- Graph RAG Pipeline ---
        graph_context = await graph_processor.retrieve(question)

        # --- Generate Voice Answers via LLM ---
        vec_prompt = UNIVERSAL_PROMPT.format(context=vec_context, question=question)
        graph_prompt = UNIVERSAL_PROMPT.format(context=graph_context, question=question)

        vec_answer = local_judge.invoke([HumanMessage(content=vec_prompt)]).content
        graph_answer = local_judge.invoke([HumanMessage(content=graph_prompt)]).content

        # --- Fluency (Syntactic) Evaluation ---
        # Flesch-Kincaid Grade: Lower score = easier to read = more conversational/fluent for voice
        vec_fluency = textstat.flesch_kincaid_grade(vec_answer)
        graph_fluency = textstat.flesch_kincaid_grade(graph_answer)

        # Store data
        results["question"].append(question)
        results["ground_truth"].append(ground_truth)
        results["vector_context"].append(vec_context)
        results["graph_context"].append(graph_context)
        results["vector_answer"].append(vec_answer)
        results["graph_answer"].append(graph_answer)
        results["vector_fluency_score"].append(vec_fluency)
        results["graph_fluency_score"].append(graph_fluency)

    # 4. Run Ragas Evaluation
    print("\n⚖️ Running RAGAS Evaluation on VectorRAG Answers...")
    vector_dataset = Dataset.from_dict({
        "question": results["question"],
        "answer": results["vector_answer"],
        "contexts": [[c] for c in results["vector_context"]],
        "ground_truth": results["ground_truth"]
    })
    # Evaluating answer correctness against ground truth using Llama-3 as judge
    vector_ragas_score = evaluate(
        vector_dataset, 
        metrics=[answer_correctness], 
        llm=local_judge,
        embeddings=local_embeddings,
        run_config=RunConfig(max_workers=1, timeout=300)
    )

    print("\n⚖️ Running RAGAS Evaluation on GraphRAG Answers...")
    graph_dataset = Dataset.from_dict({
        "question": results["question"],
        "answer": results["graph_answer"],
        "contexts": [[c] for c in results["graph_context"]],
        "ground_truth": results["ground_truth"]
    })
    graph_ragas_score = evaluate(
        graph_dataset, 
        metrics=[answer_correctness], 
        llm=local_judge,
        embeddings=local_embeddings,
        run_config=RunConfig(max_workers=1, timeout=300)
    )

    # 5. Output Summary
    print("\n" + "="*50)
    print("📈 EVALUATION SUMMARY (Vector vs Graph)")
    print("="*50)
    
    avg_vec_fluency = sum(results["vector_fluency_score"]) / len(results["vector_fluency_score"])
    avg_graph_fluency = sum(results["graph_fluency_score"]) / len(results["graph_fluency_score"])

    print("\n[FLUENCY (textstat)] -> Lower grade is better for Voice Agents (easier to speak/listen)")
    print(f"VectorRAG Average Readability Grade: {avg_vec_fluency:.2f}")
    print(f"GraphRAG Average Readability Grade:  {avg_graph_fluency:.2f}")
    
    print("\n[ACCURACY (Ragas)] -> Higher score is better (0.0 to 1.0)")
    
    vec_ac = vector_ragas_score['answer_correctness']
    if isinstance(vec_ac, list): vec_ac = sum(vec_ac) / len(vec_ac) if vec_ac else 0.0
    print(f"VectorRAG Answer Correctness: {vec_ac:.4f}")
    
    graph_ac = graph_ragas_score['answer_correctness']
    if isinstance(graph_ac, list): graph_ac = sum(graph_ac) / len(graph_ac) if graph_ac else 0.0
    print(f"GraphRAG Answer Correctness:  {graph_ac:.4f}")
    
    # Save detailed results to inspect where one beat the other
    df_results = pd.DataFrame(results)
    df_results.to_csv("vector_vs_graph_evaluation.csv", index=False)
    print("\n✅ Detailed line-by-line results saved to 'vector_vs_graph_evaluation.csv'")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
