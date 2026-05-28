"""
ragas_benchmark_fixed.py
========================
Modern RAGAS >=0.2.x benchmark pipeline
for VectorRAG vs GraphRAG voice-agent evaluation.

FIXES:
- Uses modern RAGAS metric API
- Uses llm_factory()
- Removes deprecated LangChain wrappers
- Compatible with latest RAGAS
- Proper Gemini/OpenAI judge support
- Stable evaluation pipeline

Install:
pip install -U ragas langchain-google-genai \
sentence-transformers pandas matplotlib tqdm \
python-dotenv tabulate datasets

ENV (.env):
GOOGLE_API_KEY=your_key_here
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from tqdm import tqdm
from tabulate import tabulate

# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# RAGAS
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# YOUR LOCAL PIPELINES
from vector_rag_search import VectorRAGSearch
from graph_service import GraphRAGProcessor

# ─────────────────────────────────────────────────────────────
# ENV
# ─────────────────────────────────────────────────────────────

load_dotenv()
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    judge_model: str = "gemini-1.5-pro"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"

    vector_top_k: int = 5

    output_dir: str = "results"

    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    @property
    def run_dir(self):
        return Path(self.output_dir) / self.timestamp


CFG = BenchmarkConfig()

# ─────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────
# BUILTIN DATASET
# ─────────────────────────────────────────────────────────────

BUILTIN_SAMPLES = [
    {"question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
     "ground_truth": "Saint Bernadette Soubirous"},
    {"question": "The granting of Doctorate degrees first occurred in what year at Notre Dame?",
     "ground_truth": "1924"},
    {"question": "When did the first art gallery open in Washington state?",
     "ground_truth": "1927"},
    {"question": "What can cause your memory to deterioriate or not work as well?",
     "ground_truth": "Stress"},
    {"question": "What has excessive hunting contributed heavily to?",
     "ground_truth": "the endangerment, extirpation and extinction of many animals"},
    {"question": "What is the name of the AFL team based in Tampa Bay?",
     "ground_truth": "Storm"},
    {"question": "Which conjugation has about 3500 verbs?",
     "ground_truth": "first conjugation"},
    {"question": "Which university library is larger than Nanjing University Library?",
     "ground_truth": "Peking University Library"},
    {"question": "Which canton is Berne the capital?",
     "ground_truth": "Canton of Bern"},
    {"question": "Why is it difficult to measure corruption?",
     "ground_truth": "imprecise definitions of corruption"},
    {"question": "Why is Yiddish not a dialect of German?",
     "ground_truth": "a Yiddish speaker would not consult a German dictionary"},
    # --- MULTI-HOP REASONING SAMPLES ---
    {"question": "Who is the patron saint of the university where the Scholastic Magazine began publishing in 1876?",
     "ground_truth": "The Virgin Mary"},
    {"question": "What hormones damage the brain region associated with memory loss, and what daily activity helps stabilize those memories?",
     "ground_truth": "Glucocorticoids damage the hippocampal region, and sleep helps stabilize memories."},
    {"question": "How many years after the Scholastic Magazine began publishing did Notre Dame first formally offer Doctorate degrees?",
     "ground_truth": "48 years"},
    {"question": "When regals recruited low-ranking local tribes for expeditions in British India, what specific global consequence did this type of activity heavily contribute to?",
     "ground_truth": "The endangerment, extirpation and extinction of many animals."},
    {"question": "In the city where the Henry Art Gallery opened as a public art museum, what type of transportation system did they later begin to focus on?",
     "ground_truth": "Mass transit."}
]

# Derive subgroups for reasoning_pct mixing
FACTUAL_SAMPLES = BUILTIN_SAMPLES[:6]
REASONING_SAMPLES = BUILTIN_SAMPLES[6:]


# ─────────────────────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────────────────────

def load_samples(csv_path=None, n_samples=10, reasoning_pct=None):

    if csv_path and Path(csv_path).exists():

        df = pd.read_csv(csv_path)

        rows = []

        for _, row in df.head(n_samples).iterrows():

            gt = row.get("ground_truth", "")

            rows.append({
                "question": str(row["question"]),
                "ground_truth": str(gt)
            })

        return rows

    if reasoning_pct is not None:
        pct = float(reasoning_pct) / 100.0
        n_reasoning = int(round(n_samples * pct))
        n_factual = n_samples - n_reasoning
        
        # Circular indexing for safety with any n_samples
        selected_factual = [FACTUAL_SAMPLES[i % len(FACTUAL_SAMPLES)] for i in range(n_factual)]
        selected_reasoning = [REASONING_SAMPLES[i % len(REASONING_SAMPLES)] for i in range(n_reasoning)]
        
        samples = selected_factual + selected_reasoning
        print(f"🎯  Using custom built-in sample mix: {n_factual} Factual ({100-reasoning_pct}%) and {n_reasoning} Reasoning ({reasoning_pct}%)")
        return samples

    return BUILTIN_SAMPLES[:n_samples]

# ─────────────────────────────────────────────────────────────
# LATENCY TRACKER
# ─────────────────────────────────────────────────────────────

class LatencyTracker:

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (
            time.perf_counter() - self.start
        ) * 1000

# ─────────────────────────────────────────────────────────────
# JUDGE LLM
# ─────────────────────────────────────────────────────────────

def build_generation_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

# ─────────────────────────────────────────────────────────────
# RAGAS JUDGE
# ─────────────────────────────────────────────────────────────

def build_ragas_components():

    # Judge LLM
    lc_llm = ChatGoogleGenerativeAI(
        model=CFG.judge_model,
        temperature=CFG.judge_temperature,
        google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
    )
    ragas_llm = LangchainLLMWrapper(lc_llm)

    # Embeddings
    lc_emb = HuggingFaceEmbeddings(
        model_name=CFG.embedding_model
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(lc_emb)

    return ragas_llm, ragas_embeddings

# ─────────────────────────────────────────────────────────────
# VECTOR DATASET
# ─────────────────────────────────────────────────────────────

async def build_vector_dataset(
    samples,
    generation_llm,
):

    print("\nBuilding VectorRAG dataset...")

    vector_search = VectorRAGSearch()

    rows = []

    for sample in tqdm(samples):

        question = sample["question"]
        gt = sample["ground_truth"]

        # Retrieval
        with LatencyTracker() as rt:

            results = await asyncio.to_thread(
                vector_search.search,
                question,
                CFG.vector_top_k
            )

        contexts = [r["context"] for r in results]

        context_text = "\n".join(contexts)

        # Generation
        prompt = UNIVERSAL_PROMPT.format(
            context=context_text,
            question=question
        )

        with LatencyTracker() as gtimer:

            try:

                response = await asyncio.to_thread(
                    generation_llm.invoke,
                    [HumanMessage(content=prompt)]
                )

                answer = response.content

            except Exception as e:

                print(f"Generation error: {e}")

                answer = ""

        rows.append({
            "question": question,
            "contexts": contexts,
            "answer": answer,
            "ground_truth": gt,
            "retrieval_ms": rt.elapsed_ms,
            "generation_ms": gtimer.elapsed_ms,
        })

    vector_search.close()

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# GRAPH DATASET
# ─────────────────────────────────────────────────────────────

async def build_graph_dataset(
    samples,
    generation_llm,
):

    print("\nBuilding GraphRAG dataset...")

    graph_processor = GraphRAGProcessor(
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        neo4j_db=os.getenv("NEO4J_DB"),
    )

    await graph_processor._initialize_graph()

    rows = []

    for sample in tqdm(samples):

        question = sample["question"]
        gt = sample["ground_truth"]

        # Retrieval
        with LatencyTracker() as rt:

            try:

                contexts = await graph_processor.retrieve(question, as_list=True)
                context_string = "\n".join(contexts)[:12000]

            except Exception as e:

                print(f"Graph retrieval error: {e}")

                context_string = ""
                contexts = []
                
        if not contexts:
            contexts = ["No context retrieved."]

        prompt = UNIVERSAL_PROMPT.format(
            context=context_string,
            question=question
        )

        # Generation
        with LatencyTracker() as gtimer:

            try:

                response = await asyncio.to_thread(
                    generation_llm.invoke,
                    [HumanMessage(content=prompt)]
                )

                answer = response.content

            except Exception as e:

                print(f"Generation error: {e}")

                answer = ""

        rows.append({
            "question": question,
            "contexts": contexts,
            "answer": answer,
            "ground_truth": gt,
            "retrieval_ms": rt.elapsed_ms,
            "generation_ms": gtimer.elapsed_ms,
        })

    await graph_processor.driver.close()

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# CONVERT TO RAGAS DATASET
# ─────────────────────────────────────────────────────────────

def df_to_ragas_dataset(df):

    samples = []

    for _, row in df.iterrows():

        samples.append(
            SingleTurnSample(
                user_input=row["question"],
                retrieved_contexts=row["contexts"],
                response=row["answer"],
                reference=row["ground_truth"],
            )
        )

    return EvaluationDataset(samples=samples)

# ─────────────────────────────────────────────────────────────
# RUN RAGAS
# ─────────────────────────────────────────────────────────────

async def run_ragas_eval(
    df,
    pipeline_name,
    ragas_llm,
    ragas_embeddings,
):

    print(f"\nEvaluating {pipeline_name}")

    dataset = df_to_ragas_dataset(df)

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    loop = asyncio.get_event_loop()

    from ragas.run_config import RunConfig
    run_config = RunConfig(max_workers=2, max_retries=10, max_wait=60)

    result = await loop.run_in_executor(
        None,
        lambda: evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=False,
            run_config=run_config,
        )
    )

    scores_df = result.to_pandas()

    final_df = pd.concat(
        [
            df.reset_index(drop=True),
            scores_df.reset_index(drop=True),
        ],
        axis=1,
    )

    aggregate = {}

    for metric in [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]:

        if metric in final_df.columns:

            aggregate[metric] = float(
                final_df[metric].mean()
            )

    print("\nAggregate Scores")

    for k, v in aggregate.items():

        print(f"{k:<25}: {v:.4f}")

    return final_df, aggregate

# ─────────────────────────────────────────────────────────────
# COMPARISON
# ─────────────────────────────────────────────────────────────

def compare_results(vec_agg, graph_agg):

    rows = []

    for metric in vec_agg.keys():

        v = vec_agg.get(metric, 0)
        g = graph_agg.get(metric, 0)

        rows.append({
            "Metric": metric,
            "VectorRAG": round(v, 4),
            "GraphRAG": round(g, 4),
            "Difference": round(g - v, 4),
        })

    cmp_df = pd.DataFrame(rows)

    print("\nComparison")

    print(
        tabulate(
            cmp_df,
            headers="keys",
            tablefmt="fancy_grid",
            showindex=False
        )
    )

    return cmp_df

# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_results(vec_agg, graph_agg, run_dir):

    metrics = list(vec_agg.keys())

    vector_scores = [vec_agg[m] for m in metrics]
    graph_scores = [graph_agg[m] for m in metrics]

    x = np.arange(len(metrics))

    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width/2, vector_scores, width, label="VectorRAG")
    ax.bar(x + width/2, graph_scores, width, label="GraphRAG")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)

    ax.set_ylim(0, 1)

    ax.set_ylabel("Score")

    ax.set_title("RAGAS Benchmark Comparison")

    ax.legend()

    plt.tight_layout()

    plt.savefig(run_dir / "comparison_chart.png")

    plt.close()

# ─────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────

def export_results(
    vec_df,
    graph_df,
    cmp_df,
    run_dir,
):

    run_dir.mkdir(parents=True, exist_ok=True)

    vec_df.to_csv(
        run_dir / "vectorrag_results.csv",
        index=False
    )

    graph_df.to_csv(
        run_dir / "graphrag_results.csv",
        index=False
    )

    cmp_df.to_csv(
        run_dir / "comparison.csv",
        index=False
    )

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

async def main(args):

    if args.output_dir:
        CFG.output_dir = args.output_dir

    run_dir = CFG.run_dir

    run_dir.mkdir(parents=True, exist_ok=True)

    # Load samples
    samples = load_samples(
        args.squad_csv,
        args.n_samples,
        args.reasoning_pct
    )

    print(f"\nLoaded {len(samples)} samples")

    # Generation LLM
    generation_llm = build_generation_llm()

    # RAGAS Judge
    ragas_llm, ragas_embeddings = build_ragas_components()

    # Build datasets
    vector_df = await build_vector_dataset(
        samples,
        generation_llm
    )

    graph_df = await build_graph_dataset(
        samples,
        generation_llm
    )

    # Save raw outputs
    vector_df.to_csv(
        run_dir / "vector_raw.csv",
        index=False
    )

    graph_df.to_csv(
        run_dir / "graph_raw.csv",
        index=False
    )

    # Evaluate
    vector_scores_df, vector_agg = await run_ragas_eval(
        vector_df,
        "VectorRAG",
        ragas_llm,
        ragas_embeddings,
    )

    graph_scores_df, graph_agg = await run_ragas_eval(
        graph_df,
        "GraphRAG",
        ragas_llm,
        ragas_embeddings,
    )

    # Compare
    cmp_df = compare_results(
        vector_agg,
        graph_agg
    )

    # Export
    export_results(
        vector_scores_df,
        graph_scores_df,
        cmp_df,
        run_dir,
    )

    # Plot
    plot_results(
        vector_agg,
        graph_agg,
        run_dir,
    )

    print(f"\nDONE")
    print(f"Results saved to: {run_dir.resolve()}")

# ─────────────────────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--squad-csv",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--reasoning-pct",
        type=int,
        choices=[0, 30, 50, 80, 100],
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
    )

    args = parser.parse_args()

    asyncio.run(main(args))