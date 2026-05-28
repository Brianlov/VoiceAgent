"""
ragas_benchmark.py
==================
End-to-end RAGAS evaluation: runs your VectorRAG & GraphRAG pipelines live,
collects contexts + answers, then judges with Gemini 2.5 Pro.

Judge LLM : Google Gemini 2.5 Pro  (strongest available)
Embeddings: all-mpnet-base-v2  (same model used by both RAG pipelines)
RAGAS API : >= 0.2.x

Dependencies (add to existing venv):
  pip install ragas>=0.2.0 langchain-google-genai langchain-huggingface \
              sentence-transformers pandas matplotlib tqdm python-dotenv tabulate

Usage:
  python ragas_benchmark.py                          # uses built-in SQuAD samples
  python ragas_benchmark.py --squad-csv squad_train_dataset.csv --n-samples 50
  python ragas_benchmark.py --output-dir results/run1
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

# Local RAG pipelines
from vector_rag_search import VectorRAGSearch
#from Graph_service_purever1 import GraphRAGProcessor
#from Graph_service_purever2 import GraphRAGProcessor
#from Graph_service_purever3 import GraphRAGProcessorV3 as GraphRAGProcessor
from Graph_service_purever4 import GraphRAGProcessorV4 as GraphRAGProcessor


matplotlib.use("Agg")
load_dotenv()
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# ── Judge provider — change this one line ─────────────────────────────────────
# Options:
#   "gemini"         → Gemini 2.5 Flash via Google AI Studio (free, key already in .env)
#   "groq"           → Gemma 2 9B via Groq cloud (free tier, 30 RPM)
#   "ollama"         → llama3 locally — already installed on your machine (4.7GB)
#   "ollama_cloud"   → gpt-oss:20b-cloud via Ollama cloud (runs on their servers, not your PC)
JUDGE_PROVIDER = "ollama_cloud"

_JUDGE_MODELS = {
    "gemini":       "models/gemini-2.5-flash",
    "groq":         "llama-3.3-70b-versatile",  # 70B ultra-fast Groq model!
    "ollama":       "llama3:latest",            # Your local Llama 3 8B model!
    "ollama_cloud": "gemma4:31b-cloud",
}

@dataclass
class BenchmarkConfig:
    judge_model: str = field(default_factory=lambda: _JUDGE_MODELS[JUDGE_PROVIDER])
    judge_temperature: float = 0.0
    embedding_model: str = "all-mpnet-base-v2"
    vector_top_k: int = 5
    raise_on_failure: bool = False
    output_dir: str = "results"
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.timestamp

CFG = BenchmarkConfig()

# Prompt identical to evaluate_squad_rag.py for fair comparison
UNIVERSAL_PROMPT = """\
You are a helpful conversational voice assistant.
Answer the [QUESTION] using ONLY the provided [CONTEXT].

Instructions:
1. Read the [CONTEXT] carefully.
2. Find the exact answer to the [QUESTION] in the [CONTEXT].
3. Provide a natural, conversational sentence as your answer.
4. If the answer is not in the [CONTEXT], say: "I'm sorry, I couldn't find that in the database."

[CONTEXT]
{context}

[QUESTION]
{question}
"""

# ─────────────────────────────────────────────────────────────────────────────
# LLM & Embeddings
# ─────────────────────────────────────────────────────────────────────────────

def build_generator_llm():
    """
    Returns the LLM used for RAG answer generation.
    Always uses gemma3:12b-cloud via Ollama.
    """
    from langchain_ollama import ChatOllama
    print(f"⚙️   Generator LLM: gemma3:12b-cloud")
    return ChatOllama(model="gemma3:12b-cloud", temperature=0.0)


def build_judge_llm():
    """
    Returns (raw_lc_llm, ragas_wrapped_llm).
    Change JUDGE_PROVIDER at the top of the file to switch backends.
    """
    if JUDGE_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        print(f"⚡  Judge LLM (Groq cloud — free): {CFG.judge_model}")
        lc_llm = ChatGroq(
            model=CFG.judge_model,
            temperature=CFG.judge_temperature,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
    elif JUDGE_PROVIDER in ("ollama", "ollama_cloud"):
        from langchain_ollama import ChatOllama
        label = "Ollama cloud" if JUDGE_PROVIDER == "ollama_cloud" else "local Ollama"
        print(f"🦙  Judge LLM ({label}): {CFG.judge_model}")
        lc_llm = ChatOllama(
            model=CFG.judge_model,
            temperature=CFG.judge_temperature,
            timeout=300.0,  # Wait up to 5 minutes for the huge 31B cloud model to respond!
        )
    else:  # gemini (default)
        print(f"🤖  Judge LLM (Gemini API — free): {CFG.judge_model}")
        lc_llm = ChatGoogleGenerativeAI(
            model=CFG.judge_model,
            temperature=CFG.judge_temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        )
    return lc_llm, LangchainLLMWrapper(lc_llm)


def build_embeddings() -> LangchainEmbeddingsWrapper:
    print(f"📐  Embeddings: {CFG.embedding_model}")
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=CFG.embedding_model))


def build_metrics(llm: LangchainLLMWrapper, emb: LangchainEmbeddingsWrapper) -> list[Any]:
    metrics = [
        Faithfulness(llm=llm),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]
    # Groq does not support n > 1 parameters required by AnswerRelevancy
    if JUDGE_PROVIDER != "groq":
        metrics.append(AnswerRelevancy(llm=llm, embeddings=emb))
    return metrics

# ─────────────────────────────────────────────────────────────────────────────
# SQuAD dataset loader
# ─────────────────────────────────────────────────────────────────────────────

# Fallback built-in samples if no CSV is provided
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


def load_squad_samples(json_path: str | None, n_samples: int) -> list[dict]:
    """Load question/ground_truth pairs from SQuAD JSON or fall back to built-ins."""
    effective_path = json_path if json_path else "test_samples_300vec.json"
    if Path(effective_path).exists():
        print(f"📂  Loading SQuAD from JSON: {effective_path}  (n={n_samples})")
        with open(effective_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = []
        for item in data[:n_samples]:
            samples.append({
                "question": item["question"],
                "ground_truth": item.get("answers", item.get("ground_truth", ""))
            })
        return samples
    else:
        print(f"⚠️   No JSON found at {effective_path} — using {len(BUILTIN_SAMPLES)} built-in SQuAD samples.")
        return BUILTIN_SAMPLES[:n_samples]

# ─────────────────────────────────────────────────────────────────────────────
# Latency tracking
# ─────────────────────────────────────────────────────────────────────────────

class LatencyTracker:
    def __init__(self):
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1_000


def percentiles(values: list[float]) -> dict[str, float]:
    arr = np.array(values)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms":  float(np.percentile(arr, 50)),
        "p95_ms":  float(np.percentile(arr, 95)),
        "max_ms":  float(np.max(arr)),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline dataset builders
# ─────────────────────────────────────────────────────────────────────────────

async def build_vector_dataset(
    samples: list[dict],
    judge_llm: ChatGoogleGenerativeAI,
    checkpoint_path: Path,
) -> pd.DataFrame:
    """
    For each sample, retrieve via VectorRAG then generate an answer with Gemini.
    Returns DataFrame with: question, contexts, answer, ground_truth, retrieval_ms, generation_ms
    Resumes from checkpoint if it exists.
    """
    print("\n🔍  Building VectorRAG eval dataset...")
    
    if checkpoint_path.exists():
        try:
            existing_df = pd.read_csv(checkpoint_path)
            # Parse 'contexts' column back to list from string representation
            existing_df["contexts"] = existing_df["contexts"].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else [x]
            )
            print(f"🔄  Found existing VectorRAG pipeline cache with {len(existing_df)} samples.")
        except Exception:
            existing_df = pd.DataFrame()
    else:
        existing_df = pd.DataFrame()

    completed_questions = set(existing_df["question"].tolist()) if not existing_df.empty else set()
    remaining_samples = [s for s in samples if s["question"] not in completed_questions]

    if not remaining_samples:
        print("🎉  All VectorRAG pipeline outputs loaded from cache!")
        return existing_df.head(len(samples))

    vector_search = VectorRAGSearch()
    rows = list(existing_df.to_dict(orient="records")) if not existing_df.empty else []
    retrieval_times, gen_times = [], []

    print(f"🚀  Querying VectorRAG pipeline for {len(remaining_samples)} remaining samples...")
    for sample in tqdm(remaining_samples, desc="VectorRAG retrieval+generation"):
        q = sample["question"]
        gt = sample["ground_truth"]

        # Retrieval (sync, run in thread to not block event loop)
        print(f"\n🔍 [Query {len(rows)+1}/{len(remaining_samples)+len(rows)}] Retrieving context for: '{q[:50]}...'")
        with LatencyTracker() as ret_t:
            results = await asyncio.to_thread(vector_search.search, q, CFG.vector_top_k)
        retrieval_times.append(ret_t.elapsed_ms)
        print(f"   ↳ Retrieved {len(results)} chunks in {ret_t.elapsed_ms:.1f}ms")

        contexts = [r["context"] for r in results]
        context_text = "\n".join(contexts)

        # Generation
        print(f"⚙️  Generating answer using {judge_llm.model}...")
        prompt = UNIVERSAL_PROMPT.format(context=context_text, question=q)
        with LatencyTracker() as gen_t:
            try:
                answer = (await asyncio.to_thread(
                    judge_llm.invoke, [HumanMessage(content=prompt)]
                )).content
            except Exception as e:
                print(f"  ⚠️  Generation failed for '{q[:40]}': {e}")
                answer = ""
        gen_times.append(gen_t.elapsed_ms)
        print(f"   ↳ Generation completed in {gen_t.elapsed_ms:.1f}ms")

        rows.append({
            "question": q,
            "contexts": contexts,
            "answer": answer,
            "ground_truth": gt,
            "retrieval_ms": ret_t.elapsed_ms,
            "generation_ms": gen_t.elapsed_ms,
        })
        
        # Save cache incrementally
        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    vector_search.close()
    if retrieval_times:
        print(f"    Retrieval  — {percentiles(retrieval_times)}")
    if gen_times:
        print(f"    Generation — {percentiles(gen_times)}")
    return pd.DataFrame(rows).head(len(samples))


async def build_graph_dataset(
    samples: list[dict],
    judge_llm: ChatGoogleGenerativeAI,
    checkpoint_path: Path,
) -> pd.DataFrame:
    """
    For each sample, retrieve via GraphRAG then generate an answer with Gemini.
    Returns DataFrame with: question, contexts, answer, ground_truth, retrieval_ms, generation_ms
    Resumes from checkpoint if it exists.
    """
    print("\n🧠  Building GraphRAG eval dataset...")
    
    if checkpoint_path.exists():
        try:
            existing_df = pd.read_csv(checkpoint_path)
            existing_df["contexts"] = existing_df["contexts"].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else [x]
            )
            print(f"🔄  Found existing GraphRAG pipeline cache with {len(existing_df)} samples.")
        except Exception:
            existing_df = pd.DataFrame()
    else:
        existing_df = pd.DataFrame()

    completed_questions = set(existing_df["question"].tolist()) if not existing_df.empty else set()
    remaining_samples = [s for s in samples if s["question"] not in completed_questions]

    if not remaining_samples:
        print("🎉  All GraphRAG pipeline outputs loaded from cache!")
        return existing_df.head(len(samples))

    graph_processor = GraphRAGProcessor(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "20010808"),
        neo4j_db=os.getenv("NEO4J_DB", "llmknowledgegraph"),
    )
    await graph_processor._initialize_graph()

    rows = list(existing_df.to_dict(orient="records")) if not existing_df.empty else []
    retrieval_times, gen_times = [], []

    print(f"🚀  Querying GraphRAG pipeline for {len(remaining_samples)} remaining samples...")
    for sample in tqdm(remaining_samples, desc="GraphRAG retrieval+generation"):
        q = sample["question"]
        gt = sample["ground_truth"]

        # Retrieval (async)
        print(f"\n🔍 [Query {len(rows)+1}/{len(remaining_samples)+len(rows)}] Retrieving context for: '{q[:50]}...'")
        with LatencyTracker() as ret_t:
            try:
                contexts = await graph_processor.retrieve(q, as_list=True)
                context_str = "\n".join(contexts)[:12000]
            except Exception as e:
                print(f"  ⚠️  GraphRAG retrieval failed for '{q[:40]}': {e}")
                context_str = ""
                contexts = []
        retrieval_times.append(ret_t.elapsed_ms)
        print(f"   ↳ Retrieved {len(contexts)} chunks in {ret_t.elapsed_ms:.1f}ms")

        if not contexts:
            contexts = ["No context retrieved."]

        # Generation
        print(f"⚙️  Generating answer using {judge_llm.model}...")
        prompt = UNIVERSAL_PROMPT.format(context=context_str, question=q)
        with LatencyTracker() as gen_t:
            try:
                answer = (await asyncio.to_thread(
                    judge_llm.invoke, [HumanMessage(content=prompt)]
                )).content
            except Exception as e:
                print(f"  ⚠️  Generation failed for '{q[:40]}': {e}")
                answer = ""
        gen_times.append(gen_t.elapsed_ms)
        print(f"   ↳ Generation completed in {gen_t.elapsed_ms:.1f}ms")

        rows.append({
            "question": q,
            "contexts": contexts,
            "answer": answer,
            "ground_truth": gt,
            "retrieval_ms": ret_t.elapsed_ms,
            "generation_ms": gen_t.elapsed_ms,
        })
        
        # Save cache incrementally
        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    await graph_processor.driver.close()
    if retrieval_times:
        print(f"    Retrieval  — {percentiles(retrieval_times)}")
    if gen_times:
        print(f"    Generation — {percentiles(gen_times)}")
    return pd.DataFrame(rows).head(len(samples))

# ─────────────────────────────────────────────────────────────────────────────
# RAGAS evaluation engine
# ─────────────────────────────────────────────────────────────────────────────

RENAME = {
    "faithfulness":     "Faithfulness",
    "answer_relevancy": "AnswerRelevancy",
    "context_precision":"ContextPrecision",
    "context_recall":   "ContextRecall",
}


def df_to_ragas_dataset(df: pd.DataFrame) -> EvaluationDataset:
    samples = []
    for _, row in df.iterrows():
        ctxs = row["contexts"]
        if isinstance(ctxs, str):
            try:
                ctxs = json.loads(ctxs)
            except json.JSONDecodeError:
                ctxs = [ctxs]
        samples.append(SingleTurnSample(
            user_input=str(row["question"]),
            retrieved_contexts=ctxs,
            response=str(row["answer"]),
            reference=str(row["ground_truth"]),
        ))
    return EvaluationDataset(samples=samples)


async def run_ragas_eval(
    df: pd.DataFrame,
    pipeline_name: str,
    metrics: list[Any],
    checkpoint_path: Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    print(f"\n{'='*60}")
    print(f"  ⚖️  RAGAS Evaluating: {pipeline_name}  ({len(df)} samples)")
    print(f"{'='*60}")

    if checkpoint_path.exists():
        try:
            existing_df = pd.read_csv(checkpoint_path)
            print(f"🔄  Found existing RAGAS evaluation checkpoint with {len(existing_df)} samples for {pipeline_name}.")
        except Exception:
            existing_df = pd.DataFrame()
    else:
        existing_df = pd.DataFrame()

    completed_questions = set(existing_df["question"].tolist()) if not existing_df.empty else set()
    unevaluated_df = df[~df["question"].isin(completed_questions)].copy()

    if unevaluated_df.empty:
        print(f"🎉  All samples for {pipeline_name} already evaluated in checkpoint.")
        scores_df = existing_df
    else:
        print(f"⏳  Evaluating {len(unevaluated_df)} remaining samples for {pipeline_name}...")
        
        from ragas.run_config import RunConfig
        run_config = RunConfig(max_workers=2, max_retries=10, max_wait=300)
        loop = asyncio.get_event_loop()
        new_scores_list = []

        for idx, row in tqdm(unevaluated_df.iterrows(), total=len(unevaluated_df), desc=f"RAGAS [{pipeline_name}]"):
            single_row_df = pd.DataFrame([row])
            single_dataset = df_to_ragas_dataset(single_row_df)

            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: evaluate(
                        dataset=single_dataset,
                        metrics=metrics,
                        raise_exceptions=CFG.raise_on_failure,
                        run_config=run_config,
                    ),
                )
                single_score_df = result.to_pandas()
                
                # Add metadata columns
                meta = single_row_df[["question", "answer", "ground_truth", "retrieval_ms", "generation_ms"]].reset_index(drop=True)
                single_score_df = pd.concat([meta, single_score_df], axis=1)
                single_score_df.rename(columns=RENAME, inplace=True)
                
                new_scores_list.append(single_score_df)

                # Incremental checkpoint update
                if existing_df.empty and len(new_scores_list) == 1:
                    combined_current = single_score_df
                else:
                    combined_current = pd.concat([existing_df] + new_scores_list, ignore_index=True)
                
                combined_current.to_csv(checkpoint_path, index=False)

            except Exception as e:
                print(f"\n❌ Error evaluating sample '{row['question'][:40]}': {e}. Skipping row.")
                
                # Append dummy row with NaN scores so we don't get stuck on it forever
                meta = single_row_df[["question", "answer", "ground_truth", "retrieval_ms", "generation_ms"]].reset_index(drop=True)
                dummy_scores = pd.DataFrame([{col: pd.NA for col in RENAME.values()}])
                single_score_df = pd.concat([meta, dummy_scores], axis=1)
                new_scores_list.append(single_score_df)
                
                # Save progress up to now
                combined_current = pd.concat([existing_df] + new_scores_list, ignore_index=True)
                combined_current.to_csv(checkpoint_path, index=False)
                
                # Cool down
                await asyncio.sleep(5)
                continue

        if new_scores_list:
            scores_df = pd.concat([existing_df] + new_scores_list, ignore_index=True)
        else:
            scores_df = existing_df

    score_cols = [v for v in RENAME.values() if v in scores_df.columns]
    aggregate = {col: float(scores_df[col].mean()) for col in score_cols}

    print(f"\n✅  {pipeline_name} aggregate:")
    for k, v in aggregate.items():
        print(f"    {k:<22} {v:.4f}")

    return scores_df, aggregate

# ─────────────────────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────────────────────

def compare_results(vec_agg: dict, graph_agg: dict) -> pd.DataFrame:
    rows = []
    for metric in sorted(set(vec_agg) | set(graph_agg)):
        v = vec_agg.get(metric, float("nan"))
        g = graph_agg.get(metric, float("nan"))
        d = g - v
        arrow = "▲" if d > 0.005 else ("▼" if d < -0.005 else "─")
        rows.append({
            "Metric": metric,
            "VectorRAG": round(v, 4),
            "GraphRAG":  round(g, 4),
            "Δ (Graph−Vector)": round(d, 4),
            "Winner": f"{arrow} {'GraphRAG' if d > 0 else 'VectorRAG' if d < 0 else 'Tie'}",
        })
    cmp = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("  📊  VectorRAG vs GraphRAG — Comparison")
    print("=" * 70)
    print(cmp.to_string(index=False))
    return cmp

# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def export_results(vec_df, graph_df, cmp_df, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    vec_df.to_csv(run_dir / "vectorrag_per_query_eval.csv", index=False)
    graph_df.to_csv(run_dir / "graphrag_per_query_eval.csv", index=False)
    cmp_df.to_csv(run_dir / "ragas_summary_comparison.csv", index=False)
    print(f"\n💾  Results saved to: {run_dir.resolve()}")

# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

METRICS   = ["Faithfulness", "AnswerRelevancy", "ContextPrecision", "ContextRecall"]
C_VECTOR  = "#4C72B0"
C_GRAPH   = "#DD8452"


def plot_grouped_bar(vec_agg, graph_agg, run_dir):
    ms = [m for m in METRICS if m in vec_agg or m in graph_agg]
    x = np.arange(len(ms)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bv = ax.bar(x - w/2, [vec_agg.get(m, 0) for m in ms], w, label="VectorRAG", color=C_VECTOR, alpha=0.9)
    bg = ax.bar(x + w/2, [graph_agg.get(m, 0) for m in ms], w, label="GraphRAG",  color=C_GRAPH,  alpha=0.9)
    for bar in list(bv) + list(bg):
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(ms, fontsize=11)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score"); ax.legend()
    ax.set_title("VectorRAG vs GraphRAG — RAGAS Metrics", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(run_dir / "comparison_bar_chart.png", dpi=150); plt.close(fig)


def plot_radar(vec_agg, graph_agg, run_dir):
    ms = [m for m in METRICS if m in vec_agg or m in graph_agg]
    N = len(ms)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
    v_vals = [vec_agg.get(m, 0) for m in ms] + [vec_agg.get(ms[0], 0)]
    g_vals = [graph_agg.get(m, 0) for m in ms] + [graph_agg.get(ms[0], 0)]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, v_vals, "o-", lw=2, label="VectorRAG", color=C_VECTOR)
    ax.fill(angles, v_vals, alpha=0.15, color=C_VECTOR)
    ax.plot(angles, g_vals, "o-", lw=2, label="GraphRAG",  color=C_GRAPH)
    ax.fill(angles, g_vals, alpha=0.15, color=C_GRAPH)
    ax.set_thetagrids(np.degrees(angles[:-1]), ms, fontsize=11)
    ax.set_ylim(0, 1); ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("RAGAS Radar Chart", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout(); fig.savefig(run_dir / "radar_chart.png", dpi=150); plt.close(fig)


def plot_boxplots(vec_df, graph_df, run_dir):
    ms = [m for m in METRICS if m in vec_df.columns and m in graph_df.columns]
    fig, axes = plt.subplots(1, len(ms), figsize=(5*len(ms), 6))
    if len(ms) == 1: axes = [axes]
    for ax, m in zip(axes, ms):
        bp = ax.boxplot(
            [vec_df[m].dropna().tolist(), graph_df[m].dropna().tolist()],
            patch_artist=True, widths=0.5,
            medianprops=dict(color="white", linewidth=2),
        )
        bp["boxes"][0].set_facecolor(C_VECTOR)
        bp["boxes"][1].set_facecolor(C_GRAPH)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["VectorRAG", "GraphRAG"])
        ax.set_title(m, fontweight="bold"); ax.set_ylim(-0.05, 1.1)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-Sample Score Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(run_dir / "boxplot_distributions.png", dpi=150); plt.close(fig)


def plot_latency(vec_df, graph_df, run_dir):
    """Bar chart comparing mean retrieval + generation latency."""
    labels = ["VectorRAG", "GraphRAG"]
    ret_ms  = [vec_df["retrieval_ms"].mean(),  graph_df["retrieval_ms"].mean()]
    gen_ms  = [vec_df["generation_ms"].mean(), graph_df["generation_ms"].mean()]
    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, ret_ms, w, label="Retrieval",   color="#55a868", alpha=0.9)
    ax.bar(x + w/2, gen_ms, w, label="Generation",  color="#c44e52", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Latency (ms)"); ax.legend()
    ax.set_title("Mean Retrieval vs Generation Latency", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(run_dir / "latency_comparison.png", dpi=150); plt.close(fig)


def plot_all(vec_df, graph_df, vec_agg, graph_agg, run_dir):
    print("\n📈  Generating visualisations...")
    plot_grouped_bar(vec_agg, graph_agg, run_dir)
    plot_radar(vec_agg, graph_agg, run_dir)
    plot_boxplots(vec_df, graph_df, run_dir)
    plot_latency(vec_df, graph_df, run_dir)
    print(f"    Saved 4 charts to {run_dir.resolve()}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    # Dynamically isolate paths specifically for Ollama configurations
    suffix = "_ollama"
    if args.generations_dir == "generations":
        args.generations_dir = "generations_ollama"
    if args.output_dir == "results":
        args.output_dir = "results_ollama"

    if args.output_dir:
        CFG.output_dir = args.output_dir
    run_dir = CFG.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # Define persistent checkpoint paths under active output directory
    checkpoint_dir = Path(CFG.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    vec_chk = checkpoint_dir / f"checkpoint_vector_raw{suffix}.csv"
    graph_chk = checkpoint_dir / f"checkpoint_graph_raw{suffix}.csv"
    vec_scores_chk = checkpoint_dir / f"checkpoint_vector_scores{suffix}.csv"
    graph_scores_chk = checkpoint_dir / f"checkpoint_graph_scores{suffix}.csv"

    # Define generations folder and files
    generations_dir = Path(args.generations_dir)
    generations_dir.mkdir(parents=True, exist_ok=True)
    vec_gen_file = generations_dir / f"vectorrag_generations{suffix}.csv"
    graph_gen_file = generations_dir / f"graphrag_generations{suffix}.csv"

    # Step 1 & 2: Load and run generations if mode is 'generate' or 'all'
    if args.mode in ("generate", "all"):
        # 1. Load SQuAD questions + ground truths
        samples = load_squad_samples(args.squad_json, args.n_samples)
        print(f"\n📋  Loaded {len(samples)} evaluation samples.")

        # 2. Init generator LLM
        generator_llm = build_generator_llm()

        # 3. Build eval datasets by running live pipelines (with cache recovery)
        if args.pipeline in ("vector", "both"):
            vec_df   = await build_vector_dataset(samples, generator_llm, vec_chk)
            vec_df.to_csv(vec_gen_file, index=False)
        if args.pipeline in ("graph", "both"):
            graph_df = await build_graph_dataset(samples, generator_llm, graph_chk)
            graph_df.to_csv(graph_gen_file, index=False)
            
        print(f"\n💾  Raw generations successfully saved to: {generations_dir.resolve()}")

    # Step 3 & 4: Run evaluation if mode is 'evaluate' or 'all'
    if args.mode in ("evaluate", "all"):
        print("\n⚖️  Starting RAGAS Evaluation Phase...")
        
        # Load generated files
        if args.pipeline in ("vector", "both"):
            if not vec_gen_file.exists():
                print(f"❌ Error: VectorRAG generations not found in {generations_dir.resolve()}! Please run in '--mode generate' first.")
                return
            vec_df = pd.read_csv(vec_gen_file)
            vec_df["contexts"] = vec_df["contexts"].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else [x]
            )

        if args.pipeline in ("graph", "both"):
            if not graph_gen_file.exists():
                print(f"❌ Error: GraphRAG generations not found in {generations_dir.resolve()}! Please run in '--mode generate' first.")
                return
            graph_df = pd.read_csv(graph_gen_file)
            graph_df["contexts"] = graph_df["contexts"].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else [x]
            )

        # Init judge LLM + RAGAS wrappers
        _, ragas_llm = build_judge_llm()
        ragas_emb = build_embeddings()
        metrics = build_metrics(ragas_llm, ragas_emb)

        # Run RAGAS evaluation (with incremental resume checkpoint)
        t0 = time.perf_counter()
        if args.pipeline in ("vector", "both"):
            vec_scores_df, vec_agg = await run_ragas_eval(vec_df, "VectorRAG", metrics, vec_scores_chk)
        if args.pipeline in ("graph", "both"):
            graph_scores_df, graph_agg = await run_ragas_eval(graph_df, "GraphRAG", metrics, graph_scores_chk)
        print(f"\n⏱️   Total RAGAS eval time: {time.perf_counter()-t0:.1f}s")

        # Compare, Export & Visualise
        if args.pipeline == "both":
            cmp_df = compare_results(vec_agg, graph_agg)
            export_results(vec_scores_df, graph_scores_df, cmp_df, run_dir)
            plot_all(vec_scores_df, graph_scores_df, vec_agg, graph_agg, run_dir)
        elif args.pipeline == "graph":
            run_dir.mkdir(parents=True, exist_ok=True)
            graph_scores_df.to_csv(run_dir / "graphrag_per_query_eval.csv", index=False)
            print(f"\n💾  GraphRAG results saved to: {run_dir.resolve()}")
        elif args.pipeline == "vector":
            run_dir.mkdir(parents=True, exist_ok=True)
            vec_scores_df.to_csv(run_dir / "vectorrag_per_query_eval.csv", index=False)
            print(f"\n💾  VectorRAG results saved to: {run_dir.resolve()}")

        # Clean up active checkpoints since everything finished successfully
        checkpoints_to_clean = []
        if args.pipeline in ("vector", "both"):
            checkpoints_to_clean.extend([vec_chk, vec_scores_chk])
        if args.pipeline in ("graph", "both"):
            checkpoints_to_clean.extend([graph_chk, graph_scores_chk])
            
        for file in checkpoints_to_clean:
            if file.exists():
                try:
                    file.unlink()
                except Exception:
                    pass

        print(f"\n🎉  Benchmark evaluation complete! All outputs in: {run_dir.resolve()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS benchmark: VectorRAG vs GraphRAG")
    parser.add_argument("--mode", type=str, choices=["generate", "evaluate", "all"], default="all",
                        help="Benchmark phase to execute: 'generate' to query database & answer; "
                             "'evaluate' to judge answers with Ragas; 'all' to do both.")
    parser.add_argument("--pipeline", type=str, choices=["vector", "graph", "both"], default="both",
                        help="RAG pipeline to evaluate: 'vector', 'graph', or 'both' (default: 'both').")
    parser.add_argument("--squad-json", type=str, default="test_samples_300vec.json",
                        help="Path to SQuAD JSON file (default: test_samples_300vec.json).")
    parser.add_argument("--n-samples", type=int, default=8,
                        help="Number of questions to evaluate (default: 8).")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Root output directory for evaluation reports (default: results/).")
    parser.add_argument("--generations-dir", type=str, default="generations",
                        help="Directory to store intermediate query generation outputs (default: generations/).")
    args = parser.parse_args()
    asyncio.run(main(args))
